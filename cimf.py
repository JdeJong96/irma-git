import os
import sys
import time
import glob
import argparse
import cfgrib
import pygrib
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime
import cmocean
import warnings
import scipy
import ipywidgets as widgets
import cartopy.crs as ccrs
import numpy as np
from numba import jit, float32, float64, guvectorize, warnings
from ddxddy import ddx, ddy, ddxND, ddyND
import irmaFn as irma

# constants
g = 9.81 # gravity
a = 6371000 # radius earth
fc = 4.25e-5 # coriolis
p_ref = 100000
cp = 1004
Rdry = 286.9
kappa = Rdry/cp

# directories
dataloc = "../Data/LambertGrid/629x989interped/"
#cdataloc = "../Data/PolarGrid/200/" # for azimuthal averages
#cdatalocnobg = "../Data/PolarGrid/200nobg/" # + wind w.r.t. moving ref. frame
files = sorted(glob.glob(dataloc+"fc2017090512+???.nc"))
#cfiles = sorted(glob.glob(cdataloc+"fc2017090512+???.nc"))
#cfilesnobg = sorted(glob.glob(cdatalocnobg+"fc2017090512+???.nc"))

# settings (can be overridden by command line arguments)
verbosity = 1
method = "lagrange2d" # ['euler','lagrange2d','lagrange3d'] (see mass_continuity())
steady = False # set mass tendency to 0
use_sigma = False # deprecated (keep False)
        

def strtime():
    """Print time since start of simulation"""
    try:
        s = round(time.time() - startTime)
        return f"{s//3600:>02}:{s%3600//60:>02}:{s%3600%60:>02}"
    except NameError: # if startTime unknown, just print time
        return time.strftime("%H:%M:%S",time.localtime())


def vprint(v, msg, *args):
    """Print message depending on verbosity level"""
    if verbosity >= v:
        print(strtime()+": "+msg, *args)
    return


def latlon2dxdy(lats, lons):
    """Calculate meridional and zonal grid sizes"""
    vprint(1,"latlon2dxdy()")
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)
    if 'y' in lats.dims and 'x' in lons.dims:
        dy = a * lats.differentiate('y')
        dx = a * np.cos(lats) * lons.differentiate('x')
    elif 'dy' in lats.dims and 'dx' in lons.dims:
        dy = a * lats.differentiate('dy')
        dx = a * np.cos(lats) * lons.differentiate('dx')
    else:
        errmsg = f'(y,x) or (dy,dx) not in dimensions: {lats.dims}, {lons.dims}'
        raise NotImplementedError(errmsg)
    return dy, dx
    
    
def haversine_formula(lat1, lon1, lat2, lon2):
    """Return distance between two coordinate pairs"""
    vprint(2,"haversine_formula()")
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    h1 = np.sin((lat2-lat1)/2)**2
    h2 = np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2*a*np.arcsin(np.sqrt(h1+h2))


def get_angle(lat1, lon1, lat2, lon2):
    """Return angle of line from lat1lon1 to lat2lon2 w.r.t. east at lat1lon1"""
    vprint(2,"get_angle()")
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dx = a * (lon2 - lon1) * np.cos((lat1+lat2)/2)
    dy = a * (lat2 - lat1)
    angle = np.arctan2(dy,dx)
    angle = angle%(2*np.pi)
    return angle
    

def distance(lats1, lons1, lats2, lons2):
    """Calculate distance between any broadcastable pairs of coordinates"""
    vprint(1,"distance()")
    assert all([isinstance(arg, xr.DataArray) for arg in [lats1,lons1,lats2,lons2]])
    assert lats1.dims == lons1.dims
    assert lats2.dims == lons2.dims
    lats2 = lats2.broadcast_like(lats1).compute()
    lons2 = lons2.broadcast_like(lons1).compute()
    lats1 = lats1.broadcast_like(lats2).compute()
    lons1 = lons1.broadcast_like(lons2).compute()
    dist = xr.apply_ufunc(haversine_formula, lats1, lons1, lats2, lons2)
    angle = xr.apply_ufunc(get_angle, lats1, lons1, lats2, lons2)
    return dist, angle


def toPolar(lats, lons, xc, yc, dr=5e3, dt=2, rmax=1000e3):
    vprint(1,"toPolar()")
    assert all([type(x) in (xr.DataArray, xr.Dataset) for x in [lats, lons, xc, yc]])
    
    # set up new coordinates
    r = np.arange(0,rmax+dr,dr)
    r_bnds = np.array([(max(0,(rr-dr/2)),(rr+dr/2)) for rr in r])
    t = np.arange(0,360,dt)
    t_bnds = np.array([((tt-dt/2),(tt+dt/2)) for tt in t])%360/180*np.pi
    
    if 'y' in lats.dims and 'x' in lats.dims:
        latc = lats.isel(y=yc,x=xc); lonc = lons.isel(y=yc,x=xc)
    elif 'dy' in lats.dims and 'dx' in lats.dims:
        latc = lats.sel(dy=0,dx=0); lonc = lons.sel(dy=0,dx=0)
    else:
        raise ValueError(f'Wrong horizontal dimensions {lats.dims}, should be (y,x) or (dy,dx)')
    dist, angle = distance(latc, lonc, lats, lons)
    return dist, angle


@guvectorize(
    "(float32[:], float32, float32[:])",
    "(z), () -> (z)",
    nopython=True,
)
def calc_dp_gufunc(pres, slp, dp):
    '''Calculates the 1D vertical pressure difference along the
    vertical dimension 'z' as the difference between the current level
    and the one below it, or surface pressure if the lower level 
    lies below the earth's surface. 
    Levels below ground are filled with NaN.
    
    Parameters:
    -----------
        pres : float32(n)
            Pressure in Pa on isentropic levels.
        slp : float32()
            Surface pressure in Pa

    Returns:
    --------
        dp : float32(n)
            Vertical pressure difference
    '''
    above_ground = False
    num = len(pres)
    i = 0
    while not above_ground and i < num:
        if np.isnan(pres[i]):
            dp[i] = np.nan
        else:
            dp[i] = pres[i] - slp
            above_ground = True
        i = i + 1
    for j in range(i,num):  # above ground
        assert not np.isnan(pres[i]), 'Invalid pressure data above ground'
        dp[i] = pres[i] - pres[i-1]
        i = i + 1


def calc_dp(ds):
    '''Apply calc_dp_gufunc() on any xr.Dataset instance.
    
    Parameters:
    -----------
        ds : xr.Dataset
            Dataset instance that must contain the pressure on
            isentropic levels 'pres' and surface pressure 'p0m'.
            Must have vertical dimension 'z'.
    
    Returns:
    --------
        dp : xr.DataArray like ds.pres
            DataArray instance with vertical pressure difference.
                        
    '''
    vprint(1,"calc_dp(ds)")
    assert 'z' in ds.dims, f'Dimension "z" not found in dims : {dict(ds.dims)}.'
    assert 'pres' in ds.variables, 'Variable "pres" not found in Dataset.'
    assert 'p0m' in ds.variables, 'Variable "p0m" not found in Dataset.'
    if 'z' not in ds.pres.dims:
        ds['pres'] = ds.pres.swap_dims(theta='z')
    dp = xr.apply_ufunc(
        calc_dp_gufunc,  
        ds.pres,  
        ds.p0m, 
        input_core_dims=[['z'], []], 
        output_core_dims=[['z']], 
        dask="parallelized",
        output_dtypes=[ds.pres.dtype],
    ).compute().transpose(*ds.pres.dims)
    return dp

isnone = lambda x : "=None" if x is None else "!=None"

@guvectorize(
    "(float32[:], float32, float32[:])",
    "(z), () -> (z)", 
    nopython=True,
)
def _iwind2(v, v10, vi):
    '''Calculate the wind at grid interfaces for a column of air.
    
    The grid interface wind is estimated by averaging the wind
    on adjacent levels. Below the lowest atmospheric level, the
    boundary layer wind is estimated by the mean of that level and
    the 10m wind.
    '''
    below_surface = True
    num = len(v)
    for i in range(num):
        if below_surface and (not np.isnan(v[i])):
            vi[i] = (v[i] + v10)/2
            below_surface = False
        vi[i] = (v[i] + v[i-1])/2  
    
def iwind2(ds):
    '''Apply the 1d function iwind2 on multidimensional data
    
    Parameters:
    -----------
    ds : xr.Dataset
        Input dataset
        
    Returns:
    --------
    ui : xr.DataArray
        Zonal wind at grid cell interfaces
    vi : xr.DataArray
        Meridional wind at grid cell interfaces
    '''
    vprint(1,"iwind2")
    ui = xr.apply_ufunc(
        _iwind2,  
        ds.u,
        ds.u10,
        input_core_dims=[['z'],[]], 
        output_core_dims=[['z']], 
        dask="parallelized",
        output_dtypes=[ds.u.dtype],
    ).compute().assign_attrs(ds.u.attrs)
    vi = xr.apply_ufunc(
        _iwind2,  
        ds.v,
        ds.v10,
        input_core_dims=[['z'],[]], 
        output_core_dims=[['z']], 
        dask="parallelized",
        output_dtypes=[ds.v.dtype],
    ).compute().assign_attrs(ds.v.attrs)
    return ui, vi

    
def iwind(v,v10=None):
    '''Average wind halfway between adjacent vertical levels,
    or between lowest and surface level.
    
    Parameters:
    -----------
        v : xr.DataArray
            horizontal wind component (zonal or meridional)
        v10 : xr.DataArray, optional
            horizontal 10m wind component (zonal or meridional)
        
    Returns:
    --------
        vi : xr.DataArray
            average wind at vertical grid interface.
    '''
    vprint(1,"iwind()")
    vi = v.rolling(z=2).mean() # mean over adjacent levels
    if v10 is not None: 
        z0 = v.notnull().argmax('z').compute()  # index 1st level above ground
        vbl = (v.isel(z=z0) + v10)/2  # mean over lowest level and 10m wind
        vi = vi.where(vi.z!=z0, vbl)
    return vi.astype(v.dtype)


@guvectorize(
    ["(float32[:],float64[:],float32[:])",
     "(float64[:],float64[:],float64[:])"], 
    "(z),(z)->(z)"
)
def _ddthetagufunc(v, theta, dvdtheta):
    i0 = 0
    above_ground = ~np.isnan(v[i0])
    while not above_ground:
        dvdtheta[i0] = np.nan
        i0 = i0 + 1
        above_ground = ~np.isnan(v[i0])
    dvdtheta[i0] = (v[i0+1] - v[i0])/(theta[i0+1] - theta[i0])
    assert not np.isnan(v[i0])
    if i0 > 0:
        assert np.isnan(v[i0-1])
    for i in range(i0+1,len(theta)-1):
        dvdtheta[i] = (v[i+1]-v[i-1])/(theta[i+1]-theta[i-1])    
    dvdtheta[-1] = (v[-1] - v[-2])/(theta[-1] - theta[-2])        
        

def ddtheta(da):
    dim = 'z' if 'z' in da.dims else 'theta'
    ddadtheta = xr.apply_ufunc(
        _ddthetagufunc,
        da,
        da.theta,
        dask="parallelized",
        input_core_dims=[[dim],[dim]],
        output_core_dims=[[dim]],
    )
    return ddadtheta


def masstendency(ds, mass_column=None):
    '''Evaluate the mass tendency at each isentropic level
    
    Parameters:
    -----------
        ds : xr.Dataset
            Dataset to calculate mass tendency on
        mass_column : xr.DataArray
            total mass per m**2 within isentropes (either -dp/g or
            sigma*dtheta)

    Returns:
    --------
        T : xr.DataArray
            Mass tendency
    '''
    vprint(1,"masstendency()")
    if 'z' not in ds.dims:
        ds = ds.swap_dims(theta='z')
    if mass_column is not None:
        pass
    elif use_sigma:
        sigma = (-ds.dtdp**(-1)/g).chunk(ds.dtdp.shape)
        mass_column = sigma * ds.theta.differentiate('z')
    else:
        mass_column = -calc_dp(ds)/g
    
    if method in ['euler','lagrange2d']:
        T = mass_column.differentiate('valid_time', datetime_unit='s')
        T = T.assign_attrs(name='T', 
                           long_name='mass tendency within levels', 
                           units='kg m**-2 s**-1')
    elif method == 'lagrange3d':
        raise NotImplementedError("method 'lagrange3d' is not yet supported")
    else:
        raise ValueError(f'Invalid method argument: {method}')
    return T


def masscontinuity(ds, T=None):
    '''Evaluate the the vertical mass flux differences at each 
    isentropic level required to satisfy the continuity equation.
    
    Parameters:
    -----------
        ds : xr.Dataset
            Dataset containing isentropic zonal wind 'u', meridional
            wind 'v' and pressure 'pres'. May be a single-timestamp
            dataset if T is provided at matching time. 
            If present, also may use 10m zonal wind 'u10', meridional
            wind 'v' and isentropic density 'sigma'.
        T : xr.DataArray, optional
            Tendency of mass within each grid volume dxdydtheta (see 
            use_sigma for details). If None, T will be computed auto-
            matically using central time derivatives at all time stamps
            except the first and last, where forward and backward
            derivatives are used. For steady state results, use 
            T = xr.zeros_like(ds.pres). 
        
    Returns:
    --------
        dcimf: xr.DataArray
            Vertical difference in cross-isentropic mass flux
            
    Raises:
    -------
        AssertionError:
            if z is not a dimension in ds
        TypeError: 
            if ds has one time stamp while T is not provided
    '''
    vprint(1,f"masscontinuity({method})")
    err_msg = f'Dimension "z" not found in dims : {dict(ds.dims)}.'
    assert 'z' in ds.dims, err_msg
    if T is None and ds.valid_time.size == 1:
        raise TypeError('Cannot determine T for single step file')
    elif isinstance(T, int) and T==0:
        T = xr.zeros_like(ds.pres)
    elif (T is not None) and ('valid_time' in ds.dims):
        ds = ds.sel(valid_time=T.valid_time)   
    dx, dy = latlon2dxdy(ds.latitude, ds.longitude) # grid size
    u_bg, v_bg = translational_velocity(ds)
    if use_sigma:
        sigma = (-ds.dtdp**(-1)/g).chunk(ds.dtdp.shape)
        dtheta = ds.theta.differentiate('z').astype(sigma.dtype)
        dx = dx.astype(sigma.dtype); dy = dy.astype(sigma.dtype)
        T = masstendency(ds, sigma*dtheta) if T is None else T
        FX = sigma*dtheta*dy*ds.u
        FY = sigma*dtheta*dx*ds.v
    else:
        dp = calc_dp(ds)
        dx = dx.astype(dp.dtype); dy = dy.astype(dp.dtype)
        T = masstendency(ds, -dp/g) if T is None else T
        ui, vi = iwind2(ds)
        vprint(1,"FX, FY in masscontinuity()")
        FX = -1/g*dy*dp*(ui-u_bg) 
        FY = -1/g*dx*dp*(vi-v_bg)
    vprint(1,"ddx, ddy in masscontinuity()")
    dFXdx = ddxND(FX) if hasattr(FX, 'longitude') else ddxND(FX, ds.latitude, ds.longitude)
    dFYdy = ddyND(FY) if hasattr(FX, 'latitude') else ddyND(FY, ds.latitude, ds.longitude)
    dcimf = T - dFXdx/dy - dFYdy/dx
    dcimf = dcimf.assign_attrs(name='dcimf',
                               long_name=('vertical cross-isentropic mass flux'
                                          +' difference between levels'),
                               units='kg m**-2 s**-1',
                               method='sigma' if use_sigma else 'dp')
    return dcimf


@guvectorize(
    "(float32[:], float32[:])",
    "(n) -> (n)",
    nopython=True,
)
def integrate_dcimf_gufunc(dcimf, cimf):
    '''Adds 1D vertical cross-isentropic mass flux differences defined
    on each isentropic level with respect to the level below, or the
    surface for the first level above ground. From model top downward.

    Parameters:
    -----------
        dcimf : float32(n)
            Vertical difference in upward cross-isentropic mass flux.

    Returns:
    --------
        cimf : float32(n)
            Upward cross-isentropic mass flux
    '''
    imax = len(dcimf) - 1
    cimf[imax] = 0
    for i in range(imax,0,-1):
        cimf[i-1] = cimf[i] - dcimf[i]        

        
def integrate_dcimf(dcimf):
    '''Apply integrate_dcimf_gufunc() on dcimf.
    
    Parameters:
    -----------
        dcimf : xr.DataArray
            Vertical cross-isentropic mass flux differences
            Must have vertical dimension 'z'
    
    Returns:
    --------
        cimf : xr.DataArray
            Upward cross-isentropic mass flux
    
    Raises:
    -------
        AssertionError:
            if z is not a dimension in ds
    '''
    vprint(1,"integrate_dcimf()")
    err_msg = f'Dimension "z" not found in dims : {dcimf.dims}.'
    assert 'z' in dcimf.dims, err_msg
    cimf = xr.apply_ufunc(
        integrate_dcimf_gufunc,  
        dcimf.astype('float32'),  
        input_core_dims=[['z']], 
        output_core_dims=[['z']], 
        dask="parallelized",
        output_dtypes=['float32'],
    ).compute().assign_attrs(dcimf.attrs)
    cimf.attrs.update(name='cimf', long_name='cross-isentropic mass flux')
    return cimf


def calc_cimf(ds, **kwargs):
    '''Calculate cross-isentropic mass flux in ds
    
    Parameters:
    -----------
        ds : xr.Dataset
            Input dataset
        kwargs : optional arguments
            T : xr.DataArray
                mass tendency per grid height per m**2
    
    Returns:
    --------
        cimf : xr.DataArray
            Upward cross-isentropic mass flux [kg m**-2 K**-1]
    '''
    vprint(1, "calc_cimf()")
    printkwargs = {k:(v.shape if hasattr(v,"shape") else v) for k,v in kwargs.items()}
    if 'z' not in ds.dims:
        ds = ds.swap_dims(theta='z')
    dcimf = masscontinuity(ds, **kwargs)
    cimf = integrate_dcimf(dcimf).assign_coords({'theta':ds.theta})
    return cimf


def check_cimf(ds, cimf, use_sigma=False):
    '''Check if mass is conserved'''
    vprint(1,"check_cimf()")
    if use_sigma:
        sigma = -ds.dtdp**(-1)/g
        w = cimf/sigma
    else:
        dp = calc_dp(ds).isel(z=slice(1,None))
        dtheta = ds.theta.diff('z')
        sigma = -1/g*dp/dtheta
        w = cimf.isel(z=slice(1,None))/sigma
        div = (ddx(ds.u.isel(z=slice(1,None)))
               + ddy(ds.v.isel(z=slice(1,None)))
               + ddtheta(w))
    return div


def crop(ds, names='all', d=150, verbose=False):
    """Crop dataset horizontally around the pressure centre of the tropical
    cyclone given by ds.xc and ds.yc
    
    Parameters:
    -----------
    ds : xr.Dataset
        dataset from which to crop
    names : list of str
        list of variable names to crop
    d : int, optional
        amount of grid points around TC centre to crop
    
    Returns:
    -------
    new_ds : xr.Dataset
        dataset with cropped data variable and corresponding latitude and 
        longitude as new datavariables. The new coordinates are dx and dy,
        the amount of grid cells from the TC centre.
    """
    vprint(1,"crop()")
    ds = ds.transpose(...,'y','x')
    new_coords = {
        'dy':('dy', range(-d,d+1), {'long_name':'number of grid points from centre meridionally'}), 
        'dx':('dx', range(-d,d+1), {'long_name':'number of grid points from centre zonally'})
    }
    
    # set list of variables to crop and variables to copy (those with no x,y dep.)
    copynames = None
    if names == 'all':
        has_xydim = [{'x','y'}.issubset(ds[var].dims) for var in ds.variables]
        names = [var for var,hasxy in zip(ds.variables,has_xydim) if hasxy]
        copynames = [var for var,hasxy in zip(ds.variables,has_xydim) if not hasxy]
    
    # add latitude and longitude to variables to crop
    names = list(names)
    names = names + ['latitude'] if ('latitude' not in names) else names
    names = names + ['longitude'] if ('longitude' not in names) else names
    ds['latitude'] = ds['latitude'].astype('float32').broadcast_like(ds.yc)
    ds['longitude'] = ds['longitude'].astype('float32').broadcast_like(ds.xc)
    ds = ds.reset_coords(['latitude','longitude'])

#     # fasther method (unfortunately does not work yet)
#     @guvectorize(
#     "(y,x),(),(),(),(dy,dx) -> (dy,dx)",
#     nopython=True)
#     def guselect_data2d(da, yc, xc, d, dummy, result):
#         print(da.shape,yc,xc,d,dummy.shape,result.shape)
#         result = np.full((2*d+1,2*d+1), np.nan, da.dtype)
#         yslice = slice(max(0,yc-d), min(yc+d+1,da.shape[0]))
#         xslice = slice(max(0,xc-d), min(xc+d+1,da.shape[1]))
#         dyslice = slice(yslice.start-yc+d, yslice.stop-yc+d)
#         dxslice = slice(xslice.start-xc+d, xslice.stop-xc+d)
#         result[dyslice,dxslice] = da[yslice,xslice]
        

    def _crop2d(da, yc, xc, d):
        """Select data from da(y,x) up to d gridpoints of central point yc,xc"""
        result = np.full((2*d+1,2*d+1), np.nan, da.dtype)
        yslice = slice(max(0,yc-d), min(yc+d+1,da.shape[0]))
        xslice = slice(max(0,xc-d), min(xc+d+1,da.shape[1]))
        dyslice = slice(yslice.start-yc+d, yslice.stop-yc+d)
        dxslice = slice(xslice.start-xc+d, xslice.stop-xc+d)
        result[dyslice,dxslice] = da[yslice,xslice]
        return result
    
    # fill new dataset with cropped data
    new_ds = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
    new_ds[names] = xr.apply_ufunc(
        _crop2d, 
        ds[names], 
        ds.yc, 
        ds.xc, 
        d,
        input_core_dims=[['y','x'],[],[],[]],
        output_core_dims=[['dy','dx']],
        exclude_dims=set(('y','x')),
        vectorize=True,
        dask='parallelized',
        output_dtypes=['float32'],
        dask_gufunc_kwargs={'output_sizes':{'dy':2*d+1,'dx':2*d+1}}
    ).compute().assign_coords(new_coords)
    if copynames is not None:
        new_ds = new_ds.assign(ds[copynames])
    for name in new_ds.variables:
        try:
            new_ds[name].attrs = ds[name].attrs
        except KeyError:
            pass
    return new_ds


def translational_velocity(ds):
    """Calculate the translational velocity based on center coordinates"""
    vprint(1,"translational_velocity()")
    if method == 'euler':
        return (0,0)
    if 'valid_time' not in ds.dims:
        print("Cannot determine translational velocity"
              +" from single time stamp")
        return None
    elif method == 'lagrange2d':
        latcs = np.deg2rad(ds.latitude.sel(dx=0,dy=0))
        loncs = np.deg2rad(ds.longitude.sel(dx=0,dy=0))
        dlatdt = latcs.differentiate('valid_time', datetime_unit='s')
        dlondt = loncs.differentiate('valid_time', datetime_unit='s')
        u_bg = a * np.cos(latcs) * dlondt
        v_bg = a * dlatdt
        return (u_bg, v_bg)
    elif method == 'lagrange3d':
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown method {method}')
        

def transform_vector(Vx, Vy, angle):
    """Calculate radial and tangential component of vector"""
    vprint(1,'transform_vector()')
    V = np.sqrt(Vx**2 + Vy**2)
    Vdir = np.arctan2(Vy, Vx)
    Vrad = V * np.cos(Vdir - angle)
    Vtan = V * np.sin(Vdir - angle)
    return Vrad, Vtan
    
def convert_wind(ds, angle=None):
    """Convert zonal/meridional wind to radial and tangential wind"""
    vprint(1, 'convert_wind()')
    try:
        u = ds.u
        v = ds.v
    except AttributeError:
        if ('u_rad' in ds) and ('v_tan' in ds):
            vprint(1, 'convert_wind(): using existing u_rad and v_tan')
            return ds
        else:
            raise
    if angle is None:
        _,angle = toPolar(ds.latitude, ds.longitude, ds.xc, ds.yc)
    u_bg, v_bg = translational_velocity(ds) if 'valid_time' in ds.xc.dims else (0,0)
    u_rad, v_tan = transform_vector(ds.u-u_bg, ds.v-v_bg, angle)
    u_rad.attrs = {'long_name':'radial velocity','units':'m s**-1'}
    v_tan.attrs = {'long_name':'tangential velocity','units':'m s**-1'}
    return ds.assign({'u_rad':u_rad, 'v_tan':v_tan})


def cartesian_gradient(ds,var,comp='all'):
    vprint(1, 'cartesian_gradient()')
    if isinstance(var, str):
        var = ds[var]
    dvdx = None
    dvdy = None
    dvdz = None
    if comp in ['all','x']:
        dvdx = ddxND(var) if hasattr(var, 'longitude') else ddxND(var, ds.latitude, ds.longitude)
        dvdx = dvdx.chunk({'valid_time':len(dvdx.valid_time)})
    if comp in ['all','y']:
        dvdy = ddyND(var) if hasattr(var, 'latitude') else ddyND(var, ds.latitude, ds.longitude)
        dvdy = dvdy.chunk({'valid_time':len(dvdy.valid_time)})
    if comp in ['all','z']:
        dvdz = ddtheta(var)/ddtheta(ds.height)
        dvdz = dvdz.chunk({'valid_time':len(dvdz.valid_time)})
        #dvdz = var.differentiate('theta')/ds.height.differentiate('theta')
    return dvdx, dvdy, dvdz
    

def vorticity_flux(ds, angle=None):
    """Calculate vorticity flux in cartesian coordinates and return radial component"""
    vprint(1,'vorticity_flux()')
    if use_sigma:
        sigma = -1/g * ds.dtdp**(-1)
        dthetadt = ds.cimf/sigma
    else:
        vprint(1,'Calculating dthetadt with (slightly) misaligned dp and sigma.')
        dp = calc_dp(ds)
        if 'z' not in ds.dims:
            ds = ds.swap_dims(theta='z')
        dth = ds.theta.differentiate('z')
        sigma = -1/g*dp/dth
        dthetadt = ds.cimf/sigma
    rho = ds.pres / (Rdry * ds.t)
    wc = ds.cimf/rho # (cartesian) vertical velocity in m/s
    u_bg, v_bg = translational_velocity(ds)
    dzdx, dzdy, dzdz = cartesian_gradient(ds, ds.height)
    dudx, dudy, dudz = cartesian_gradient(ds, ds.u-u_bg)
    dvdx, dvdy, dvdz = cartesian_gradient(ds, ds.v-v_bg)
    dwdx, dwdy, dwdz = cartesian_gradient(ds, dthetadt) # w is shorthand for dthetadt
    dwcdx, dwcdy, dwcdz = cartesian_gradient(ds, wc)
    #dvdx = ddxND(ds.v) if hasattr(ds.v, 'longitude') else ddxND(ds.v, ds.latitude, ds.longitude)
    #dudy = ddyND(ds.u) if hasattr(ds.u, 'latitude') else ddyND(ds.u, ds.latitude, ds.longitude)
    av = (dvdx - dudy + fc).transpose(*ds.u.dims)
    av = xr.DataArray(av, coords=ds.u.coords, dims=ds.u.dims, name='eta')
    try:
        dudth = ddtheta(ds.u - u_bg)#.differentiate('theta')
        dvdth = ddtheta(ds.v - v_bg)#.differentiate('theta')
        dzdth = ddtheta(ds.height)#.differentiate('theta')
    except ValueError:
        vprint(1,'Could not calculate diabatic vorticity flux, set to 0')
        dudth = xr.zeros_like(ds.u)
        dvdth = xr.zeros_like(ds.v)
        dzdth = xr.zeros_like(ds.height)
    Jh_x = dthetadt * dvdth
    Jh_y = -dthetadt * dudth
    l_sq = 100*ds.Ke**2 / ds.tke # square of turbulent length scale
    lapu = dudx**2 + dudy**2 + dudz**2
    lapv = dvdx**2 + dvdy**2 + dvdz**2
    lapw = dwdx**2 + dwdy**2 + dwdz**2
    lapwc = dwcdx**2 + dwcdy**2 + dwcdz**2
    #l_sq = 2 * ds.tke / (gradu + gradv + gradwc)
    uu = l_sq * lapu
    vv = l_sq * lapu
    ww = l_sq * lapu
    wcwc = l_sq * (dwcdx**2 + dwcdy**2 + dwcdz**2)
    uv = l_sq * (dudx*dvdx + dudy*dvdy + dudz*dvdz)
    uw = l_sq * (dudx*dwdx + dudy*dwdy + dudz*dwdz)
    vw = l_sq * (dvdx*dwdx + dvdy*dwdy + dvdz*dwdz)
    duvdx,_,_ = cartesian_gradient(ds, uv, 'x')
    _,dvvdy,_ = cartesian_gradient(ds, vv, 'y')
    _,_,dvwdz = cartesian_gradient(ds, vw, 'z')
    duudx,_,_ = cartesian_gradient(ds, uu, 'x')
    _,duvdy,_ = cartesian_gradient(ds, uv, 'y')
    _,_,duwdz = cartesian_gradient(ds, uw, 'z')
    Je_x = duvdx + dvvdy + dvwdz
    Je_y = -(duudx + duvdy + duwdz)
    if angle is None:
        _,angle = toPolar(ds.latitude, ds.longitude, ds.xc, ds.yc)
    Ja_rad, _ = transform_vector((ds.u-u_bg)*av, (ds.v-v_bg)*av, angle)
    Jh_rad, _ = transform_vector(Jh_x, Jh_y, angle)
    Je_rad, _ = transform_vector(Je_x, Je_y, angle)
    av.attrs = {'long_name':'absolute vorticity', 'units':'s**-1'}
    sigma.attrs = {'long_name':'isentropic density', 'units':'kg m**-2 K**-1'}
    Ja_rad.attrs = {'long_name':'advective component radial vorticity flux','units':'m s**-2'}
    Jh_rad.attrs = {'long_name':'diabatic component radial vorticity flux','units':'m s**-2'}
    Je_rad.attrs = {'long_name':'eddy component radial vorticity flux','units':'m s**-2'}
    ds = ds.assign({'l_sq':l_sq,'uu':uu,'vv':vv,'ww':ww,'wcwc':wcwc})
    return ds.assign({'eta':av, 'sigma':sigma, 'Ja':Ja_rad, 'Jh':Jh_rad, 'Je':Je_rad})


# # Old 
# def azimean(ds, dr=5e3, rmax=1000e3):
#     vprint(1,"azimean()")
#     if 'theta' in ds.dims:
#         ds = ds.swap_dims(theta='z')
#     if 'hybrid' in ds.dims:
#         ds = ds.swap_dims(hybrid='z')
#     r = np.arange(0,rmax,dr)
#     r_bnds = np.array([0,*(r+dr/2)])
#     dist, angle = toPolar(ds.latitude, ds.longitude, ds.xc, ds.yc)
#     ds = ds.assign_coords({'r':dist})
#     ds = convert_wind(ds, angle)
#     ds = vorticity_flux(ds)
#     with dask.config.set(**{'array.slicing.split_large_chunks': False}):
#         dsgroupby = ds.groupby_bins(
#             'r',r_bnds,labels=r/1000,include_lowest=True
#         )
#         stacked_dim = [dim for dim in dsgroupby.dims if 'stacked' in dim]
#         ds = dsgroupby.mean(stacked_dim).rename({'r_bins':'r'}).transpose(...,'z','r')
#     ds.r.attrs = {'long_name':'radius', 'units':'km'}
#     return ds



# OLD 2
# @guvectorize([
#     'void(float32[:,:], int64[:], int64[:], float64[:], float64[:], float32[:], float32[:])',
#     'void(float64[:,:], int64[:], int64[:], float64[:], float64[:], float64[:], float64[:])'],
#     '(y, x),(ncol),(rb),(r),(rb)->(r),(r)', nopython=False)
# def azimean_gufunc(arr, idsin, splitids, rnew, rnew_bnds, res, nanratio):
#     if arr.shape == (1,1):
#         return
#     arr_s = arr.flatten()[idsin]
#     groups = np.split(arr_s, splitids)[1:-1]
#     for i,grp in enumerate(groups):
#         print(i,grp)
#         if grp.size == 0:
#             res[i] = np.nan
#             nanratio[i] = 1
#         else:
#             isnan = np.isnan(grp)
#             res[i] = np.nanmean(grp) if (not np.all(isnan)) else np.nan
#             nanratio[i] = np.mean(isnan)


# def azimean(ds, rmax=1000e3, dr=5e3):
#     """Take azimuthal average of dataset
    
#     """
#     vprint(1,'azimean()')

#     # define new radial coordinate
#     r = np.arange(0,rmax,dr, dtype='float64')
#     ds = ds.assign_coords({'r':r})
#     ds['r'] = ds.r.assign_attrs({'long_name':'radius','units':'m'})
#     r_bnds = np.array([0,*(r+dr/2)], dtype='float64')
#     dist, angle = toPolar(ds.latitude, ds.longitude, ds.xc, ds.yc)
#     dist.attrs = {'long_name':'radial distance from centre','units':'m'}
#     ds['radius'] = dist.compute().astype('float32')
    
#     # calculate indices that sort by radius and split by radial bin
#     hdims = ('y','x') if {'y','x'}.issubset(ds.dims) else ('dy','dx')
#     dist = dist.stack({'ncol':hdims}).reset_index('ncol',drop=True).transpose(...,'ncol')
#     vprint(1, "calc ids()")
#     sortids = dist.argsort().compute()
#     splitids = xr.apply_ufunc(
#         np.searchsorted, dist.isel(ncol=sortids), r_bnds, 
#         input_core_dims=[['ncol'],['rb']], 
#         output_core_dims=[['rb']], vectorize=True)
    
#     # transform vector fields and take azimuthal mean
#     ds = convert_wind(ds, angle)
#     ds = vorticity_flux(ds)
#     vprint(1, "azimean_gufunc()")
#     hasxy = [da for da in ds.data_vars if set(hdims).issubset(ds[da].dims)]
#     ds[hasxy], nanratio = xr.apply_ufunc(
#         azimean_gufunc,
#         ds[hasxy],
#         sortids,
#         splitids,
#         r,
#         r_bnds,
#         input_core_dims=[hdims,['ncol'],['rb'],['r'],['rb']],
#         output_core_dims=[['r'],['r']],
#         dask='parallelized',
#         keep_attrs=True,
#     )
#     stillhasxy = [da for da in ds.variables if any([hdim in ds[da].dims for hdim in hdims])]
#     vprint(1, f"removing {stillhasxy} from azimuthal mean dataset")
#     ds = ds.drop(stillhasxy)
#     return ds.compute(), nanratio.compute()


@guvectorize([
    'void(float32[:,:], float32[:,:], int64[:], float64[:], float64[:], float32[:], float32[:])',
    'void(float64[:,:], float32[:,:], int64[:], float64[:], float64[:], float64[:], float64[:])'],
    '(y, x),(y,x),(ncol),(r),(rb)->(r),(r)', nopython=False)
def azimean_gufunc(arr, dist, sortids, rnew, rnew_bnds, res, nanratio):
    if arr.shape == (1,1):
        return
    arr_s = arr.flatten()[sortids]
    dist_s = dist.flatten()[sortids]
    groups = np.split(arr_s, np.searchsorted(dist_s,rnew_bnds))[1:-1]
    for i,grp in enumerate(groups):
        if grp.size == 0:
            res[i] = np.nan
            nanratio[i] = 1
        else:
            isnan = np.isnan(grp)
            res[i] = np.nanmean(grp) if (not np.all(isnan)) else np.nan
            nanratio[i] = np.mean(isnan)


def azimean(ds, dr=5e3):
    """Take azimuthal average of dataset
    
    """
    vprint(1,'azimean()')
    
    # determine horizontal extent of data
    dist, angle = toPolar(ds.latitude, ds.longitude, ds.xc, ds.yc)
    dist.attrs = {'long_name':'radial distance from centre','units':'m'}
    ds['radius'] = dist.compute().astype('float32')
    if {'y','x'}.issubset(ds.dims):
        hdims = ('y','x')
        rmax = ds.radius.isel(y=ds.yc).max('x').mean().data
    elif {'dy','dx'}.issubset(ds.dims):
        hdims = ('dy','dx')
        rmax = ds.radius.sel(dy=0).max('dx').mean().data
    else:
        raise NotImplementedError(f'Cannot take azimuthal average over dims {ds.dims}')
    
    # define new radial coordinate
    r = np.arange(0,rmax,dr, dtype='float64')
    ds = ds.assign_coords({'r':r})
    ds['r'] = ds.r.assign_attrs({'long_name':'radius','units':'m'})
    r_bnds = np.array([0,*(r+dr/2)], dtype='float64')
    
    # calculate indices that sort a horizontally flattened array by radius
    dist = dist.stack({'ncol':hdims}).reset_index('ncol',drop=True).transpose(...,'ncol')
    sortids = dist.argsort().compute()
    
    # transform vector fields and take azimuthal mean
    #vprint(1,"azimean: convert_wind()")
    ds = convert_wind(ds, angle)
    #vprint(1,"azimean: done.")
    try:
        vprint(1,"azimean: vorticity_flux()")
        ds = vorticity_flux(ds)
        vprint(1,"azimean: done.")
    except:
        vprint(1,"azimean: not calculating vorticity flux")
        pass
    vprint(1, "azimean_gufunc()")
    hasxy = [da for da in ds.data_vars if set(hdims).issubset(ds[da].dims)]
    ds[hasxy], nanratio = xr.apply_ufunc(
        azimean_gufunc,
        ds[hasxy],
        ds.radius,
        sortids,
        r,
        r_bnds,
        input_core_dims=[hdims,hdims,['ncol'],['r'],['rb']],
        output_core_dims=[['r'],['r']],
        dask='parallelized',
        keep_attrs=True,
    )
    stillhasxy = [da for da in ds.variables if any([hdim in ds[da].dims for hdim in hdims])]
    vprint(1, f"removing {stillhasxy} from azimuthal mean dataset")
    ds = ds.drop(stillhasxy).compute()
    nanratio = nanratio.compute()
    return ds, nanratio


def find_centre3d(ds, d=14, thresh=0.75):
    '''Find height dependent centre of tropical cyclone
    based on curl of wind direction.
    
    Within d grid points of the surface pressure centre of the TC,
    evaluate the cosines of the deviation of the wind direction on 
    zonally and meridionally adjacent grid cells with respect to a
    cylindrical flow. They are evaluated on a grid with 1/4th of the
    actual grid size using linearly interpolated winds. The point
    with a local maximum of the mean of these cosines is considered
    the vortex centre if the mean >= thresh, otherwise the centre
    will be inter-/extrapolated from other levels using a nearest
    neighbour approach.
    
    Parameters:
    -----------
    d : int
        search within d grid points of slp based centre
    thresh : (0 - 1) float
        minimum mean of cosines of wind direction deviations
        
    Returns:
    --------
    coords : dict('latitude':xr.DataArray, 'longitude':xr.DataArray)
        height dependent coordinates of TC centre
    '''
    vprint(1,"find_centre3d()")
    #print(f'{strtime()}: find_centre3d')
    # helper function
    def cos(da): 
        '''fill boundary and NaN's with pi/2, then take cosine'''
        da.loc[dict(x=[0,-1])] = np.pi/2
        da.loc[dict(y=[0,-1])] = np.pi/2
        da = da.fillna(np.pi/2)
        return np.cos(da)
    
  
    d += 1 # add extra grid length because the outer points will be lost
    # if 'valid_time' in ds.dims:
    #     for t in ds.valid_time:
    #         xc = ds.xc.sel(valid_time=t).data
    #         yc = ds.yc.sel(valid_time=t).data
    #         dom = dict(x=np.linspace(xc-d,xc+d,8*d+1),
    #                    y=np.linspace(yc-d,yc+d,8*d+1))
            
    dom = dict(x=np.linspace(ds.xc-d,ds.xc+d,8*d+1), # horizontal domain
               y=np.linspace(ds.yc-d,ds.yc+d,8*d+1)) # refined x4
        
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # do not show annoying FutureWarning
        ds = ds.interp(dom, method='linear')
        # ds = xr.apply_ufunc(ds.interp, 
        #                     dom, 
        #                     'linear', 
        #                     input_core_dims=[['y','x'],[]], 
        #                     output_core_dims=[['ynew','xnew']],
        #                     exclude_dims=set(['y','x']),
        #                     dask='parallelized')
        
    if 'theta' in ds.dims:
        ds = ds.swap_dims(theta='z')
    wdir = np.arctan2(ds.v, ds.u) # angle of wind vector w.r.t. east
    curl = 0.25 * (
        cos(wdir.shift(y=-1)+np.pi) +
        cos(wdir.shift(y=1)) + 
        cos(wdir.shift(x=-1)-np.pi/2) + 
        cos(wdir.shift(x=1)+np.pi/2)
    ).compute()
    clocs = curl.argmax(('y','x')) # indices of max curl
    mask1 = wdir.isnull().any(('x','y')) # no NaN values allowed
    mask2 = curl.sel(clocs) < thresh # curl must obey threshold
    clocs = {
        k: v.where(~(mask1 | mask2), np.nan)
        .interpolate_na('z','nearest',fill_value='extrapolate')
        .astype('int') for k, v in clocs.items()
    } # indices of max curl with masked entries nearest neighbour filled
    coords = {
        'latitude':ds.latitude.sel(clocs),
        'longitude':ds.longitude.sel(clocs)
    }
    return coords


def iterate_find_centre3d(ds):
    vprint(1,"iterate_find_centre3d()")
    zeros = xr.zeros_like(ds.xc.expand_dims({'theta':ds.theta.data}))
    coords0 = {'latitude':zeros, 'longitude':zeros}
    coords1 = {'latitude':zeros, 'longitude':zeros}
    if 'valid_time' in ds.dims:
        for t in ds.valid_time:
            c1 = find_centre3d(ds.sel(valid_time=t))
            coords1['latitude'].loc[{'valid_time':t}] = c1['latitude']
            coords1['longitude'].loc[{'valid_time':t}] = c1['longitude']
    else:
        coords1 = find_centre3d(ds)
    dsc = ds.copy()
    while coords1 is not coords0:
        dlatdt = coords1['latitude'].differentiate('valid_time', datetime_unit='s')
        dlondt = coords1['longitude'].differentiate('valid_time', datetime_unit='s')
        v_bg = a * np.deg2rad(dlatdt)
        u_bg = a * np.deg2rad(dlondt) * np.cos(np.deg2rad(coords1['latitude']))
        dsc.u = ds.u - u_bg
        dsc.v = ds.v - v_bg
        coords0 = coords1
        if 'valid_time' in ds.dims:
            for t in ds.valid_time:
                c1 = find_centre3d(dsc.sel(valid_time=t))
                coords1['latitude'].loc[{'valid_time':t}] = c1['latitude']
                coords1['longitude'].loc[{'valid_time':t}] = c1['longitude']
        else:
            coords1 = find_centre3d(dsc)
        #coords1 = find_centre3d(dsc)
        print(u_bg.mean('theta'), v_bg.mean('theta'))
    print('done iterating')
    ds['latc3d'] = coords1['latitude']
    ds['lonc3d'] = coords1['longitude']
    return ds


def parse_arguments(**kwargs):
    """When arguments are passed, override global variables"""
    global method, steady, verbosity
    parser = argparse.ArgumentParser(
        description="Calculate cross-isentropic mass flux and other variables")
    parser.add_argument('-m',choices=['euler','lagrange2d'],
                        dest='method',
                        help='method for determining the fluxes')
    parser.add_argument('--steady', action='store_true',
                        help='assume steady-state in flux calculations.')
    parser.add_argument('-v','--verbosity',help='verbosity level (integer)',type=int)
    parser.add_argument('-f',help='necessary for JupyterLab, ignore')
    args = parser.parse_args()
    if args.method:
        method = args.method
    if args.steady: 
        steady = args.steady
    if args.verbosity:
        verbosity = args.verbosity
    return 


def main(files):
    """calculate cross-isentropic mass flux"""
    global startTime
    startTime = time.time()
    vprint(1,25*"-"+"main()"+25*"-")
    parse_arguments()
    vprint(1,f"Running s05cimf.py with method = {method}, steady = {steady}, verbosity = {verbosity}")
    ds = xr.open_mfdataset(files, concat_dim='valid_time', combine='nested')
    ds = ds.swap_dims(theta='z')
    if method == 'lagrange3d':
        ds = iterate_find_centre3d(ds)
        return ds
    if (method in ['lagrange2d','lagrange3d']):# and (not steady):
        ds = crop(ds)
    ds['cimf'] = calc_cimf(ds, T=0 if steady else None)
    ds,nanratio = azimean(ds)
    vprint(1,"done.")
    return ds,nanratio


if __name__ == '__main__':
    main(files[3:5])