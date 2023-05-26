#!/usr/bin/python3

# Import modules
import glob
import os
import sys
import copy
import warnings
import matplotlib.pyplot as plt
import matplotlib as mp
import netCDF4 as nc
import numpy as np
import math
import xarray as xr
import irmaFn as irma

# Define constants
a = 6371000  # radius earth [m]
fc = 4.25e-5 # coriolis parameter [1/s]
Rsp = 287    # Specific gas constant dry air [J/kg/K]
kappa = 2/7  # R/c_p [J/kg/K]
g = 9.81     # Gravitational constant [m/s2]

def distance2centre(lats, lons, latc, lonc):
    """
    Calculate distance between different latitude(s), longitude(s) 
    using haversine formula
    
    Parameters
    ----------
    lats, lons: numpy.ndarray
        latitude(s) and longitude(s) to compute distance (0/1/2-d)
    latc, lonc: numpy.ndarray
        reference latitude(s), longitude(s)
        must have same number of dimensions as lats, lons (0/1/2-d)
    
    Returns
    -------
    dist: numpy.ndarray
        distance [m] to reference latitude, longitude
        shape: lats.shape
    angle: numpy.ndarray 
        angle [rad] of coordinates w.r.t. eastward axis 
        (positive anti-clockwise)
        shape: lats.shape
    """
    print("distance2centre is executed")
    def expand_dims(x, new_dims):
        new_dims = set(x.dims)
    
    if isinstance(lats, xr.DataArray):
        latsr = (lats * (np.pi/180)).expand_dims({dim:len(latc[dim]) for dim in latc.dims if (dim not in lats.dims)})
        lonsr = (lons * (np.pi/180)).expand_dims({dim:len(lonc[dim]) for dim in lonc.dims if (dim not in lons.dims)})
        latcr = (latc * (np.pi/180)).expand_dims({dim:len(latsr[dim]) for dim in latsr.dims if (dim not in latc.dims)})
        loncr = (lonc * (np.pi/180)).expand_dims({dim:len(lonsr[dim]) for dim in lonsr.dims if (dim not in lonc.dims)})
        #latcr = xr.DataArray(latcr.broadcast_like(latsr).compute().data, latsr.coords, latsr.dims)
        #loncr = xr.DataArray(loncr.broadcast_like(lonsr).compute().data, lonsr.coords, lonsr.dims)
#         yx = {d:s for d,s in zip(lats.dims, lats.shape)}
        
#         latcr = latc.expand_dims(yx, axis=[-2,-1]) * np.pi/180
#         loncr = lonc.expand_dims(yx, axis=[-2,-1]) * np.pi/180
#         o = set(latcr.dims) - set(yx.keys())
#         print(o)
#         latsr = lats.expand_dims(o) * np.pi/180
#         lonsr = lons.expand_dims({d:s for d,s in zip(latcr.dims, latcr.shape)}) * np.pi/180
        pass
    else:
        if lats.ndim == 0 and lons.ndim == 0:
            latsr = (latc * np.pi/180)
            lonsr = (lonc * np.pi/180)  
        if lats.ndim == 1 and lons.ndim == 1:
            latsr = (lats *np.pi/180)[:,None,None]
            lonsr = (lons *np.pi/180)[:,None,None]
        elif lats.ndim == 2 and lons.ndim == 2: 
            latsr = (lats * np.pi/180)[None,:,:]
            lonsr = (lons * np.pi/180)[None,:,:]
        if latc.ndim == 0 and lonc.ndim == 0:
            latcr = (latc * np.pi/180)
            loncr = (lonc * np.pi/180)
        elif latc.ndim == 1 and lonc.ndim == 1:
            latcr = (latc * np.pi/180)[:,None,None]
            loncr = (lonc * np.pi/180)[:,None,None]
        elif latc.ndim == 2 and lonc.ndim == 2:
            latcr = (latc * np.pi/180)[None,:,:]
            loncr = (lonc * np.pi/180)[None,:,:]

    # haversine formula for distance
    dlat = latsr-latcr#.data
    dlon = lonsr-loncr
    H1 = np.sin(dlat/2)**2
    H2 = np.cos(latsr) * np.cos(latcr) * np.sin(dlon/2)**2
    dist = 2 * a * np.arcsin(np.sqrt(H1 + H2))
    if isinstance(dist, xr.DataArray):
        dist = dist.assign_coords(lats.coords)
        
    # azimuthal angle
    dx = a * (lonsr - loncr) * np.cos(latsr)
    dy = a * (latsr - latcr)
    angle = np.arctan2(dy,dx)
    if isinstance(angle, xr.DataArray):
        angle = angle%(2*np.pi)
    else:
        angle[angle<0] += 2*np.pi
    return np.squeeze(dist), np.squeeze(angle)

def latlon2xy(lats, lons, latc, lonc):
    assert lats.shape == lons.shape
    assert lats.ndim == 2
    
    assert latc.ndim == 0
    assert lonc.ndim == 0
#     assert isinstance(latc, float)
#     assert isinstance(lonc, float)
    latsr = lats * np.pi / 180
    lonsr = lons * np.pi / 180
    latcr = latc * np.pi / 180
    loncr = lonc * np.pi / 180
    y = a * (latsr - latcr)
    x = a * (lonsr - loncr) * np.cos(latsr - latcr)
    
    return y, x
    
def latlon2dxdy(lats, lons):
    """
    Compute grid size (m) of latitude-longitude grid
    
    Parameters
    ----------
    lats, lons: numpy.ndarray
        Latitude(s), longitude(s) to convert (2-d)
    
    Returns
    -------
    dx, dy: numpy.ndarray
        Length and width of grid cells [m] x: zonal, y: meridional
        shape: lats.shape
    """
    if isinstance(lats, xr.DataArray) and all([dim in lats.dims for dim in ['y','x']]):
        lats = lats.transpose(...,'y','x')
        lons = lons.transpose(...,'y','x')
        dx = xr.zeros_like(lons)
        dy = xr.zeros_like(lats)
    elif isinstance(lats, xr.DataArray) and all([dim in lats.dims for dim in ['dy','dx']]):
        lats = lats.transpose(...,'dy','dx')
        lons = lons.transpose(...,'dy','dx')
        dx = xr.zeros_like(lons)
        dy = xr.zeros_like(lats)
    else:
        dx = np.zeros(lons.shape)
        dy = np.zeros(lats.shape)
    
    # central difference
    if isinstance(dx, xr.DataArray):
        dx.data[...,:,1:-1],_ = distance2centre(lats[...,:,1:-1], lons[...,:,2:], 
            lats[...,:,1:-1], lons[...,:,:-2])
        dy.data[...,1:-1,:],_ = distance2centre(lats[...,2:,:], lons[...,1:-1,:], 
            lats[...,:-2,:], lons[...,1:-1,:])
    else:
        dx[...,:,1:-1],_ = distance2centre(lats[...,:,1:-1], lons[...,:,2:], 
            lats[...,:,1:-1], lons[...,:,:-2])
        dy[...,1:-1,:],_ = distance2centre(lats[...,2:,:], lons[...,1:-1,:], 
            lats[...,:-2,:], lons[...,1:-1,:])
    dx /= 2
    dy /= 2
    
    # linear boundary approximation
    if isinstance(dx, xr.DataArray):
        dx.data[...,:,[0,-1]] = 2 * dx.data[...,:,[1,-2]] - dx.data[...,:,[2,-3]]
        dy.data[...,[0,-1],:] = 2 * dy.data[...,[1,-2],:] - dy.data[...,[2,-3],:]
    else:
        dx[...,:,[0,-1]] = 2 * dx[...,:,[1,-2]] - dx[...,:,[2,-3]]
        dy[...,[0,-1],:] = 2 * dy[...,[1,-2],:] - dy[...,[2,-3],:]

    return dx, dy

def latlon2idx(lat, lon, latc, lonc):
    """
    Find index of latc, lonc in spherical grid

    Parameters
    ----------
    lat, lon: numpy.ndarray
        Latitude(s), longitude(s) in which to search for latc, lonc (2-d)
    latc, lonc: numpy.ndarray
        Latitude(s) and longitude(s) to find

    Returns:
        latids, lonids: numpy.ndarray of ints
            indices of latc, lonc in lat, lon
            shape: latc.shape
    """
    
    latids = np.empty(latc.shape, dtype=int)
    lonids = np.empty(lonc.shape, dtype=int)

    # for all coordinates in latc-lonc, find closest point in lat-lon
    for i,y in enumerate(latc.flat):
        ilats = np.argmin(np.abs(lat-y), axis=0)
        lon_s = lon[ilats, range(len(ilats))]    
        ilon = np.argmin(np.abs(lon_s-lonc.flat[i]))
        latids.flat[i], lonids.flat[i] = ilats[ilon], ilon
    
    return latids, lonids

def removeBGwind(tlu, tlv, latc, lonc, latc_nxt, lonc_nxt):
    """
    Remove background wind based on TC center displacement
    
    Parameters
    ----------
    tlu, tlv: numpy.ndarray
        zonal and meridional velocity [m/s] (3-d)
    latc, lonc: numpy.ndarray
        TC center latitude - longitude [deg] (1-d) 
    latc_nxt, lonc_nxt: numpy.ndarray
        same as latc, lonc, but for next timestep
    """
    dist, angle = distance2centre(latc_nxt, lonc_nxt, latc, lonc)
    u_bg = dist * np.cos(angle) / 3600
    v_bg = dist * np.sin(angle) / 3600
    u = tlu - u_bg[:,None,None]
    v = tlv - v_bg[:,None,None]
    return u, v

def relativevorticity(u, v, lats, lons):
    """
    Determine relative vorticity (dvdx-dudy)
    
    Parameters
    ----------
    u, v: numpy.ndarray
        zonal and meridional velocities [m/s] (at least 2-d)
    lat, lon: numpy.ndarray
        spherical grid (2-d)
    
    Returns
    -------
    rv: numpy.ndarray
        relative vorticity [1/s], shape: u.shape
    """
    
    du, dv = np.gradient(u, axis=-2), np.gradient(v, axis=-1)
    dx, dy = latlon2dxdy(lats, lons)
    rv = dv/dx - du/dy
    
    return rv

def moving_average1D(arr, w=3, end="cyclic"):
    """
    Moving average on 1D array with cyclic end condition
    
    Parameters
    ----------
    arr: numpy.ndarray
        array to apply moving average to
    w:  int
        moving average window (odd number preferred)
    end: string {"cyclic","regular"}
        if 'cyclic', will join beginning and end of array when 
        window is partially out of range, 
        if 'regular', will compute average only on values within range
    """
    
    if len(arr) < w:
        return arr
    
    # Compute rolling mean
    d = w//2
    extended_arr = np.concatenate((arr[-w:],arr,arr[:w]))
    mavg = (np.convolve(extended_arr, np.ones(w), 'same') / w)[w:-w]
    
    # Fix boundaries
    if end == "cyclic":
        pass
    elif end == "regular":
        mavg[:d] = [np.mean(arr[:d+1+i]) for i in range(d)]
        mavg[-d:] = [np.mean(arr[-d-i:]) for i in range(d)][::-1]
    else:
        raise Exception("Invalid value for 'end': {}".format(end))
        
    return mavg

def invalid_angles(angles, angle_ipl, d=1):
    """
    Invalidate interpolated points based on distance to original
    data
    
    Parameters
    ----------
    angles: numpy.ndarray
        original data (1-d) [deg]
    angle_ipl: numpy.ndarray
        interpolated data (1-d) [deg]
    d: int or float
        maximum allowed angular distance from original data [deg]
    
    Returns
    -------
    invalid_angles: numpy.ndarray
        subset of angle_ipl which is invalidated
    """

    angles %= 360; angle_ipl %= 360
    dist = np.min(np.abs(angle_ipl[:,None] - angles[None,:]), axis=1)
    invalid_angles = np.array(angle_ipl[dist > d], dtype=int)

    return invalid_angles

def azimuthalmean(x, lat, lon, latc, lonc, rmin, rmax=None):
    """
    Determine azimuthal average and remainder of field x 
    around central lat,lon
    
    Parameters
    ----------
    x: numpy.ndarray
        field to take azimuthal average of (at least 2-d)
    lat, lon: numpy.ndarray (2-d)
        spherical grid
    latc, lonc: numpy.ndarray
        central latitude(s), longitude(s)
    rmin, rmax: float (rmax opt.)
        minimum and maximum radius from centre [m] wherein averages 
        are computed
        if rmax is not specified, it will be set to rmin + 5000m
        
    Returns
    -------
    mean: float or numpy.ndarray
        azimuthal mean of x around central coordinates 
        shape:  len(x) if x is 3-d 
                1      if x is 2-d
    eddy: numpy.ndarray
        remainder of x interpolated to standard angle-grid
        shape:  len(x), len(angle) if x is 3-d 
                len(angle)         if x is 2-d
    angle: numpy.ndarray
        standard angle-grid [deg] from eastward axis
        positive in anti-clockwise direction
    """

    angles = np.linspace(0,359,num=360) # Azimuthal angles [deg]
    
    # preprocessing
    if not rmax:
        rmax = rmin + 5000
      
    # select data within rmin and rmax from centre  
    dist, a = distance2centre(lat, lon, latc, lonc)
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    if dist.ndim == 2:
        dist = np.expand_dims(dist, axis=0)
    mask = (dist >= rmin) * (dist < rmax) 
    x_m = {l:x[l,mask[l]] for l in range(len(x))}
    a_m = {l:a[l,mask[l]] for l in range(len(x))}
    
    # sort data by azimuthal angle (& apply mov.avg. on x)
    sortids = {l:np.argsort(a_m[l]) for l in range(len(x))}
    x_ms = {l:moving_average1D(x_m[l][sortids[l]]) for l in range(len(x))} 
    a_ms = {l:a_m[l][sortids[l]]*180/np.pi for l in range(len(x))}
    
    # interpolate to standard angles
    x_msi = np.zeros((len(x),(len(angles)))) 
    a_msi = np.zeros((len(x),(len(angles))))
    for l in range(len(x)):
        x_msi[l] = np.interp(angles, a_ms[l], x_ms[l])
        angle_invalid = invalid_angles(a_ms[l], angles, d=3)
        x_msi[l,angle_invalid] = np.nan     
        
    # take mean/eddy components
    validlvls = np.sum(np.isnan(x_msi), axis=-1) <= 18 # >95% data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(x_msi, axis=1)
    eddy = x_msi - mean[:,np.newaxis]
    mean[~validlvls] = np.nan
    angles_r = angles * np.pi/180
    mean, eddy = np.squeeze(mean), np.squeeze(eddy) 
       
    return mean, eddy, angles_r

def toPolar(lats, lons, xc, yc, dr=5e3, dt=2, rmax=1000e3):
    assert all([type(x) in (xr.DataArray, xr.Dataset) for x in [lats, lons, xc, yc]])
    
    # set up new coordinates
    r = np.arange(0,rmax+dr,dr)
    r_bnds = np.array([(max(0,(rr-dr/2)),(rr+dr/2)) for rr in r])
    t = np.arange(0,360,dt)
    t_bnds = np.array([((tt-dt/2),(tt+dt/2)) for tt in t])%360/180*np.pi
    
    if 'y' in lats.dims and 'x' in lats.dims:
        cloc = dict(y=yc,x=xc)
    elif 'dy' in lats.dims and 'dx' in lats.dims:
        cloc = dict(dy=0,dx=0)
    else:
        raise ValueError(f'Wrong horizontal dimensions {lats.dims}, should be (y,x) or (dy,dx)')
    dist, angle = distance2centre(lats, lons, lats.isel(cloc), lons.isel(cloc))
    return dist, angle

def radialvelocity(tlu, tlv, angle):
    """
    Calculate radial and tangential velocity
    
    Parameters
    ----------
    tlu, tlv: numpy.ndarray (2-d or 3-d)
        zonal and meridional velocities [m/s]
    angle: numpy.ndarray
        azimuthal angle w.r.t. eastward axis (positive anti-clockwise)
    
    Returns
    -------
    u_rad, v_tan: numpy.ndarray
        radial and tangential velocity in cylindrical coordinates
        same shape as tlu, tlv
    """
    
    V_abs = np.sqrt(tlu**2 + tlv**2)
    V_ang = np.arctan2(tlv, tlu)
    u_rad = V_abs * np.cos(V_ang - angle)
    v_tan = V_abs * np.sin(V_ang - angle)
    
    return u_rad, v_tan

def calc_dp(tlp,glp):
    """
    Calculate pressure difference between isentropes or between 
    lowest isentrope and surface level.
    
    Parameters
    ----------
    tlp: numpy.ndarray
        pressure [hPa] on isentropic surfaces (3-d) 
    glp: numpy.ndarray
        pressure [Pa] on ground level (2-d)
    
    Returns
    -------
    dp: numpy.ndarray
        dp at isentropic level i is equal to the difference in 
        pressure between i and i-1, if isentropic level i-1 does 
        not exist, ground level is used 
        dp has same shape as tlp
    """
    
    check_nan = True
    dp = np.zeros(tlp.shape)
    dp[0] = 100 * tlp[0] - glp
    
    for lvl in range(1,tlp.shape[0]):           
        dp[lvl] = 100 * (tlp[lvl] - tlp[lvl-1])
        if check_nan:
            nan_idcs = np.isnan(dp[lvl])
            dp[lvl][nan_idcs] = 100 * tlp[lvl][nan_idcs] - glp[nan_idcs]
            check_nan = False if not nan_idcs.any() else True    
        with np.errstate(invalid='ignore'):
            dp[lvl][dp[lvl]>0] = np.nan
            
    return dp  
    
def isentropicDensity(tlp, glp, lvl):
    """
    Calculate isentropic density sig = -1/g dp/dtheta
    
    Parameters
    ----------
    tlp: numpy.ndarray
        pressure [hPa] on isentropic levels (3-d)
    glp: numpy.ndarray
        pressure [Pa] on ground level (2-d)
    lvl: numpy.ndarray
        isentropic levels [K] (1-d)
    
    Returns
    -------
    sigma: numpy.ndarray
        isentropic density (kg/m^2K), same shape as tlp
    """
    
    dp = calc_dp(tlp,glp)
    dth = np.diff(np.append(np.nan,lvl))[:,None,None]
    sigma = -dp/(g*dth)
    
    return sigma

def isentropicHeight(tlp, glp, lvl):
    """
    Calculate thickness of all isentropic layers
    
    Parameters
    ----------
    tlp: numpy.ndarray
        pressure [hPa] on isentropic levels (3-d)
    glp: numpy.ndarray
        pressure [Pa] on ground level (2-d)
    lvl: numpy.ndarray
        isentropic levels [K] (1-d)
    
    Returns
    -------
    H: numpy.ndarray
        height [m] of the heighest isentrope above the surface
        same shape as glp
    dH: numpy.ndarray
        thickness [m] of each isentropic layer bounded by two 
        isentropic surfaces. same shape as tlp
    """
    
    p_ref = 1000 #hPa
    T = lvl[:,None,None]*(tlp/p_ref)**kappa
    rho = 100 * tlp / (Rsp * T) + 0.03 # use constant humidity
    dp = calc_dp(tlp,glp)
    dH = -dp/rho/g
    H = np.nancumsum(dH, axis=0)
    return H, dH

def integratePVS(absvort, dH, lat, lon, latc, lonc, rmax=50000):
    """
    Integrate absolute vorticity to get potential vorticity substance
    
    Parameters
    ----------
    absvort: numpy.ndarray
        absolute vorticity [1/s] (3-d)
    dH: numpy.ndarray
        thickness [m] of isentropic layers (3-d)
    lat, lon: numpy.ndarray
        spherical grid latitude - longitude (2-d)
    latc, lonc: numpy.ndarray
        central latitude - longitude (0/1-d)
    rmax: float
        maximum radius [m] that forms the integration boundary
    
    Returns
    -------
    PVS: numpy.ndarray
        Potential vorticity substance [m3/s], shape: len(absvort)
    """
    
    dist, angle = irma.distance2centre(lat, lon, latc, lonc) 
    if dist.ndim == 2:
        dist = dist[None,:,:]
    mask = (dist < rmax)
    dx, dy = irma.latlon2dxdy(lat, lon)
    dA = dx * dy
    dPVS = absvort * mask * dA[None,:,:] * dH
    PVS = np.sum(dPVS.reshape(len(dPVS),-1), axis=-1)
    
    return PVS

def Jheating(cimf, sig, v_tan, lvl):
    """
    Calculate the heating term of the vorticity flux

    Parameters
    ----------
    cimf: numpy.ndarray
        cross-isentropic mass flux [kg/m^2s] (3-d)
    sig: numpy.ndarray
        isentropic density [kg/m2K] (3-d)
    v_tan: numpy.ndarray
        tangential velocity [m/s] (3-d)
    lvl: numpy.ndarray
        isentropic levels [K] (1-d)

    Returns
    -------
    J_heat: numpy.ndarray
        heating term of the vorticity flux [m/s^2]
    """
    if cimf.shape != sig.shape:
        cimf = np.concatenate((cimf[:,:1], cimf, cimf[:,-1:]), axis=1)
        cimf = np.concatenate((cimf[:,:,:1], cimf, cimf[:,:,-1:]), axis=2)
        cimf = (cimf[:,1:,1:] + cimf[:,1:,:-1] 
            + cimf[:,:-1,1:] + cimf[:,:-1,:-1])/4
    dthdt = cimf/sig
    
    dv = np.diff(np.append(np.zeros(v_tan[:1].shape), v_tan, axis=0), axis=0)
    dvdth = dv / np.diff(np.append(np.nan, lvl))[:,None,None]
    J_heat = dthdt * dvdth
    
    return J_heat

    
def integrateJ(u_rad, v_tan, rv, dH, J_heat, lat, lon, latc, lonc, lvl, rmin, 
    rmax):
    """
    Integrate vorticity flux along boundary of cylinder bounded by 
    rmin, rmax
    
    Parameters:
    -----------
    u_rad, v_tan: numpy.ndarray
        radial and tangential velocity [m/s] (3-d)
    rv: numpy.ndarray
        mean and eddy component of relative vorticity [1/s] (3-d)
    dH: numpy.ndarray
        thickness of isentropic layers [m] (3-d)
    J_heat: numpy.ndarray
        heating term vorticity flux [m/s^2] (3-d)
    lvl: numpy.ndarray
        isentropic levels [K] (1-d)
    rmin, rmax: float
        minimum and maximum integration bounds radius [m] 
    
    Returns:
    --------
    JdA_mean, JdA_eddy, JdA_heat: numpy.ndarray
        integrated mean, eddy and heating vorticity flux [m^3/s^2]
    """
    # Calculate mean and eddy fields along ring-shaped integration boundary
    J_hm, J_he, _ = azimuthalmean(J_heat, lat, lon, latc, lonc, rmin, rmax)
    u_m ,  u_e, _ = azimuthalmean( u_rad, lat, lon, latc, lonc, rmin, rmax)
    rv_m, rv_e, _ = azimuthalmean(    rv, lat, lon, latc, lonc, rmin, rmax)
    dH_m, dH_e, _ = azimuthalmean(    dH, lat, lon, latc, lonc, rmin, rmax)
    J_mean  = u_m * (rv_m + fc)    # mean vorticity flux
    J_eddy  = u_e * (rv_e + fc)    # eddy vorticity flux
    J_heat  = J_hm[:,None] + J_he  # heating term vorticity flux
    dH_ring = dH_m[:,None] + dH_e  # thickness along boundary

    # Integrate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        JdA_mean = J_mean * np.pi * (rmin+rmax) * dH_m
        JdA_eddy = (np.nanmean(J_eddy * dH_ring, axis=1) 
            * np.pi * (rmin + rmax))
        JdA_heat = (np.nanmean(J_heat * dH_ring, axis=1) 
            * np.pi * (rmin + rmax))

    return JdA_mean, JdA_eddy, JdA_heat


def sqboundary(x, posdir=None, excl_edges=None):
    """

    Return x along its square horizontal boundary (clockwise from 
    SW corner). If x is a vector, the product of x with the unit 
    surface vector (positive outside) is returned.

    Parameters
    ----------
    x: array:
        field to determine boundary of (at least 2-d)
    posdir: {'north','east'}
        positive (horizontal) direction of x
        if 'east', values on the west side are multiplied by -1
        if 'north', values on the south side are multiplied by -1
    excl_edges: list of ints:
        indices of vertices which indicate where the output will be
        zero. For example [0,3] returns 0's at the western and 
        southern boundaries
    """
    x1_c, x2_c, x3_c, x4_c = 1, 1, 1, 1

    if posdir:
        if posdir == 'east':
            x1_c, x2_c, x4_c = -1, 0, 0
        elif posdir == 'north':
            x1_c, x3_c, x4_c = 0, 0, -1
        else:
            raise Exception("posdir must be {'north','east'}")

    if excl_edges:
        ex_edges = ~np.isin([0,1,2,3], excl_edges) * 1
        x1_c, x2_c, x3_c, x4_c = np.multiply(ex_edges, 
            [x1_c, x2_c, x3_c, x4_c])

    x1 = x1_c * x[...,:-1,0]
    x2 = x2_c * x[...,-1,:-1]
    x3 = x3_c * x[...,:0:-1,-1]
    x4 = x4_c * x[...,0,:0:-1]

    x_bnd = np.concatenate((x1,x2,x3,x4), axis=-1)

    return x_bnd

def storeVorticity(rnames):
    """
    Loop through files in order, calculate the relative, absolute and 
    potential vorticity and store them as new variables.
    
    Parameters
    ----------
    rnames: list of strings
        filenames to read data from
    """
    rnames = sorted(rnames)
    start = 0
    stop = len(rnames)
    for fid in range(start,stop):
        print("\rUpdating file {}/{}".format(fid+1,stop-start),end="")
        
        # read data
        ds = nc.Dataset(rnames[fid],"r+") 
        u     = np.ma.array(ds['u'][:], mask=(ds['u'][:]==ds['u'].GRIB_missingValue))
        v     = np.ma.array(ds['v'][:], mask=(ds['v'][:]==ds['v'].GRIB_missingValue))
        sigma = np.ma.array(ds['sigma'][:], mask=(ds['sigma'][:]==ds['sigma'].GRIB_missingValue))
        lats, lons = ds['latitude'][:], ds['longitude'][:]
        
        # relative vorticity
        rv = relativevorticity(u, v, lats, lons)
        if 'rv' not in ds.variables.keys():
            RV = ds.createVariable('rv','f8',('theta','y','x'))
            RV.long_name = "relative vorticity"
            RV.units = "s**-1"
            RV.GRIB_shortName = "rv"
        else:
            RV = ds['rv']
        RV[:] = rv.filled(fill_value=9999).astype('f4')
        RV.GRIB_missingValue = 9999
        
        # absolute vorticity
        av = fc + rv
        if 'av' not in ds.variables.keys():
            AV = ds.createVariable('av','f8',('theta','y','x'))
            AV.long_name = "absolute vorticity"
            AV.units = "s**-1"
            AV.GRIB_shortName = "av"
        else:
            AV = ds['av']
        AV[:] = av.filled(fill_value=9999).astype('f4')
        AV.GRIB_missingValue = 9999
 
        # potential vorticity
        pv = av / sigma * 1e6
        if 'pv' not in ds.variables.keys():
            PV = ds.createVariable('pv','f8',('theta','y','x'))
            PV.long_name = "potential vorticity"
            PV.units = "pvu"
            PV.GRIB_shortName = "pv"
        else:
            PV = ds['pv']
        PV[:] = pv.filled(fill_value=9999).astype('f4')
        PV.GRIB_missingValue = 9999

    ds.close()
    print("")
    
def storeUV(rnames):
    """
    Loop through files in order, calculate the radial and tangential velocities 
    and store them as new variables.
    
    Parameters
    ----------
    rnames: list of strings
        filenames to read data from
    """
    rnames = sorted(rnames)
    start = 0
    stop = len(rnames)
    for fid in range(start,stop):
        print(globals())
        print("\rUpdating file {}/{}  (storeUV)".format(fid+1,stop-start),end="")
        
        # read data
        ds = nc.Dataset(rnames[fid],"r+") 
        u     = np.ma.array(ds['u'][:], mask=(ds['u'][:]==ds['u'].GRIB_missingValue))
        v     = np.ma.array(ds['v'][:], mask=(ds['v'][:]==ds['v'].GRIB_missingValue))
        lats, lons = ds['latitude'][:], ds['longitude'][:]
        latc, lonc = ds['latc'][:], ds['lonc'][:]
        
        # calculate radial and tangential velocity
        dist, angle = distance2centre(lats, lons, latc, lonc)
        u_rad, v_tan = radialvelocity(u, v, angle)
        
        # store radial velocity
        if 'u_rad' not in ds.variables.keys():
            UR = ds.createVariable('u_rad','f8',('theta','y','x'))
            UR.long_name = "radial velocity"
            UR.units = "m/s"
            UR.GRIB_shortName = "u_rad"
        else:
            UR = ds['rv']
        UR[:] = u_rad.filled(fill_value=9999).astype('f4')
        UR.GRIB_missingValue = 9999
        
        # store tangential velocity
        if 'v_tan' not in ds.variables.keys():
            VT = ds.createVariable('v_tan','f8',('theta','y','x'))
            VT.long_name = "tangential velocity"
            VT.units = "m/s"
            VT.GRIB_shortName = "v_tan"
        else:
            VT = ds['v_tan']
        VT[:] = v_tan.filled(fill_value=9999).astype('f4')
        VT.GRIB_missingValue = 9999
        
    ds.close()
    print("")
    
def storeW(rnames):
    """
    Loop through files in order, calculate the vertical velocity dthetadt 
    and store as a new variable.
    
    Parameters
    ----------
    rnames: list of strings
        filenames to read data from
    """
    rnames = sorted(rnames)
    start = 0
    stop = len(rnames) - 1
    for fid in range(start,stop):
        print("\rUpdating file {}/{}  (storeW)".format(fid+1,stop-start),end="")
        
        # read data
        ds = nc.Dataset(rnames[fid],"r+") 
        sigma = np.ma.array(ds['sigma'][:], mask=(ds['sigma'][:]==ds['sigma'].GRIB_missingValue))
        cimf = np.ma.array(ds['cimf'][:], mask=(ds['cimf'][:]==ds['cimf'].GRIB_missingValue))
        
        # interpolate cimf to regular grid
        regshape = cimf.shape[:-2] + tuple(x+1 for x in cimf.shape[-2:])
        rcimf = np.ma.array(np.zeros(regshape), mask=np.zeros(regshape))
        rcimf[...,1:-1,1:-1] = np.ma.mean([cimf[...,:-1,:-1], cimf[...,1:,:-1], cimf[...,:-1,1:], cimf[...,1:,1:]], axis=0)
        rcimf[...,[0,-1],:] = rcimf[...,[1,-2],:]
        rcimf[...,:,[0,-1]] = rcimf[...,:,[1,-2]]
        
        # interpolate cimf to regular grid
#     cimfmask = np.zeros(cimf.shape)
#     cimfmask[...,1:,1:] += cimf.mask[...,:,:] 
#     cimfmask[...,1:,:-1] += cimf.mask[...,:,:]
#     cimfmask[...,:-1,1:] += cimf.mask[...,:,:]
#     cimfmask[...,:-1,:-1] += cimf.mask[...,:,:]
#     cimfmask = (cimfmask > 0)
#     cimfdata = np.zeros(u.shape)
#     CIMFdata = ds['cimf'][:]
    
#     cimfdata[:,1:,1:] = CIMFdata[...,:,:]
#     cimf = np.ma.array(cimfdata, mask=cimfmask)

        
        # calculate vertical velocity
        dtdt = rcimf / sigma
        
        # store radial velocity
        if 'w' not in ds.variables.keys():
            W = ds.createVariable('w','f8',('theta','y','x'))
            W.long_name = "vertical velocity"
            W.units = "K/s"
            W.GRIB_shortName = "w"
        else:
            W = ds['w']
        W[:] = dtdt.filled(fill_value=9999).astype('f4')
        W.GRIB_missingValue = 9999
        
    ds.close()
    print("")
    
#  def restoreLatcLonc(rnames):
#     rnames = sorted(rnames[:-4])
#     for fid in range(len(rnames)):
#         fhy = np.load(rnames[fid]+".npy",allow_pickle=True).item()
#         fhz = dict(np.load(rnames[fid]+".npz"))

def updateLatLonNxt(rnames):
    """
    Loop through files in order and set latc_nxt, lonc_nxt to latc, lonc 
    of next file
    
    Parameters
    ----------
    rnames: list of strings
        filenames to read data from
    """
    rnames = sorted(rnames)
    start = 0
    stop = len(rnames)
    ext = os.path.splitext(rnames[0])[-1]
    if ext in ['npy','npz']:
        fh_cur = dict(np.load(rnames[start]))
        for fid in range(start, stop):
            print("\rUpdating file {}/{}".format(fid+1,stop-start),end="")
            try:
                fh_nxt = dict(np.load(rnames[fid+1]))
                latc_nxt = fh_nxt['latc']
                lonc_nxt = fh_nxt['lonc']
            except IndexError:
                latc_nxt = np.nan
                lonc_nxt = np.nan
            fh_cur['latc_nxt'] = latc_nxt
            fh_cur['lonc_nxt'] = lonc_nxt
            np.savez(rnames[fid], **fh_cur)
            fh_cur = fh_nxt
        print("")
    elif ext in ['.nc']:
        fh_cur = nc.Dataset(rnames[start],"r+")
        for fid in range(start, stop):
            print("\rUpdating file {}/{}".format(fid+1,stop-start),end="")
            try:
                fh_nxt = nc.Dataset(rnames[fid+1],"r+")
                latc_nxt = fh_nxt['latc'][:]
                lonc_nxt = fh_nxt['lonc'][:]
            except IndexError:
                latc_nxt = np.nan
                lonc_nxt = np.nan
            if 'latc_nxt' not in fh_cur.variables.keys():
                LATC_NXT = fh_cur.createVariable('latc_nxt','f8',('theta'))
                LATC_NXT.long_name = "latitude of TC center next step"
                LATC_NXT.units = "degrees_north"
                LATC_NXT.GRIB_shortName = "latc"
            else:
                LATC_NXT = fh_cur['latc_nxt']
            LATC_NXT[:] = latc_nxt
            if 'lonc_nxt' not in fh_cur.variables.keys():
                LONC_NXT = fh_cur.createVariable('lonc_nxt','f8',('theta'))
                LONC_NXT.long_name = "longitude of TC center next step"
                LONC_NXT.units = "degrees_east"
                LONC_NXT.GRIB_shortName = "lonc"
            else:
                LONC_NXT = fh_cur['lonc_nxt']
            LONC_NXT[:] = lonc_nxt
            fh_cur.close()
            fh_cur = fh_nxt
        print("")
    else:
        raise NotImplementedError(f"Cannot load from {ext} file.")   

def findCenter(rnames):
    """Find center of tropical cyclone by minimizing azimuthal variance of 
    pressure field.
    
    Parameters
    ----------
    rpath: str
        path to file containing pressure field
    
    Returns
    -------
    latc, lonc: float
        central latitude and longitude of the cyclone, for which the azimuthal
        variance of the pressure field is a minimum
    """
    def calc_variance(f, lats, lons):
        """
        Calculates a measure for the azimuthal variance of field f 
        within ~65km (20gridpoints) 
        """
        assert f.shape[-1] == 41 and f.shape[-2] == 41
        assert lats.shape == lons.shape and lats.shape == f.shape[-2:]
        latc = lats[20,20]; lonc = lons[20,20]
        dist, angle = distance2centre(lats, lons, latc, lonc)
        dist, angle = np.squeeze(dist), np.squeeze(angle)
        dphi = 0.2*np.pi
        means = np.empty(10) #stores means of 'pizza slices' around center
        for ii, phi in enumerate(np.linspace(0,2*np.pi-dphi,10)):
            mask = (angle>=phi) * (angle<phi+dphi) * (dist<65000)
            means[ii] = np.nanmean(f[mask])
        return np.var(means)

        
    rnames = sorted(rnames)
    ext = os.path.splitext(rnames[0])[-1]
    
    if ext in ['npy','npz']:
        fig, ax = plt.subplots(3,3, figsize=(20,20))
    
        for ii, rname in enumerate(rnames[::4]):
            print(f"\rFinding TC center in file {ii+1}", end=' ')
            fh = np.load(rname)
            latc, lonc = fh['latc'][4], fh['lonc'][4]
            lat, lon = fh['lat'], fh['lon']
            yc, xc = latlon2idx(lat, lon, latc, lonc) #lat-lon index of center
            variances = np.empty((21,21))
            for dx in range(21):
                for dy in range(21):
                    x = xc + dx - 10; y = yc + dy - 10
                    variances[dy,dx] = calc_variance(
                        fh['glp'][y-20:y+21, x-20:x+21],
                        fh['lat'][y-20:y+21, x-20:x+21],
                        fh['lon'][y-20:y+21, x-20:x+21])
            dy, dx = np.unravel_index(np.argmin(variances), variances.shape)
            yc_n = yc + dy - 10; xc_n = xc + dx - 10
        
            #change y and x
            dx, dy = 20, 20
            lat = lat[yc-dy:yc+dy, xc-dx:xc+dx]
            lon = lon[yc-dy:yc+dy, xc-dx:xc+dx]
            tlu = fh['tlu'][:, yc-dy:yc+dy, xc-dx:xc+dx]
            tlv = fh['tlv'][:, yc-dy:yc+dy, xc-dx:xc+dx]
            tlp = fh['tlp'][:, yc-dy:yc+dy, xc-dx:xc+dx]
            glp = fh['glp'][yc-dy:yc+dy, xc-dx:xc+dx]
            cimf = fh['CIMF'][:, yc-dy:yc+dy, xc-dx:xc+dx]
            sig = fh['sig'][:, yc-dy:yc+dy, xc-dx:xc+dx]
            lvl = fh['lvl']
            rv = relativevorticity(tlu, tlv, lat, lon)
            X, Y = np.meshgrid(lon[yc_n-yc+dy//2,:], lvl)
            cf = ax.flat[ii].contourf(X, Y, cimf[:,yc_n-yc+dy//2,:], cmap = 'seismic',
                levels=25, vmin = -4.25, vmax=4.25)
    #         ax.flat[ii].quiver(lon, lat, tlu[4], tlv[4])
            cs = ax.flat[ii].contour(X, Y, sig[:,yc_n-yc+15], colors='k')
            ax.flat[ii].clabel(cs, inline=True, inline_spacing=0)
    #         ax.flat[ii].axhline(latc)
    #         ax.flat[ii].axvline(lonc)
    #         ax.flat[ii].axhline(fh['lat'][yc_n,xc_n],linestyle='--')
    #         ax.flat[ii].axvline(fh['lon'][yc_n,xc_n],linestyle='--')
        fig.colorbar(cf, ax = ax.flat[-1])
        fig.suptitle("Cross-isentropic mass flux (shading) and isentropic density")
        plt.savefig("cimf_sig_latc.png")
        plt.show()
        print("")

    elif ext in ['.nc']:
        for ii,rname in enumerate(rnames):
            print(f"\rFinding TC center in {os.path.basename(rname)}")
            
            fh = nc.Dataset(rname,"r+")
            tlu = fh['u'][:]
            tlv = fh['v'][:]
            tlp = fh['p'][:]
            glp = fh['ps'][:]
            lat = fh['latitude'][:]
            lon = fh['longitude'][:]
            rv = relativevorticity(tlu, tlv, lat, lon)
            rv[:,:,:20] = 0 # filter out erroneous maxima
            yc, xc = np.unravel_index(np.argmax(rv[12,:,:]), rv.shape[-2:])
            print(f"yc: {yc}, xc: {xc} ")
            variances = np.empty((21,21))
            for dx in range(21):
                for dy in range(21):
                    x = xc + dx - 10; y = yc + dy - 10
                    variances[dy,dx] = calc_variance(
                        glp[y-20:y+21, x-20:x+21],
                        lat[y-20:y+21, x-20:x+21],
                        lon[y-20:y+21, x-20:x+21])
            dy, dx = np.unravel_index(np.argmin(variances), variances.shape)
            yc_n = yc + dy - 10; xc_n = xc + dx - 10
            if not 'latc' in fh.variables.keys():
                LATC = fh.createVariable('latc','f8',('theta'))
                LATC.long_name = "Latitude of cyclone center"
                LATC.units = "degrees_north"
                LATC.GRIB_shortName = "latc"
            else:
                LATC = fh['latc']
            if not 'lonc' in fh.variables.keys():
                LONC = fh.createVariable('lonc','f8',('theta'))
                LONC.long_name = "Longitude of cyclone center"
                LONC.units = "degrees_east"
                LONC.GRIB_shortName = "lonc"
            else:
                LONC = fh['lonc']
            LATC[:] = lat[yc_n,xc_n] * np.ones(len(tlu))
            LONC[:] = lon[yc_n,xc_n] * np.ones(len(tlu))
            fh.close()
        print("")
    else:
        raise NotImplementedError(f"Cannot load from {ext} file.")
    
def analyseFluxOscillation(rpath, d=None, r=None):
    # Read files and compute integrals/tendencies
    if d:   
        idstr = 'sq'+str(d).zfill(4)+'pt' # square domain
    elif r:
        idstr = 'r'+str(int(r/1000)).zfill(4)+'km' # cylindrical domain
    else:
        raise ExceptionError("Arguments d and r not specified.")

    readname = 'integrfluxes_'+idstr+'_???.npz'
#     figname = 'VorticityFluxes_'+idstr+'_bgdloc'
    filenames = sorted(glob.glob(rpath+idstr[:-2]+'/'+readname))

    # Initialize arrays
    fh = np.load(filenames[0])
    lvl = fh['lvl']
    dim0 = len(filenames) - 1
    dim1 = len(lvl)
    fh.close()
    
    PVSdV = np.zeros((dim0,dim1))
    JdA_mean = np.zeros((dim0,dim1))
    JdA_eddy = np.zeros((dim0,dim1))
    JdA_heat = np.zeros((dim0,dim1))

    # Read in data from files
    for fid,fname in enumerate(filenames[:-1]):
        print("\rReading file ",fid,end="")    
        fh = np.load(fname)
        PVSdV[fid] = fh['PVSdV']
        JdA_mean[fid] = fh['JdA_mean']
        JdA_eddy[fid] = fh['JdA_eddy']
        JdA_heat[fid] = fh['JdA_heat']
        fh.close()
    print("")

    ddtPVSdV = (PVSdV[1:] - PVSdV[:-1]) / 3600
    ddtPVSdV = np.append(ddtPVSdV,ddtPVSdV[-1][None,:],axis=0)
    fh.close()
    
    X,Y = np.meshgrid(np.arange(len(filenames)-1),lvl)
    Z   = np.array([JdA_mean[:,i] for i in range(ddtPVSdV.shape[1])])
#     plt.contourf(X,Y,Z)
#     plt.show()
    plt.plot(JdA_mean[:,10])
    plt.show()
    print(lvl[11])

    fft_freq = np.fft.fftfreq(dim0, 1) # 1/hour    
    fft_mean = np.abs(np.fft.fft(JdA_mean[:,11]-JdA_mean[:,11].mean()))

    plt.plot(fft_freq, fft_mean)
    plt.show()

def plotFluxes(rpath, fpath, d=None, r=None, savefig=False):
    """
    Plot integrated vorticity fluxes over all time and levels
    
    Parameters
    ----------
    rpath: str
        file containing flux data
    fpath: str
        file location for output figure
    d: int
        width of square integration boundary in grid points
    savefig: bool
        save the figure
    """
    # Read files and compute integrals/tendencies
    if d:   
        idstr = 'sq'+str(d).zfill(4)+'pt' # square domain
    elif r:
        idstr = 'r'+str(int(r/1000)).zfill(4)+'km' # cylindrical domain
    else:
        raise ExceptionError("Arguments d and r not specified.")
        
    readname = 'integrfluxes_'+idstr+'_???.npz'
    figname = 'VorticityFluxes_'+idstr+'_bgdloc'
    filenames = sorted(glob.glob(rpath+idstr[:-2]+'/'+readname))

    # Initialize arrays
    fh = np.load(filenames[0])
    lvl = fh['lvl']
    dim1 = len(lvl)
    fh.close()
    
    PVSdV = np.zeros((len(filenames)-1,dim1))
    JdA_mean = np.zeros((len(filenames)-1,dim1))
    JdA_eddy = np.zeros((len(filenames)-1,dim1))
    JdA_heat = np.zeros((len(filenames)-1,dim1))

    # Read in data from files
    for fid,fname in enumerate(filenames[:-1]):
        print("\rReading file ",fid,end="")    
        fh = np.load(fname)
        PVSdV[fid] = fh['PVSdV']
        JdA_mean[fid] = fh['JdA_mean']
        JdA_eddy[fid] = fh['JdA_eddy']
        JdA_heat[fid] = fh['JdA_heat']
        fh.close()
    print("")

    ddtPVSdV = (PVSdV[1:] - PVSdV[:-1]) / 3600
    ddtPVSdV = np.append(ddtPVSdV,ddtPVSdV[-1][None,:],axis=0)

    cmap = mp.cm.seismic
    cbar_label = r'$m^3/s^2$'
    clvls = np.arange(-4.25e6, 4.5e6, 0.25e6)
    #clvls = np.arange(-5e5, 5.01e5, 1e4)

    X, Y   =  np.meshgrid(np.arange(len(ddtPVSdV)), lvl)
    Z      =  np.array([ddtPVSdV[:,i] for i in range(ddtPVSdV.shape[1])])
    Z_mean = -np.array([JdA_mean[:,i] for i in range(JdA_mean.shape[1])])
    Z_eddy = -np.array([JdA_eddy[:,i] for i in range(JdA_eddy.shape[1])])
    Z_heat = -np.array([JdA_heat[:,i] for i in range(JdA_heat.shape[1])])

    Zmax = np.nanmax([Z,Z_mean,Z_eddy,Z_heat])
    Zmin = np.nanmin([Z,Z_mean,Z_eddy,Z_heat])
    Zamax = max(np.abs([Zmax,Zmin]))


    fig, ax = plt.subplots(2,3,figsize=(16,8),sharex=True, sharey=True)
    for axi in ax.ravel():
        axi.tick_params(axis='both', labelsize=14)
        axi.grid()
        axi.set_ylim([290,380])

    ax[0,0].set_title(r'$\frac{\partial PVS}{\partial t}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\theta$ [K]',fontsize=16)
    cf = ax[0,0].contourf(X,Y,Z, origin='lower', levels=clvls, cmap = cmap, extend='both')

    ax[0,1].set_title(r'$-\int J_{mean} \cdot dA$', fontsize=18)
    cf_mean = ax[0,1].contourf(X,Y,Z_mean, origin='lower', levels=clvls, cmap = cmap, extend='both')

    Z_sum = Z_mean + Z_eddy + Z_heat
    ax[0,2].set_title('sum vorticity fluxes')
    ax[0,2].contourf(X,Y,Z_sum, origin='lower', levels=clvls, cmap=cmap, extend='both')

    ax[1,0].set_title(r'$-\int J_{eddy} \cdot dA$', fontsize=18)
    ax[1,0].set_xlabel('t [hr]',fontsize=16)
    ax[1,0].set_ylabel(r'$\theta$ [K]',fontsize=16)
    cf_eddy = ax[1,0].contourf(X,Y,Z_eddy, origin='lower', levels=clvls, cmap = cmap, extend='both')

    ax[1,1].set_title(r'$-\int J_{heat} \cdot dA$', fontsize=18)
    ax[1,1].set_xlabel('t [hr]',fontsize=16)
    cf_heat = ax[1,1].contourf(X,Y,Z_heat, origin='lower', levels=clvls, cmap = cmap, extend='both')

    Z_diff = Z - Z_sum
    ax[1,2].set_title(r'$\frac{\partial PVS}{\partial t}$ - sum of fluxes',fontsize=18)
    ax[1,2].set_xlabel('t [hr]', fontsize=16)
    ax[1,2].contourf(X,Y,Z_diff,origin='lower',levels=clvls,cmap=cmap, extend='both')


    norm = mp.colors.BoundaryNorm(clvls,cmap.N)
    sm = mp.cm.ScalarMappable(norm=norm,cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), extend='both')
    cbar.set_label(cbar_label,fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    if savefig:
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        plt.savefig(fpath+figname)
    plt.show()

def plotFluxesMeady(rpath, fpath, d=None, r=None, savefig=False):
    """
    Plot integrated vorticity fluxes over all time and levels
    
    Parameters
    ----------
    rpath: str
        file containing flux data
    fpath: str
        file location for output figure
    d: int
        width of square integration boundary in grid points
    savefig: bool
        save the figure
    """
    # Read files and compute integrals/tendencies
    if d:   
        idstr = 'sq'+str(d).zfill(4)+'pt' # square domain
    elif r:
        idstr = 'r'+str(int(r/1000)).zfill(4)+'km' # cylindrical domain
    else:
        raise ExceptionError("Arguments d and r not specified.")
        
    readname = 'integrfluxes_'+idstr+'_???.npz'
    figname = 'VorticityFluxes_'+idstr+'_bgdloc'
    filenames = sorted(glob.glob(rpath+idstr[:-2]+'/'+readname))

    # Initialize arrays
    fh = np.load(filenames[0])
    lvl = fh['lvl']
    dim1 = len(lvl)
    fh.close()
    
    PVSdV = np.zeros((len(filenames)-1,dim1))
    JdA_mean = np.zeros((len(filenames)-1,dim1))
    JdA_eddy = np.zeros((len(filenames)-1,dim1))
    JdA_meady = np.zeros((len(filenames)-1,dim1))
    JdA_heat = np.zeros((len(filenames)-1,dim1))

    # Read in data from files
    for fid,fname in enumerate(filenames[:-1]):
        print("\rReading file ",fid,end="")    
        fh = np.load(fname)
        PVSdV[fid] = fh['PVSdV']
        JdA_mean[fid] = fh['JdA_mean']
        JdA_eddy[fid] = fh['JdA_eddy']
        JdA_meady[fid] = fh['JdA_meady']
        JdA_heat[fid] = fh['JdA_heat']
        fh.close()
    print("")

    ddtPVSdV = (PVSdV[1:] - PVSdV[:-1]) / 3600
    ddtPVSdV = np.append(ddtPVSdV,ddtPVSdV[-1][None,:],axis=0)

    cmap = mp.cm.seismic
    cbar_label = r'$m^3/s^2$'
    clvls = np.arange(-4.25e6, 4.5e6, 0.25e6)

    X, Y   =  np.meshgrid(np.arange(len(ddtPVSdV)), lvl)
    Z      =  np.array([ddtPVSdV[:,i] for i in range(ddtPVSdV.shape[1])])
    Z_mean = -np.array([JdA_mean[:,i] for i in range(JdA_mean.shape[1])])
    Z_eddy = -np.array([JdA_eddy[:,i] for i in range(JdA_eddy.shape[1])])
    Z_meady = -np.array([JdA_meady[:,i] for i in range(JdA_meady.shape[1])])
    Z_heat = -np.array([JdA_heat[:,i] for i in range(JdA_heat.shape[1])])

    Zmax = np.nanmax([Z,Z_mean,Z_eddy,Z_heat,Z_meady])
    Zmin = np.nanmin([Z,Z_mean,Z_eddy,Z_heat,Z_meady])
    Zamax = max(np.abs([Zmax,Zmin]))


    fig, ax = plt.subplots(2,3,figsize=(16,8),sharex=True, sharey=True)
    for axi in ax.ravel():
        axi.tick_params(axis='both', labelsize=14)
        axi.grid()

    ax[0,0].set_title(r'$\frac{\partial PVS}{\partial t}$', fontsize=18)
    ax[0,0].set_ylabel(r'$\theta$ [K]',fontsize=16)
    cf = ax[0,0].contourf(X,Y,Z, origin='lower', levels=clvls, cmap = cmap, extend='both')

    ax[0,1].set_title(r'$-\int J_{mean} \cdot dA$', fontsize=18)
    cf_mean = ax[0,1].contourf(X,Y,Z_mean, origin='lower', levels=clvls, cmap = cmap, extend='both')

    Z_sum = Z_meady #Z_mean + Z_eddy + Z_heat
    ax[0,2].set_title('sum vorticity fluxes')
    ax[0,2].contourf(X,Y,Z_sum, origin='lower', levels=clvls, cmap=cmap, extend='both')

    ax[1,0].set_title(r'$-\int J_{eddy} \cdot dA$', fontsize=18)
    ax[1,0].set_xlabel('t [hr]',fontsize=16)
    ax[1,0].set_ylabel(r'$\theta$ [K]',fontsize=16)
    cf_eddy = ax[1,0].contourf(X,Y,Z_eddy, origin='lower', levels=clvls, cmap = cmap, extend='both')

#     ax[1,1].set_title(r'$-\int J_{heat} \cdot dA$', fontsize=18)
    ax[1,1].set_title(r'$-\int J_{mean+eddy} \cdot dA$', fontsize=18)
    ax[1,1].set_xlabel('t [hr]',fontsize=16)
    cf_heat = ax[1,1].contourf(X,Y,Z_meady, origin='lower', levels=clvls, cmap = cmap, extend='both')

    Z_diff = np.nancumsum(Z - Z_sum, axis=1)
    ax[1,2].set_title(r'$\frac{\partial PVS}{\partial t}$ - sum of fluxes',fontsize=18)
    ax[1,2].set_xlabel('t [hr]', fontsize=16)
    ax[1,2].contourf(X,Y,Z_diff,origin='lower',levels=clvls,cmap=cmap, extend='both')


    norm = mp.colors.BoundaryNorm(clvls,cmap.N)
    sm = mp.cm.ScalarMappable(norm=norm,cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), extend='both')
    cbar.set_label(cbar_label,fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    if savefig:
        plt.savefig(fpath+figname)
    plt.show()
        
def plotSqBoundary(fname, lv=4, d=100):
    # Load data
    (tlu, tlv, tlp, glp, cimf, lat, lon, lvl, tim, latc, lonc, 
        latc_nxt, lonc_nxt) = readInIsentropicData(fname, d)
    
    # Calculate fields
    u0, v0   =  removeBGwind(tlu, tlv, latc, lonc, latc_nxt, lonc_nxt)
    rvort    =  relativevorticity(u0, v0, lat, lon)
    H, dH    =  isentropicHeight(tlp, glp, lvl)
    sigma    =  isentropicDensity(tlp, glp, lvl)
    dx, dy   =  latlon2dxdy(lat, lon)
    avort    =  rvort + fc
    dthdt    =  cimf/sigma
    dudth    =  np.gradient(u0, axis=0) / np.gradient(lvl)[:,None,None]
    dvdth    =  np.gradient(v0, axis=0) / np.gradient(lvl)[:,None,None]
    J_heat   =  dthdt * np.array([dvdth, -dudth])
    
    #---------------------- horizontal plot -----------------------------------
    plt.scatter(lon,lat,s=0.01, color='white') # show grid points
    # filled contour
    cf = plt.contourf(lon, lat, rvort[lv], cmap='gist_ncar', levels=20)
    cb = plt.colorbar(cf)
    cb.ax.set_title(r'$\zeta$ $[s^{-1}]$')
    # quiver plot
    i = 4 # quiver every ith data point 
    qv = plt.quiver(lon[::i,::i], lat[::i,::i], u0[lv,::i,::i], v0[lv,::i,::i])
    plt.quiverkey(qv, 1.1, -0.1, 50, '50 m/s')
    # contour plot
    c = plt.contour(lon, lat, H[lv], colors='k')
    plt.clabel(c, fmt='%1.0f')
    # plot settings
    xmin = max(lon[:,0])
    xmax = min(lon[:,-1])
    plt.xlim([max(lon[:,0]), min(lon[:,-1])])
    plt.ylim([max(lat[0,:]), min(lat[-1,:])])
    plt.xlabel('Longitude ($\degree$E)')
    plt.ylabel('Latitude ($\degree$N)')
    plt.title('Relative vorticity (shading), height AGL (contour) ' 
        + f'\nand wind velocity (arrow) at {lvl[lv]}K')
    # save and show
#     plt.savefig("/Users/jasperdejong/Documents/PhD_Tropical_Cyclones/Meetings/"
#         +f"figures/210708_eddyflux/RvHV_{lvl[lv]}K.pdf")
    plt.show()
    #---------------------- vertical plot -------------------------------------
    latcid, loncid = latlon2idx(lat, lon, latc, lonc)
    x,y = np.meshgrid(lon[latcid[4],:], lvl[:,None])
    cf = plt.contourf(x, y, cimf[:,latcid[4],:], 
        cmap='gist_ncar', levels=50)
    c0 = plt.contour(x, y, cimf[:,latcid[4],:], colors='k', linestyles='--', levels=0)
    cb = plt.colorbar(cf)
    cb.ax.set_title(r'CIMF [kg/m2s]')
    c = plt.contour(x, y, sigma[:,latcid[4],:], colors='k')
    plt.clabel(c, inline=True, inline_spacing=0, fmt='%1.0f')
    plt.xlim([max(lon[:,0]), min(lon[:,-1])])
    plt.xlabel('Longitude ($\degree$E)')
    plt.ylabel(r'$\theta$ [K]')
    plt.title(r'CIMF (shading), isentropic density (contour)')
    plt.show()
    
    # Evaluate field boundaries
    V_bnd  = sqboundary(u0, posdir='east') + sqboundary(v0, posdir='north')    
    av_bnd = sqboundary(avort)
    Juz_bnd = (sqboundary((u0*avort), posdir='east') 
        + sqboundary((v0*avort), posdir='north'))
    dH_bnd = sqboundary(dH)                                                    
    Jh_bnd = (sqboundary(J_heat[0], posdir='east') 
        + sqboundary(J_heat[1], posdir='north')) 
    dL_bnd = (sqboundary(dx, excl_edges=[0,2]) 
        + sqboundary(dy, excl_edges=[1,3]))   
    dA_bnd = dL_bnd * dH_bnd                                                

    # Evaluate vorticity flux
    J_mean = V_bnd.mean(axis=-1) * av_bnd.mean(axis=-1)
    J_eddy = ((V_bnd - V_bnd.mean(axis=-1)[:,None]) 
        * (av_bnd - av_bnd.mean(axis=-1)[:,None]))
    JdA_mean = (J_mean[:,None] * dA_bnd).sum(axis=-1)
    JdA_eddy = (J_eddy * dA_bnd).sum(axis=-1)
    JdA_meady = (Juz_bnd * dA_bnd).sum(axis=-1)
    JdA_heat = (Jh_bnd * dA_bnd).sum(axis=-1)

    # Evaluate PVS
    dA = dx * dy
    PVSdV = (avort * dA[None,:,:] * dH).reshape(len(avort),-1).sum(axis=1)

def readInIsentropicData(fname, d=100):
    """
    Read data from isentropic data files
    
    Parameters
    ----------
    fname: str
        path to datafile
    d: int
        slice data keeping an array of d x d gridpoints (lat x lon) 
        around the TC centre (latc, lonc)
    """
    
    ext = os.path.splitext(fname)[-1]
    if ext in ['.npy','.npz']:
        fh = np.load(fname)

        # Constrain dimensions (around TC centre on 315K)
        yc, xc = latlon2idx(fh['lat'], fh['lon'], fh['latc'][3], fh['lonc'][3])
        xlim = np.clip([xc-d//2, xc+round(d/2)], 0, fh['lat'].shape[1] - 1)
        ylim = np.clip([yc-d//2, yc+round(d/2)], 0, fh['lat'].shape[0] - 1)

        # Initialize variables
        tlu  = fh['tlu'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
        tlv  = fh['tlv'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
        tlp  = fh['tlp'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
        glp  = fh['glp'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
        cimf = fh['CIMF'][:,ylim[0]-1:ylim[1],xlim[0]-1:xlim[1]]
        cimf = (cimf[:,1:,1:] + cimf[:,:-1,:-1] 
            + cimf[:,:-1,1:] + cimf[:,1:,:-1])/4
        lat  = fh['lat'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
        lon  = fh['lon'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
        lvl  = fh['lvl']
        tim  = fh['tim']
        latc = fh['latc']
        lonc = fh['lonc']
        latc_nxt = fh['latc_nxt']
        lonc_nxt = fh['lonc_nxt']
    
        fh.close()
    
    elif ext in ['.nc']:
        fh = nc.Dataset(fname,"r")
                
        # Constrain dimensions (around TC centre on 315K)
        yc, xc = latlon2idx(fh['latitude'][:], fh['longitude'][:], 
            fh['latc'][3], fh['lonc'][3])
        xlim = np.clip([xc-d//2, xc+round(d/2)], 0, fh['latitude'].shape[1] - 1)
        ylim = np.clip([yc-d//2, yc+round(d/2)], 0, fh['latitude'].shape[0] - 1)

        # Initialize variables
        tlu  = fh['u'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
        tlu  = np.ma.array(tlu, mask=(tlu==fh['u'].GRIB_missingValue))
        tlv  = fh['v'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
        tlv  = np.ma.array(tlv, mask=(tlv==fh['v'].GRIB_missingValue))
        tlp  = fh['p'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]  
        tlp  = np.ma.array(tlp, mask=(tlp==fh['p'].GRIB_missingValue)) / 100 # code works with hPa for tlp instead of Pa
        glp  = fh['ps'][ylim[0]:ylim[1],xlim[0]:xlim[1]] # glp is in Pa
        glp  = np.ma.array(glp, mask=(glp==fh['ps'].GRIB_missingValue))
        cimf = fh['cimf'][:,ylim[0]-1:ylim[1],xlim[0]-1:xlim[1]]
        cimf = np.ma.array(cimf, mask=(cimf==fh['cimf'].GRIB_missingValue))
        cimf = np.ma.mean([cimf[...,:-1,:-1], cimf[...,1:,:-1], cimf[...,:-1,1:], cimf[...,1:,1:]], axis=0)
#         cimf = (cimf[:,1:,1:] + cimf[:,:-1,:-1] 
#             + cimf[:,:-1,1:] + cimf[:,1:,:-1])/4
        lat  = fh['latitude'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
        lon  = fh['longitude'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
        lvl  = fh['theta'][:]
        tim  = fh['step'][:] * 3600
        latc = fh['latc'][:]
        lonc = fh['lonc'][:]
        latc_nxt = fh['latc_nxt'][:]
        lonc_nxt = fh['lonc_nxt'][:]
        lat_s = fh['latitude_s'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
        lon_s = fh['longitude_s'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
    
        fh.close()
    
    else:
        raise NotImplementedError(f"Cannot load from {ext} file.")

#     # interpolate cimf to regular grid
#     regshape = cimf.shape[:-2] + tuple(x+1 for x in cimf.shape[-2:])
#     rcimf = np.ma.array(np.zeros(regshape), mask=np.zeros(regshape))
#     rcimf[...,1:-1,1:-1] = np.ma.mean([cimf[...,:-1,:-1], cimf[...,1:,:-1], cimf[...,:-1,1:], cimf[...,1:,1:]], axis=0)
#     rcimf[...,[0,-1],:] = rcimf[...,[1,-2],:]
#     rcimf[...,:,[0,-1]] = rcimf[...,:,[1,-2]]
#     cimf = rcimf

    return (tlu, tlv, tlp, glp, cimf, lat, lon, lat_s, lon_s, lvl, tim, 
        latc, lonc, latc_nxt, lonc_nxt)
    
def runSquareDomain(fname, wpath, d=100):
    """
    Calculate vorticity flux at and PVS within square domain of 
    length 2d from data
    
    Parameters
    ----------
    fname: str
        file location of the data file to read
    wpath: str
        folder location for output data
    d: int
        width of square domain [no. of grid cells | 1cell~3.2km]
    """

    # Load data
    (tlu, tlv, tlp, glp, cimf, lat, lon, lat_s, lon_s, lvl, tim, latc, lonc, 
        latc_nxt, lonc_nxt) = readInIsentropicData(fname, d)
        
    # Calculate fields
    u0, v0      =  removeBGwind(tlu, tlv, latc, lonc, latc_nxt, lonc_nxt)
    rvort       =  relativevorticity(u0, v0, lat, lon)
    _, dH       =  isentropicHeight(tlp, glp, lvl)
    sigma       =  isentropicDensity(tlp, glp, lvl)
    dx, dy      =  latlon2dxdy(lat, lon)
    avort       =  rvort + fc
    dthdt       =  cimf/sigma
    dudth       =  np.gradient(u0, axis=0) / np.gradient(lvl)[:,None,None]
    dvdth       =  np.gradient(v0, axis=0) / np.gradient(lvl)[:,None,None]
    J_heat      =  dthdt * np.array([dvdth, -dudth])
 
    # Evaluate field boundaries
    V_bnd  = sqboundary(u0, posdir='east') + sqboundary(v0, posdir='north')    
    av_bnd = sqboundary(avort)
    dH_bnd = sqboundary(dH)                                                    
    Jh_bnd = (sqboundary(J_heat[0], posdir='east') 
        + sqboundary(J_heat[1], posdir='north')) 
    dL_bnd = (sqboundary(dx, excl_edges=[0,2]) 
        + sqboundary(dy, excl_edges=[1,3]))   
    dA_bnd = dL_bnd * dH_bnd                                                

    # Evaluate vorticity flux
    J_mean = V_bnd.mean(axis=-1) * av_bnd.mean(axis=-1)
    J_eddy = ((V_bnd - V_bnd.mean(axis=-1)[:,None]) 
        * (av_bnd - av_bnd.mean(axis=-1)[:,None]))
    JdA_mean = (J_mean[:,None] * dA_bnd).sum(axis=-1)
    JdA_eddy = (J_eddy * dA_bnd).sum(axis=-1)
    JdA_heat = (Jh_bnd * dA_bnd).sum(axis=1)

    # Evaluate PVS
    dA = dx * dy
    PVSdV = (avort * dA[None,:,:] * dH).reshape(len(avort),-1).sum(axis=1)

    # Saving data
    writename = 'integrfluxes_sq{}pt_{}'.format(str(d).zfill(4), 
        str(int(tim/3600)).zfill(3))
    print("Saving ",writename, end='  ')
    ds = {
        'PVSdV'   : PVSdV,
        'JdA_mean': JdA_mean,
        'JdA_eddy': JdA_eddy,
        'JdA_heat': JdA_heat,
        'lvl'     : lvl
    }
    outdir = wpath + 'sq' + str(d).zfill(4) + '/' 
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    np.savez(outdir + writename, **ds)
    print("Done.")
    
def runCylindricalDomain(fname, wpath, rlim=50000, dr=5000):
    """
    Calculate vorticity flux at and PVS within cylindrical domain of 
    radius rlim
    
    Parameters
    ----------
    fname: str
        file location of the data file to read
    wpath: str
        folder location for output data
    rlim: float
        radius of integration domain [m]
    dr: float
        interval around rlim to look for data
    """
    
    # Load data
    d = np.ceil((2*rlim + dr)/3000).astype(int)
    (tlu, tlv, tlp, glp, cimf, lat, lon, lat_s, lon_s, lvl, tim, latc, lonc,
        latc_nxt, lonc_nxt) = readInIsentropicData(fname, d)
        
    # Copy latc(_nxt), lonc(_nxt) values for every isentropic level
    if latc.size == 1: latc = np.full(lvl.shape, latc.item())
    if lonc.size == 1: lonc = np.full(lvl.shape, lonc.item())
    if latc_nxt.size == 1: latc_nxt = np.full(lvl.shape, latc_nxt.item())
    if lonc_nxt.size == 1: lonc_nxt = np.full(lvl.shape, lonc_nxt.item())

    # Computing fluxes
    dist, angle = distance2centre(lat, lon, latc, lonc)
    tlu, tlv = removeBGwind(tlu, tlv, latc, lonc, latc_nxt, lonc_nxt)
    u_rad, v_tan = radialvelocity(tlu, tlv, angle)
    rv = relativevorticity(tlu, tlv, lat, lon)
    H, dH = isentropicHeight(tlp, glp, lvl)
    PVSdV = integratePVS(rv + fc, dH, lat, lon, latc, lonc, rlim)
    sigma = isentropicDensity(tlp,glp,lvl)
    J_heat = Jheating(cimf, sigma, v_tan, lvl)
    JdA_mean, JdA_eddy, JdA_heat = integrateJ(u_rad, v_tan, rv, dH, J_heat, 
        lat, lon, latc, lonc, lvl, rlim - dr/2, rlim + dr/2)
    
#     for var in [tlu, tlv, u_rad, v_tan, rv, H, dH, sigma, J_heat]:
#         plt.figure()
#         yc, xc = latlon2idx(lat, lon, latc[21], lonc[21])
#         d = 50
#         plt.contourf(var[21,yc-d:yc+d,xc-d:xc+d])
#         plt.colorbar()
#         plt.show()

    # Saving data
    writename = 'integrfluxes_r{}km_{}'.format(str(int(rlim/1000)).zfill(4), 
        str(int(tim/3600)).zfill(3))
    print("Saving ",writename, end='    ')
    ds = {
        'PVSdV': PVSdV,
        'JdA_mean': JdA_mean,
        'JdA_eddy': JdA_eddy,
        'JdA_heat': JdA_heat,
        'lvl'     : lvl
    }
    outdir = wpath + 'r' + str(int(rlim/1000)).zfill(4) + '/' 
    if not os.path.exists(outdir): os.mkdir(outdir)
    np.savez(outdir + writename, **ds)
    print("Done.")    
    
#----------------------------------------------------
#  ANALYSIS FUNCTIONS VORTICITY FLUX IN NETCDF FILES
#----------------------------------------------------

def takeAzimuthalMean(var, dist, rbnd=np.delete(np.arange(0,301,2),1)*1e3, rmin0=False):
    """
    Take azimuthal mean of variable var in rings around latc, lonc bounded by radii in rbnd. 
    If rmin0 == True, calculate mean in full circles of radii given by rbnd.
    """
    r = []
    var_rad = []
    #dist, angle = irma.distance2centre(lats, lons, latc, lonc)  
    for i,rmin in enumerate(rbnd[:-1]):
        if rmin0: rmin = 0
        rmax = rbnd[i+1]
        mask = (dist >= rmin) & (dist < rmax)
        r.append(dist[mask].mean())
        var_rad.append(var[...,mask].mean(axis=-1))

    # Fix r and mask unreliable data
    r = np.ma.array(r)
    r[r.mask] = ((rbnd[1:] + rbnd[:-1])/2)[r.mask] # replace masked values in r
    r = r.data
    var_rad = np.ma.array(var_rad).T
    
    return var_rad, r

def takeAzimuthalMeanEddyJ(u, zeta, dist, dH=None, rbnd=np.delete(np.arange(0,301,2),1)*1e3):
    """
    Take azimuthal mean and eddy component of vorticity flux around latc, lonc bounded by radii in rbnd
    """
    r = []
    u_mn = []
    z_mn = []
    uz_ed = []
    for i,rmin in enumerate(rbnd[:-1]):
        rmax = rbnd[i+1]
        mask = (dist >= rmin) & (dist < rmax)
        r.append(dist[mask].mean())
        u_mn.append(u[...,mask].mean(axis=-1))
        z_mn.append(zeta[...,mask].mean(axis=-1))
        if dH is None:
            uz_ed.append(((u-u_mn[-1][:,None,None])*(zeta-z_mn[-1][:,None,None]))[...,mask].mean(axis=-1))
        else:
            uz_ed.append(((u-u_mn[-1][:,None,None])*(zeta-z_mn[-1][:,None,None])*dH)[...,mask].mean(axis=-1))

    # Fix r and mask unreliable data
    r = np.ma.array(r)
    r[r.mask] = ((rbnd[1:] + rbnd[:-1])/2)[r.mask] # replace masked values in r
    r = r.data
    u_mn = np.ma.array(u_mn).T
    z_mn = np.ma.array(z_mn).T
    uz_ed = np.ma.array(uz_ed).T
    
    return u_mn * z_mn, uz_ed, r

def readNetCDF(fname, d=100):
    fh = nc.Dataset(fname,"r")

    # Constrain dimensions (around TC centre on 315K)
    yc, xc = latlon2idx(fh['latitude'][:], fh['longitude'][:], 
        fh['latc'][3], fh['lonc'][3])
    xlim = np.clip([xc-d//2, xc+round(d/2)], 0, fh['latitude'].shape[1] - 1)
    ylim = np.clip([yc-d//2, yc+round(d/2)], 0, fh['latitude'].shape[0] - 1)

    # Initialize variables
    tlu  = fh['u'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    tlu  = np.ma.array(tlu, mask=(tlu==fh['u'].GRIB_missingValue))
    tlv  = fh['v'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    tlv  = np.ma.array(tlv, mask=(tlv==fh['v'].GRIB_missingValue))
    tlp  = fh['p'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]  
    tlp  = np.ma.array(tlp, mask=(tlp==fh['p'].GRIB_missingValue)) / 100 # code works with hPa for tlp instead of Pa
    glp  = fh['ps'][ylim[0]:ylim[1],xlim[0]:xlim[1]] # glp is in Pa
    glp  = np.ma.array(glp, mask=(glp==fh['ps'].GRIB_missingValue))
    cimf = fh['cimf'][:,ylim[0]-1:ylim[1],xlim[0]-1:xlim[1]]
    cimf = np.ma.array(cimf, mask=(cimf==fh['cimf'].GRIB_missingValue))
    cimf = np.ma.mean([cimf[...,:-1,:-1], cimf[...,1:,:-1], cimf[...,:-1,1:], cimf[...,1:,1:]], axis=0)
    sig = fh['sigma'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    sig = np.ma.array(sig,mask=(sig==fh['sigma'].GRIB_missingValue))
    rv   = fh['rv'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    rv = np.ma.array(rv,mask=(rv==fh['rv'].GRIB_missingValue))
    av   = fh['av'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    av = np.ma.array(av,mask=(av==fh['av'].GRIB_missingValue))
    pv   = fh['pv'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    pv = np.ma.array(pv,mask=(pv==fh['pv'].GRIB_missingValue))
    u_rad   = fh['u_rad'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    u_rad = np.ma.array(u_rad,mask=(u_rad==fh['u_rad'].GRIB_missingValue))
    v_tan   = fh['v_tan'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    v_tan = np.ma.array(v_tan,mask=(v_tan==fh['v_tan'].GRIB_missingValue))
    w   = fh['w'][:,ylim[0]:ylim[1],xlim[0]:xlim[1]]
    w = np.ma.array(w,mask=(w==fh['w'].GRIB_missingValue))
    lat  = fh['latitude'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
    lon  = fh['longitude'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
    lvl  = fh['theta'][:]
    tim  = fh['step'][:] * 3600
    latc = fh['latc'][:]
    lonc = fh['lonc'][:]
    latc_nxt = fh['latc_nxt'][:]
    lonc_nxt = fh['lonc_nxt'][:]
    lat_s = fh['latitude_s'][ylim[0]:ylim[1],xlim[0]:xlim[1]]
    lon_s = fh['longitude_s'][ylim[0]:ylim[1],xlim[0]:xlim[1]]

    fh.close()

    return (tlu, tlv, tlp, glp, cimf, sig, rv, av, pv, u_rad, v_tan, w, lat, lon, lvl, tim, 
        latc, lonc, latc_nxt, lonc_nxt, lat_s, lon_s)

def ddtheta(f,lvl):
    dfdt = f.copy()
    lvl = lvl[:,None,None]
    dfdt[1:-1] = (f[2:] - f[:-2]) / (lvl[2:] - lvl[:-2])
    dfdt[[0,-1]] = (f[[1,-1]] - f[[0,-2]]) / (lvl[[1,-1]] - lvl[[0,-2]])
    return dfdt

def ddz(f,z):
    dfdz = f.copy()
    dfdz[1:-1] = (f[2:] - f[:-2]) / (z[2:] - z[:-2])
    dfdz[[0,-1]] = (f[[1,-1]] - f[[0,-2]]) / (z[[1,-1]] - z[[0,-2]])
    return dfdz

def vorticityFlux02(u_rad, v_tan, w, av, lvl, H, xi=0):
    """Radial vorticity flux using fixed mixing length (Prandtl)"""
    V = np.sqrt(u_rad**2 + v_tan**2)
    J1 = u_rad * av
    J2 = ddtheta(v_tan, lvl) * w
    J3 = -xi**2 * np.abs(ddz(V, H)) * ddz(v_tan, H)
    return J1, J2, J3

def vorticityFlux01(u_rad, v_tan, w, av, lvl, H, K=0):
    """Radial vorticity flux with constant eddy viscosity"""
    J1 = u_rad * av
    J2 = ddtheta(v_tan, lvl) * w
    J3 = -K * ddz(v_tan, H)
    return J1, J2, J3

def vorticityFlux00(u_rad, v_tan, w, av, lvl, K=None):
    """Radial vorticity flux with constant eddy viscosity 
    and d/dtheta instead of d/dz"""
    J1 = u_rad * av
    J2 = ddtheta(v_tan, lvl) * w
    if K:
        J3 = -K * ddtheta(v_tan, lvl)
        return J1, J2, J3
    return J1, J2

# def vorticityFlux(u_rad, v_tan, w, av, lvl):
#     J1 = u_rad * av
#     J2 = ddtheta(v_tan, lvl) * w
#     return J1, J2

def plotAmeanVorticityFlux(fname, d=100):
    (tlu, tlv, tlp, glp, cimf, rv, av, pv, u_rad, v_tan, w, lat, lon, lvl, tim, 
        latc, lonc, latc_nxt, lonc_nxt, lat_s, lon_s) = readNetCDF(fname, d=100)
    dist, angle = distance2centre(lat, lon, latc[0], lonc[0])
    J1, J2 = vorticityFlux(u_rad, v_tan, w, av, lvl)
    J1_mn, r1 = takeAzimuthalMean(J1, dist)
    J2_mn, r2 = takeAzimuthalMean(J2, dist)

    plt.contourf(r1,lvl,J1_mn,cmap='gnuplot')
    plt.show()
    return