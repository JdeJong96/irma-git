#!/usr/bin/python
"""This module interpolates the data from hybrid to isentropic levels.

This file will read the Harmonie data on hybrid levels and interpolate
them onto isentropic levels. The results are stored in separate files.
A basis for the interpolation procedure can be found in: "On the maintenance of 
potential vorticity in isentropic coordinates. Edouard et al. 
Q.J.R Meteorol. Soc. (1997), 123, pp 2069-2094".
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import sys
import glob
import numpy as np
from numba import float32, float64, guvectorize

p_ref = 100000
cp = 1004
Rdry = 286.9
kappa = Rdry/cp
dtdp_crit = -2e-4 # ignoring layers where dtheta/dp > dtdp_crit
lvls_pt = (np.cumsum([0] + [10/6] * 39 + [10/4] * 6 + [10/2] * 3 + [10] 
    + [50] * 2 + [100] * 3) + 295).astype('float64') # new theta levels
files = sorted(glob.glob("../Data/LambertGrid/629x989/fc2017090512+???.nc"))
saveloc = "../Data/LambertGrid/629x989interped/"

@guvectorize(
    "(float64[:], float64[:], float64[:], float32[:])",
    " (n), (n), (m) -> (m)",
    nopython=True,
)
def interp1d_gu(f, x, xi, out):
    """Interpolate field f(x) to xi in ln(x) coordinates."""
    i, imax, x0, f0 = 0, len(xi), x[0], f[0]
    while xi[i]<x0 and i < imax:
        out[i] = np.nan      
        i = i + 1 
    for x1,f1 in zip(x[1:], f[1:]):
        while xi[i] <= x1 and i < imax:
            out[i] = (f1-f0)/np.log(x1/x0)*np.log(xi[i]/x0)+f0
            i = i + 1
        x0, f0 = x1, f1
    while i < imax:
        out[i] = np.nan
        i = i + 1

@guvectorize(
    "(float64[:], float64[:], float64[:], float32[:])",
    " (n), (n), (m) -> (m)",
    nopython=True,
)
def interp1d_pres_gu(p, x, xi, out):
    """Interpolate field p(x) to xi in ln(x) coordinates."""
    i, imax, x0, p0 = 0, len(xi), x[0], p[0]
    while xi[i]<x0 and i < imax:
        out[i] = np.nan      
        i = i + 1 
    for x1,p1 in zip(x[1:], p[1:]):
        while xi[i] <= x1 and i < imax:
            gamma = np.log(p1/p0)/np.log(x1/x0)
            pi = p1 * (xi[i]/x1)**gamma
            out[i] = pi
            i = i + 1
        x0, p0 = x1, p1
    while i < imax:
        out[i] = np.nan
        i = i + 1
        
@guvectorize(
    "(float64[:], float64[:], float64[:])",
    " (n), (n) -> (n)",
    nopython=True,
)
def calc_dtdp_gu(t, p, dtdp):
    """Calculate dthetadp on hybrid levels."""
    i, imax, ln = 1, len(t), np.log
    dtdp[0] = (ln(t[2]/t[0])/ln(p[2]/p[0]) - ln(t[2]/t[1])/ln(p[2]/p[1]) 
               + ln(t[1]/t[0])/ln(p[1]/p[0])) * t[0]/p[0]
    if dtdp[0] > dtdp_crit: # failure of parabola fitting
        dtdp[0] = (t[1]-t[0])/(p[1]-p[0])
    while not np.isnan(t[i+1]) and i < (imax - 1):
        dtdp[i] =  t[i]/p[i] * (
            ln(p[i]/p[i-1])*ln(t[i+1]/t[i])/(ln(p[i+1]/p[i])*ln(p[i+1]/p[i-1]))
            + ln(p[i+1]/p[i])*ln(t[i]/t[i-1])/(ln(p[i]/p[i-1])*ln(p[i+1]/p[i-1]))
        )
        i = i + 1
    dtdp[i] = (ln(t[i]/t[i-2])/ln(p[i]/p[i-2]) + ln(t[i]/t[i-1])/ln(p[i]/p[i-1]) 
               - ln(t[i-1]/t[i-2])/ln(p[i-1]/p[i-2])) * t[i]/p[i]
    i = i + 1
    while i < imax:
        dtdp[i] = np.nan
        i = i + 1

def argsort3d(da, dim):
    m, n, k = da.shape
    ids = np.ogrid[:m,:n,:k]
    ax = da.dims.index(dim)
    ids[ax] = da.argsort(ax)
    return tuple(ids)

def stabilize(ds, theta, pres):
    """Mask theta values where dtheta/dp > dtdp_crit"""
    assert theta.dims == ('hybrid','y','x')
    assert pres.dims == ('hybrid','y','x')
    vars3d = [var for var in ds.data_vars if 'hybrid' in ds[var].dims and ds[var].ndim == 3]
    y,x = np.mgrid[:theta.shape[1],:theta.shape[2]]
    i0 = np.zeros(theta.shape[1:], dtype=int) 
    for i1 in range(1,len(theta)):
        t0, p0 = theta.data[i0,y,x], pres.data[i0,y,x]
        t1, p1 = theta.data[i1,:,:], pres.data[i1,:,:]
        dtdp = ((t1-t0)/(p1-p0))
        isInvalid = dtdp > dtdp_crit
        theta.data[i1][isInvalid] = np.nan
        for var3d in vars3d:
            ds[var3d].data[i1][isInvalid] = np.nan
        i0[~isInvalid] = i1
    return ds, theta

for file in files:
    ds = xr.open_dataset(file)
    pres = (ds.a + ds.b * ds.p0m)
    theta = (ds.t * (p_ref/pres)**kappa)
    ds, theta = stabilize(ds, theta, pres)
    ids = argsort3d(theta, 'hybrid')
    theta[:] = theta.data[ids]
    pres[:] = pres.data[ids]
    dtdp = xr.apply_ufunc(
        calc_dtdp_gu, theta, pres,
        input_core_dims=[['hybrid'], ['hybrid']], 
        output_core_dims=[['hybrid']], 
        output_dtypes=[theta.dtype],
    ).assign_attrs({'units':'K/Pa','long_name':'dtheta/dp'})
    ds['dtdp'] = dtdp#.transpose('hybrid','y','x')
    ds = ds.assign_coords({'theta':lvls_pt})
    ds['theta'] = ds.theta.assign_attrs(
        {"long_name":"potential temperature","units":"K"})
    vars3d = [ds[var] for var in ds.data_vars if ds[var].ndim==3]
    for var3d in vars3d:
        var3d = var3d.astype('float64').transpose('hybrid','y','x')
        if var3d.name != 'dtdp':
            var3d[:] = var3d.data[ids]
        ds[var3d.name] = xr.apply_ufunc(
            interp1d_gu,  var3d, theta, ds.theta,
            input_core_dims=[['hybrid'], ['hybrid'], ['theta']], 
            output_core_dims=[['theta']], 
            exclude_dims=set(('hybrid',)),  
            output_dtypes=['float32'],
        ).transpose('theta','y','x').assign_attrs(var3d.attrs)
    ds['pres'] = xr.apply_ufunc(
        interp1d_pres_gu,  pres, theta, ds.theta,
        input_core_dims=[['hybrid'], ['hybrid'], ['theta']], 
        output_core_dims=[['theta']], 
        exclude_dims=set(('hybrid',)),  
        output_dtypes=['float32'],
    ).transpose('theta','y','x').assign_attrs({'units':'Pa','long_name':'Pressure'})
    del ds['a']
    del ds['b']
    del ds['hybrid']
    savename = saveloc + os.path.basename(file)
    print(f"Saving {savename}")
    ds.to_netcdf(savename)
    ds.close()
#--------------------------OLD-VERSION------------------------------

# # Import modules
# import numpy as np
# import cfgrib
# import netCDF4 as nc
# import cartopy.crs as ccrs
# import cartopy.feature as cf
# import matplotlib.pyplot as plt
# import matplotlib
# import xarray as xr
# import datetime as dt
# import glob
# import sys
# import os

# # Assure working directory equals file directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # I/O settings
# wpath = "../Data/LambertGrid/629x989interp/"
# files = sorted(glob.glob('../Data/LambertGrid/629x989/fc2017090512+???.nc'))

# # Constants
# g = 9.81        # gravitational constant [m/s**2]
# p_ref = 100000  # reference pressure [Pa]
# cp = 1004       # isobaric heat capacity dry air [J/(kg K)]
# cv = 717        # isochoric heat capacity dry air [J/(kg K)]
# Rdry = 286.9    # specific gas constant dry air [J/(kg K)]
# Rwet = 461.4    # specific gas constant water vapour [J/(kg K)]
# kappa = Rdry/cp # [J/kg/K]
# eps = Rdry/Rwet # dimensionless constant
# dtdp_crit = -2e-4 # Maximum value dthetadp is allowed to have
# lim = 7.5       # Max curviness (deg) for using parabolic fitting dthetadp

# # New isentropic levels
# lvls_pt = np.cumsum([0] + [10/6] * 39 + [10/4] * 6 + [10/2] * 3 + [10] 
#     + [50] * 2 + [100] * 3) + 295

# # for convenience
# os.system('clear') # clear screen
# ln = np.log

# def interpolate(lvls_pt, t0, t1, f0, f1, fldname=None, mask=None):
#     """Interpolate to isentrope using data on adjacent hybrid levels"""
#     pt = lvls_pt[:,None,None] # new isentropic levels
#     if fldname == 'pres': # pressure interpolation
#         gamma = ln(f0/f1)/ln(t0/t1)
#         f_interp = f0 * (lvls_pt[:,None,None]/t0)**gamma
#     else:               # regular interpolation
#         f_interp = f0 - ln(t0/lvls_pt[:,None,None])/ln(t0/t1) * (f0 - f1)
#     if type(mask) != type(None):
#         f_interp = np.ma.array(f_interp, mask=mask)
#     return f_interp
    
# # Start main program
# for fid,fname in enumerate(filenames):
#     print(dt.datetime.now(), end="    ", flush=True)
    
#     # load variables
#     dsr = nc.Dataset(fname,"r")
#     ds_attrs = dsr.__dict__
#     time = dsr['time'][:];        time_ = dsr['time'].__dict__
#     step = dsr['step'][:];        step_ = dsr['step'].__dict__
#     hybrid = dsr['hybrid'][:];    hybrid_ = dsr['hybrid'].__dict__
#     lat = dsr['latitude'][:];     lat_ = dsr['latitude'].__dict__
#     lon = dsr['longitude'][:];    lon_ = dsr['longitude'].__dict__
#     vtime = dsr['valid_time'][:]; vtime_ = dsr['valid_time'].__dict__
#     t = dsr['t'][:];              t_ = dsr['t'].__dict__
#     u = dsr['u'][:];              u_ = dsr['u'].__dict__
#     v = dsr['v'][:];              v_ = dsr['v'].__dict__
#     q = dsr['q'][:];              q_ = dsr['q'].__dict__
#     a = dsr['a'][:];              a_ = dsr['a'].__dict__
#     b = dsr['b'][:];              b_ = dsr['b'].__dict__
#     ps = dsr['ps'][:];            ps_ = dsr['ps'].__dict__
#     ts = dsr['ts'][:];            ts_ = dsr['ts'].__dict__
#     dsr.close()

#     # Time of snapshot
#     dtime = (dt.datetime(1970,1,1) + dt.timedelta(seconds=int(time)) 
#         + dt.timedelta(hours=int(step)))

#     # Calculate pressure on hybrid levels
#     pres = a[:,None,None] + b[:,None,None] * ps[None,:,:]  # pressure [Pa]

#     # Calculate potential temperature on hybrid levels
#     theta = t * (p_ref/pres)**kappa               # pot. temp. [K]
#     theta_s = ts * (p_ref/ps)**kappa     # surface pot. temp [K]
    
#     theta3D = np.ma.array(theta, mask=np.zeros(theta.shape))
#     pres3D = pres.copy()
#     y,x = np.mgrid[0:theta3D.shape[1],0:theta3D.shape[2]]

#     # Mask invalid entries in theta3D (determined by dtdp_crit)
#     prev = np.zeros(theta3D.shape[1:], dtype=int) 
#     for h in range(1, len(theta3D)):
#         isInvalid = ((theta3D[h] - theta3D[prev,y,x])
#             / (pres3D[h] - pres3D[prev,y,x])) > dtdp_crit
#         theta3D.mask[h][isInvalid] = 1
#         prev[~isInvalid] = h

#     # Remove invalid entries and sort theta3D ascending
#     hi = len(theta3D) - np.sum(theta3D.mask, axis=0) # number of valid entries
#     hybrid_sort = np.argsort(theta3D, axis=0) # hybrid level indices for sorted theta 
#     theta3D_sort = theta3D[hybrid_sort,y,x]
    
#     # Calculate dthetadp on hybrid levels
#     dtdp_sort = np.ma.empty_like(theta3D_sort)
#     pres3D_sort = pres3D[hybrid_sort,y,x]
#     p0 = pres3D_sort[2:];  t0 = theta3D_sort[2:] 
#     p1 = pres3D_sort[1:-1]; t1 = theta3D_sort[1:-1]
#     p2 = pres3D_sort[:-2];   t2 = theta3D_sort[:-2]
#     A = (ln(p1/p2) * ln(t0/t1)) / (ln(p0/p1) * ln(p0/p2))
#     B = (ln(p0/p1) * ln(t1/t2)) / (ln(p1/p2) * ln(p0/p2))
#     dtdp_sort[1:-1,:,:] = t1/p1 * (A + B)  # estimate dtdp by fit
#     a1 = np.arctan2(ln(t1/t0),ln(p1/p0)) * 180/np.pi
#     a2 = np.arctan2(ln(t2/t1),ln(p2/p1)) * 180/np.pi
#     useLinear = np.abs(a2-a1) > lim # where to use linear interpolation
#     dtdp_sort[1:-1][useLinear] = ((t2[useLinear] - t0[useLinear]) \
#         / (p2[useLinear] - p0[useLinear]))
#     dtdp_sort[0] = ((theta3D_sort[1] - theta3D_sort[0]) \
#                     / (pres3D_sort[1] - pres3D_sort[0]))
#     dtdp_sort[hi-1,y,x] = ((theta3D_sort[hi-1,y,x] - theta3D_sort[hi-2,y,x]) \
#                 / (pres3D_sort[hi-1,y,x] - pres3D_sort[hi-2,y,x]))

#     # Sorted search for all isentropic levels
#     lvls_pt = np.sort(lvls_pt)
#     h = np.zeros(theta3D.shape[1:], dtype=int)
#     lvlids = np.empty((2, len(lvls_pt), theta3D.shape[1], theta3D.shape[2]), \
#         dtype=int)
#     for l,lvl in enumerate(lvls_pt):
#         isEnclosed = ((theta3D_sort[h,y,x] < lvl) 
#             & (lvl < theta3D_sort[(h+1).clip(max=hi-1),y,x])) 
#         isBelow = (lvl < theta3D_sort[h,y,x])
#         isAbove = (lvl > theta3D_sort[hi-1,y,x])
#         found = isEnclosed | isBelow | isAbove
#         while not found.all():
#             h[~found] += 1
#             isEnclosed = ((theta3D_sort[h,y,x] < lvl) \
#                 & (lvl < theta3D_sort[h+1,y,x]))
#             found += isEnclosed
#         h[isAbove] = (hi - 1)[isAbove]
#         h[isBelow] = -1 
#         lvlids[:,l,:,:] = np.array([h+1, h])[None,None,:,:]
#         lvlids[:,l,:,:][:,isBelow | isAbove] = -1
#         h[isBelow] = 0
#     lvlids[lvlids > hi - 1] = -1
      
#     # Select adjacent hybrid indices and dtdp
#     hybrid_a = hybrid_sort[lvlids,y,x]
#     hybrid_a[lvlids<0] = -1
#     dtdp_a = np.ma.array(dtdp_sort[lvlids,y,x], mask=(lvlids<0))
    
#     # Entries directly above and below each isentrope
#     i0, i1 = hybrid_a
#     t0, t1 = theta[hybrid_a,y,x]
#     p0, p1 = pres[hybrid_a,y,x]
    
#     # Interpolate hybrid level variables
#     t_i = interpolate(lvls_pt, t0, t1, t[i0,y,x], t[i1,y,x], mask=(i0==-1))
#     u_i = interpolate(lvls_pt, t0, t1, u[i0,y,x], u[i1,y,x], mask=(i0==-1))
#     v_i = interpolate(lvls_pt, t0, t1, v[i0,y,x], v[i1,y,x], mask=(i0==-1))
#     q_i = interpolate(lvls_pt, t0, t1, q[i0,y,x], q[i1,y,x], mask=(i0==-1))
#     p_i = interpolate(lvls_pt, t0, t1, pres[i0,y,x], pres[i1,y,x], 
#         fldname='pres', mask=(i0==-1))
#     dtdp_i = interpolate(lvls_pt, t0, t1, dtdp_a[0], dtdp_a[1], mask=(i0==-1))

#     # Calculate isentropic density
#     s_i = -1/g * dtdp_i**(-1)
    
#     # Set new attributes
#     for dic in [t_, u_, v_, q_]:
#         dic['GRIB_NV'] = len(lvls_pt)
#         dic['GRIB_typeOfLevel'] = 'theta'
#         dic['coordinates'] = dic['coordinates'].replace('hybrid','theta')
    
#     # Save new dataset
#     dsw = nc.Dataset(os.path.join(wpath, os.path.basename(fname)), "w")
#     print(f"Saving {dsw.filepath()}  ",end="", flush=True)

#     dsw.setncatts(ds_attrs)

#     tht_dim = dsw.createDimension('theta', len(lvls_pt))
#     lat_dim = dsw.createDimension('y', lat.shape[0])
#     lon_dim = dsw.createDimension('x', lat.shape[1])

#     STEP = dsw.createVariable('step', 'f8', ())
#     STEP[:] = step
#     step_.pop('_FillValue',0)
#     STEP.setncatts(step_)
#     STEP.GRIB_shortName = 'step'
    
#     THT = dsw.createVariable('theta','f8',('theta'))
#     THT[:] = lvls_pt.astype('f8')
#     tht_ = {'GRIB_shortName':'theta', 'GRIB_units':'K', 'GRIB_name':'theta',
#         'long_name':'potential temperature'}
#     THT.setncatts(tht_)

#     for var, var_, name in zip([time, vtime],[time_, vtime_], \
#         ['time','valid_time']):
#         VAR = dsw.createVariable(name, 'i8', ())
#         VAR[:] = var.astype('i8')
#         var_.pop('_FillValue',0)
#         VAR.setncatts(var_)
#         VAR.GRIB_shortName = name

#     for var,var_,name in zip([lat, lon, ps, ts],[lat_, lon_, ps_, ts_], \
#         ['latitude','longitude','ps','ts']):
#         VAR = dsw.createVariable(name, 'f8', ('y','x'))
#         VAR[:] = var.astype('f8')
#         var_.pop('_FillValue',0)
#         VAR.setncatts(var_)
#         VAR.GRIB_shortName = name

#     for var,var_ in zip([t_i,u_i,v_i,q_i],[t_,u_,v_,q_]):
#         VAR = dsw.createVariable(
#             var_['GRIB_shortName'], 'f4', ('theta','y','x'))
#         VAR[:] = var.filled(fill_value=var_['GRIB_missingValue']).astype('f4')
#         var_.pop('_FillValue',0)
#         VAR.setncatts(var_)
    
#     SIG = dsw.createVariable('sigma', 'f4', ('theta','y','x'))
#     s_ = u_.copy()
#     s_.pop('_FillValue',0)
#     s_.pop('GRIB_paramId',0)
#     s_['GRIB_NV'] = len(lvls_pt)
#     s_['GRIB_typeOfLevel'] = 'theta'
#     s_['GRIB_name'] = 'isentropic density'
#     s_['GRIB_shortName'] = 'sigma'
#     s_['GRIB_units'] = 'kg m**-2 K**-1'
#     s_['units'] = 'kg m**-2 K**-1'
#     s_['long_name'] = 'isentropic density'
#     s_['coordinates'] = s_['coordinates'].replace('hybrid','theta')
#     SIG[:] = s_i.filled(fill_value=s_['GRIB_missingValue']).astype('f4')
#     SIG.setncatts(s_)
    
#     PRES = dsw.createVariable('p', 'f4', ('theta','y','x'))
#     p_ = u_.copy()
#     p_.pop('_FillValue',0)
#     p_.pop('GRIB_paramId',0)
#     p_['GRIB_NV'] = len(lvls_pt)
#     p_['GRIB_typeOfLevel'] = 'theta'
#     p_['GRIB_name'] = 'pressure'
#     p_['GRIB_shortName'] = 'p'
#     p_['GRIB_units'] = 'Pa'
#     p_['units'] = 'Pa'
#     p_['long_name'] = 'pressure'
#     p_['coordinates'] = p_['coordinates'].replace('hybrid','theta')
#     PRES[:] = p_i.filled(fill_value=p_['GRIB_missingValue']).astype('f4')
#     PRES.setncatts(p_)


#     dsw.close()
#     print("done.", flush=True)