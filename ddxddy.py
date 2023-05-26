import numpy as np
from numba import jit, float32, float64, guvectorize, warnings
import xarray as xr

a = 6371000 # earth's radius [m]
warnings.filterwarnings('ignore',lineno=170)

@jit(nopython=True)
def _ddx2D(f, lat, lon, a=6371000):
    imax,jmax = lat.shape
    assert f.shape == (imax,jmax)
    dfdx = np.zeros(f.shape, dtype=f.dtype)
    for i in range(1,imax-1):
        for j in range(1,jmax//2): # lat increases with j
            cl = (lat[i,j]-lat[i,j-1])/(lat[i+1,j-1]-lat[i,j-1])
            cr = (lat[i,j]-lat[i-1,j+1])/(lat[i,j+1]-lat[i-1,j+1])
            fl = f[i,j-1] + cl*(f[i+1,j-1]-f[i,j-1])
            fr = f[i-1,j+1] + cr*(f[i,j+1]-f[i-1,j+1])
            fc = f[i,j]
            if np.isnan(fc) or (np.isnan(fl) and np.isnan(fr)):
                dfdx[i,j] = np.nan
                continue
            elif (not np.isnan(fl)) and (not np.isnan(fr)):
                df = fr - fl
                lonr = lon[i-1,j+1] + cr*(lon[i,j+1]-lon[i-1,j+1])
                lonl = lon[i,j-1] + cl*(lon[i+1,j-1]-lon[i,j-1])
                dlon = np.deg2rad(lonr-lonl)
            elif np.isnan(fl):
                df = fr - fc
                lonr = lon[i-1,j+1] + cr*(lon[i,j+1]-lon[i-1,j+1])
                dlon = np.deg2rad(lonr-lon[i,j])
            elif np.isnan(fr):
                df = fc - fl
                lonl = lon[i,j-1] + cl*(lon[i+1,j-1]-lon[i,j-1])
                dlon = np.deg2rad(lon[i,j]-lonl)
            dx = a * np.cos(np.deg2rad(lat[i,j])) * dlon
            dfdx[i,j] = df/dx
    for i in range(1,imax-1):
        for j in range(jmax//2,jmax-1): # lat decreases with j
            cl = (lat[i,j]-lat[i-1,j-1])/(lat[i,j-1]-lat[i-1,j-1])
            cr = (lat[i,j]-lat[i,j+1])/(lat[i+1,j+1]-lat[i,j+1])
            fl = f[i-1,j-1] + cl*(f[i,j-1]-f[i-1,j-1])
            fr = f[i,j+1] + cr*(f[i+1,j+1]-f[i,j+1])
            fc = f[i,j]
            if np.isnan(fc) or (np.isnan(fl) and np.isnan(fr)):
                dfdx[i,j] = np.nan
                continue
            elif (not np.isnan(fl)) and (not np.isnan(fr)):
                df = fr - fl
                lonr = lon[i,j+1] + cr*(lon[i+1,j+1]-lon[i,j+1])
                lonl = lon[i-1,j-1] + cl*(lon[i,j-1]-lon[i-1,j-1])
                dlon = np.deg2rad(lonr-lonl)
            elif np.isnan(fl):
                df = fr - fc
                lonr = lon[i,j+1] + cr*(lon[i+1,j+1]-lon[i,j+1])
                dlon = np.deg2rad(lonr-lon[i,j])
            elif np.isnan(fr):
                df = fc - fl
                lonl = lon[i-1,j-1] + cl*(lon[i,j-1]-lon[i-1,j-1])
                dlon = np.deg2rad(lon[i,j]-lonl)
            dx = a * np.cos(np.deg2rad(lat[i,j])) * dlon
            dfdx[i,j] = df/dx
    for j in range(1,jmax-1):
        if not np.isnan(dfdx[2,j]):
            dfdx[0,j] = 2*dfdx[1,j] - dfdx[2,j]
        else:
            dfdx[0,j] = dfdx[1,j]
        if not np.isnan(dfdx[-3,j]):
            dfdx[-1,j] = 2*dfdx[-2,j] - dfdx[-3,j]
        else:
            dfdx[-1,j] = dfdx[-2,j]
    for i in range(imax):
        if not np.isnan(dfdx[i,2]):
            dfdx[i,0] = 2*dfdx[i,1] - dfdx[i,2]
        else:
            dfdx[i,0] = dfdx[i,1]
        if not np.isnan(dfdx[i,-3]):
            dfdx[i,-1] = 2*dfdx[i,-2] - dfdx[i,-3]
        else:
            dfdx[i,-1] = dfdx[i,-2]
    return dfdx

@jit(nopython=True)
def _ddx3D(f, lat, lon, a=6371000):
    imax,jmax = lat.shape[-2:]
    assert f.shape[-2:] == (imax,jmax)
    dfdx = np.zeros(f.shape, dtype=f.dtype)
    for i in range(1,imax-1):
        for j in range(1,jmax//2): # lat increases with j
            cl = (lat[i,j]-lat[i,j-1])/(lat[i+1,j-1]-lat[i,j-1])
            cr = (lat[i,j]-lat[i-1,j+1])/(lat[i,j+1]-lat[i-1,j+1])
            for h in range(len(f)):
                fl = f[h,i,j-1] + cl*(f[h,i+1,j-1]-f[h,i,j-1])
                fr = f[h,i-1,j+1] + cr*(f[h,i,j+1]-f[h,i-1,j+1])
                fc = f[h,i,j]
                if np.isnan(fc) or (np.isnan(fl) and np.isnan(fr)):
                    dfdx[h,i,j] = np.nan
                    continue
                elif (not np.isnan(fl)) and (not np.isnan(fr)):
                    df = fr - fl
                    lonr = lon[i-1,j+1] + cr*(lon[i,j+1]-lon[i-1,j+1])
                    lonl = lon[i,j-1] + cl*(lon[i+1,j-1]-lon[i,j-1])
                    dlon = np.deg2rad(lonr-lonl)
                elif np.isnan(fl):
                    df = fr - fc
                    lonr = lon[i-1,j+1] + cr*(lon[i,j+1]-lon[i-1,j+1])
                    dlon = np.deg2rad(lonr-lon[i,j])
                elif np.isnan(fr):
                    df = fc - fl
                    lonl = lon[i,j-1] + cl*(lon[i+1,j-1]-lon[i,j-1])
                    dlon = np.deg2rad(lon[i,j]-lonl)
                dx = a * np.cos(np.deg2rad(lat[i,j])) * dlon
                dfdx[h,i,j] = df/dx
    for i in range(1,imax-1):
        for j in range(jmax//2,jmax-1): # lat decreases with j
            cl = (lat[i,j]-lat[i-1,j-1])/(lat[i,j-1]-lat[i-1,j-1])
            cr = (lat[i,j]-lat[i,j+1])/(lat[i+1,j+1]-lat[i,j+1])
            for h in range(len(f)):
                fl = f[h,i-1,j-1] + cl*(f[h,i,j-1]-f[h,i-1,j-1])
                fr = f[h,i,j+1] + cr*(f[h,i+1,j+1]-f[h,i,j+1])
                fc = f[h,i,j]
                if np.isnan(fc) or (np.isnan(fl) and np.isnan(fr)):
                    dfdx[h,i,j] = np.nan
                    continue
                elif (not np.isnan(fl)) and (not np.isnan(fr)):
                    df = fr - fl
                    lonr = lon[i,j+1] + cr*(lon[i+1,j+1]-lon[i,j+1])
                    lonl = lon[i-1,j-1] + cl*(lon[i,j-1]-lon[i-1,j-1])
                    dlon = np.deg2rad(lonr-lonl)
                elif np.isnan(fl):
                    df = fr - fc
                    lonr = lon[i,j+1] + cr*(lon[i+1,j+1]-lon[i,j+1])
                    dlon = np.deg2rad(lonr-lon[i,j])
                elif np.isnan(fr):
                    df = fc - fl
                    lonl = lon[i-1,j-1] + cl*(lon[i,j-1]-lon[i-1,j-1])
                    dlon = np.deg2rad(lon[i,j]-lonl)
                dx = a * np.cos(np.deg2rad(lat[i,j])) * dlon
                dfdx[h,i,j] = df/dx
    for j in range(1,jmax-1):
        for h in range(len(f)):
            if not np.isnan(dfdx[h,2,j]):
                dfdx[h,0,j] = 2*dfdx[h,1,j] - dfdx[h,2,j]
            else:
                dfdx[h,0,j] = dfdx[h,1,j]
            if not np.isnan(dfdx[h,-3,j]):
                dfdx[h,-1,j] = 2*dfdx[h,-2,j] - dfdx[h,-3,j]
            else:
                dfdx[h,-1,j] = dfdx[h,-2,j]
    for i in range(imax):
        for h in range(len(f)):
            if not np.isnan(dfdx[h,i,2]):
                dfdx[h,i,0] = 2*dfdx[h,i,1] - dfdx[h,i,2]
            else:
                dfdx[h,i,0] = dfdx[h,i,1]
            if not np.isnan(dfdx[h,i,-3]):
                dfdx[h,i,-1] = 2*dfdx[h,i,-2] - dfdx[h,i,-3]
            else:
                dfdx[h,i,-1] = dfdx[h,i,-2]
    return dfdx

def ddx(f, lat=None, lon=None):
    if isinstance(f, xr.DataArray):
        fc = f.compute()
        lat = fc.latitude.data if lat is None else lat
        lon = fc.longitude.data if lon is None else lon
        dfdx = xr.zeros_like(f, f.dtype)
        dfdx.data = ddx(fc.data, lat, lon)
        return dfdx
#     f = f.data if isinstance(f, xr.DataArray) else f
#     lat = lat.data if isinstance(lat, xr.DataArray) else lat
#     lon = lon.data if isinstance(lon, xr.DataArray) else lon
    fshape = f.shape
    *_,imax,jmax = lat.shape
    f = f.swapaxes(fshape.index(imax),-2).swapaxes(fshape.index(jmax),-1)
    if len(f.shape) == 2:
        dfdx = _ddx2D(f, lat, lon)
    elif len(f.shape) == 3:
        dfdx = _ddx3D(f, lat, lon)
    elif len(f.shape) > 3:
        dfdx = _ddx3D(f.reshape(-1,imax,jmax), lat.reshape(-1,imax,jmax), lon.reshape(-1,imax,jmax))
        dfdx = dfdx.reshape(fshape)
    else:
        return np.empty(fshape, dtype=f.dtype)
    dfdx = np.reshape(dfdx, fshape)
    return dfdx

@jit(nopython=True)
def _ddy2D(f, lat, lon, a=6371000):
    imax,jmax = lat.shape
    assert f.shape == (imax,jmax)
    dfdy = np.zeros(f.shape, dtype=f.dtype)
    for i in range(1,imax-1):
        for j in range(1,jmax//2): # lon decreases with i
            cb = (lon[i,j]-lon[i-1,j-1])/(lon[i-1,j]-lon[i-1,j-1])
            ct = (lon[i,j]-lon[i+1,j])/(lon[i+1,j+1]-lon[i+1,j])           
            fb = f[i-1,j-1] + cb*(f[i-1,j]-f[i-1,j-1])
            ft = f[i+1,j] + ct*(f[i+1,j+1]-f[i+1,j])
            fc = f[i,j]
            if np.isnan(fc) or (np.isnan(fb) and np.isnan(ft)):
                dfdy[i,j] = np.nan
                continue
            elif (not np.isnan(fb)) and (not np.isnan(ft)):
                df = ft - fb
                latb = lat[i-1,j-1] + cb*(lat[i-1,j]-lat[i-1,j-1])
                latt = lat[i+1,j] + ct*(lat[i+1,j+1]-lat[i+1,j])
                dlat = np.deg2rad(latt-latb)
            elif np.isnan(fb):
                df = ft - fc
                latt = lat[i+1,j] + ct*(lat[i+1,j+1]-lat[i+1,j])
                dlat = np.deg2rad(latt-lat[i,j])
            elif np.isnan(ft):
                df = fc - fb
                latb = lat[i-1,j-1] + cb*(lat[i-1,j]-lat[i-1,j-1])
                dlat = np.deg2rad(lat[i,j]-latb)
            dy = a * dlat
            dfdy[i,j] = df/dy
    for i in range(1,imax-1):
        for j in range(jmax//2,jmax-1): # lat decreases with j
            cb = (lon[i,j]-lon[i-1,j])/(lon[i-1,j+1]-lon[i-1,j])
            ct = (lon[i,j]-lon[i+1,j-1])/(lon[i+1,j]-lon[i+1,j-1])
            fb = f[i-1,j] + cb*(f[i-1,j+1]-f[i-1,j])
            ft = f[i+1,j-1] + ct*(f[i+1,j]-f[i+1,j-1])
            fc = f[i,j]
            if np.isnan(fc) or (np.isnan(fb) and np.isnan(ft)):
                dfdy[i,j] = np.nan
                continue
            elif (not np.isnan(fb)) and (not np.isnan(ft)):
                df = ft - fb
                latb = lat[i-1,j] + cb*(lat[i-1,j+1]-lat[i-1,j])
                latt = lat[i+1,j-1] + ct*(lat[i+1,j]-lat[i+1,j-1])
                dlat = np.deg2rad(latt-latb)
            elif np.isnan(fb):
                df = ft - fc
                latt = lat[i+1,j-1] + ct*(lat[i+1,j]-lat[i+1,j-1])
                dlat = np.deg2rad(latt-lat[i,j])
            elif np.isnan(ft):
                df = fc - fb
                latb = lat[i-1,j] + cb*(lat[i-1,j+1]-lat[i-1,j])
                dlat = np.deg2rad(lat[i,j]-latb)
            dy = a * dlat
            dfdy[i,j] = df/dy
    for j in range(1,jmax-1):
        if not np.isnan(dfdy[2,j]):
            dfdy[0,j] = 2*dfdy[1,j] - dfdy[2,j]
        else:
            dfdy[0,j] = dfdy[1,j]
        if not np.isnan(dfdy[-3,j]):
            dfdy[-1,j] = 2*dfdy[-2,j] - dfdy[-3,j]
        else:
            dfdy[-1,j] = dfdy[-2,j]
    for i in range(imax):
        if not np.isnan(dfdy[i,2]):
            dfdy[i,0] = 2*dfdy[i,1] - dfdy[i,2]
        else:
            dfdy[i,0] = dfdy[i,1]
        if not np.isnan(dfdy[i,-3]):
            dfdy[i,-1] = 2*dfdy[i,-2] - dfdy[i,-3]
        else:
            dfdy[i,-1] = dfdy[i,-2]
    return dfdy

@jit(nopython=True)
def _ddy3D(f, lat, lon, a=6371000):
    imax,jmax = lat.shape
    assert f.shape == (len(f),imax,jmax)
    dfdy = np.zeros(f.shape, dtype=f.dtype)
    for i in range(1,imax-1):
        for j in range(1,jmax//2): # lon decreases with i
            cb = (lon[i,j]-lon[i-1,j-1])/(lon[i-1,j]-lon[i-1,j-1])
            ct = (lon[i,j]-lon[i+1,j])/(lon[i+1,j+1]-lon[i+1,j]) 
            for h in range(len(f)):          
                fb = f[h,i-1,j-1] + cb*(f[h,i-1,j]-f[h,i-1,j-1])
                ft = f[h,i+1,j] + ct*(f[h,i+1,j+1]-f[h,i+1,j])
                fc = f[h,i,j]
                if np.isnan(fc) or (np.isnan(fb) and np.isnan(ft)):
                    dfdy[h,i,j] = np.nan
                    continue
                elif (not np.isnan(fb)) and (not np.isnan(ft)):
                    df = ft - fb
                    latb = lat[i-1,j-1] + cb*(lat[i-1,j]-lat[i-1,j-1])
                    latt = lat[i+1,j] + ct*(lat[i+1,j+1]-lat[i+1,j])
                    dlat = np.deg2rad(latt-latb)
                elif np.isnan(fb):
                    df = ft - fc
                    latt = lat[i+1,j] + ct*(lat[i+1,j+1]-lat[i+1,j])
                    dlat = np.deg2rad(latt-lat[i,j])
                elif np.isnan(ft):
                    df = fc - fb
                    latb = lat[i-1,j-1] + cb*(lat[i-1,j]-lat[i-1,j-1])
                    dlat = np.deg2rad(lat[i,j]-latb)
                dy = a * dlat
                dfdy[h,i,j] = df/dy
    for i in range(1,imax-1):
        for j in range(jmax//2,jmax-1): # lat decreases with j
            cb = (lon[i,j]-lon[i-1,j])/(lon[i-1,j+1]-lon[i-1,j])
            ct = (lon[i,j]-lon[i+1,j-1])/(lon[i+1,j]-lon[i+1,j-1])
            for h in range(len(f)):
                fb = f[h,i-1,j] + cb*(f[h,i-1,j+1]-f[h,i-1,j])
                ft = f[h,i+1,j-1] + ct*(f[h,i+1,j]-f[h,i+1,j-1])
                fc = f[h,i,j]
                if np.isnan(fc) or (np.isnan(fb) and np.isnan(ft)):
                    dfdy[h,i,j] = np.nan
                    continue
                elif (not np.isnan(fb)) and (not np.isnan(ft)):
                    df = ft - fb
                    latb = lat[i-1,j] + cb*(lat[i-1,j+1]-lat[i-1,j])
                    latt = lat[i+1,j-1] + ct*(lat[i+1,j]-lat[i+1,j-1])
                    dlat = np.deg2rad(latt-latb)
                elif np.isnan(fb):
                    df = ft - fc
                    latt = lat[i+1,j-1] + ct*(lat[i+1,j]-lat[i+1,j-1])
                    dlat = np.deg2rad(latt-lat[i,j])
                elif np.isnan(ft):
                    df = fc - fb
                    latb = lat[i-1,j] + cb*(lat[i-1,j+1]-lat[i-1,j])
                    dlat = np.deg2rad(lat[i,j]-latb)
                dy = a * dlat
                dfdy[h,i,j] = df/dy
    for j in range(1,jmax-1):
        for h in range(len(f)):
            if not np.isnan(dfdy[h,2,j]):
                dfdy[h,0,j] = 2*dfdy[h,1,j] - dfdy[h,2,j]
            else:
                dfdy[h,0,j] = dfdy[h,1,j]
            if not np.isnan(dfdy[h,-3,j]):
                dfdy[h,-1,j] = 2*dfdy[h,-2,j] - dfdy[h,-3,j]
            else:
                dfdy[h,-1,j] = dfdy[h,-2,j]
    for i in range(imax):
        for h in range(len(f)):
            if not np.isnan(dfdy[h,i,2]):
                dfdy[h,i,0] = 2*dfdy[h,i,1] - dfdy[h,i,2]
            else:
                dfdy[h,i,0] = dfdy[h,i,1]
            if not np.isnan(dfdy[h,i,-3]):
                dfdy[h,i,-1] = 2*dfdy[h,i,-2] - dfdy[h,i,-3]
            else:
                dfdy[h,i,-1] = dfdy[h,i,-2]
    return dfdy

def ddy(f, lat=None, lon=None):
    if isinstance(f, xr.DataArray):
        fc = f.compute()
        lat = fc.latitude.data if lat is None else lat
        lon = fc.longitude.data if lon is None else lon
        dfdy = xr.zeros_like(f, f.dtype)
        dfdy.data = ddy(fc.data, lat, lon)
        return dfdy
#     f = f.data if isinstance(f, xr.DataArray) else f
#     lat = lat.data if isinstance(lat, xr.DataArray) else lat
#     lon = lon.data if isinstance(lon, xr.DataArray) else lon
    fshape = f.shape
    *_,imax,jmax = lat.shape
    f = f.swapaxes(fshape.index(imax),-2).swapaxes(fshape.index(jmax),-1)
    if len(f.shape) == 2:
        dfdy = _ddy2D(f, lat, lon)
    elif len(f.shape) == 3:
        dfdy = _ddy3D(f, lat, lon)
    elif len(f.shape) > 3:
        dfdy = _ddy3D(f.reshape(-1,imax,jmax), lat, lon)
        dfdy = dfdy.reshape(fshape)
    else:
        return np.empty(fshape, dtype=f.dtype)
    dfdy = np.reshape(dfdy, fshape)
    return dfdy


@guvectorize(
    "float32[:,:], float32[:,:], float32[:,:], float32, float32[:,:]",
    "(y,x), (y,x), (y,x), () -> (y,x)", nopython=True)
def gu_ddx2D(f, lat, lon, a, dfdx):
    imax,jmax = lat.shape
    assert f.shape == (imax,jmax)
    #dfdx = np.zeros(f.shape, dtype=f.dtype)
    for i in range(1,imax-1):
        for j in range(1,jmax//2): # lat increases with j
            cl = (lat[i,j]-lat[i,j-1])/(lat[i+1,j-1]-lat[i,j-1])
            cr = (lat[i,j]-lat[i-1,j+1])/(lat[i,j+1]-lat[i-1,j+1])
            fl = f[i,j-1] + cl*(f[i+1,j-1]-f[i,j-1])
            fr = f[i-1,j+1] + cr*(f[i,j+1]-f[i-1,j+1])
            fc = f[i,j]
            if np.isnan(fc) or (np.isnan(fl) and np.isnan(fr)):
                dfdx[i,j] = np.nan
                continue
            elif (not np.isnan(fl)) and (not np.isnan(fr)):
                df = fr - fl
                lonr = lon[i-1,j+1] + cr*(lon[i,j+1]-lon[i-1,j+1])
                lonl = lon[i,j-1] + cl*(lon[i+1,j-1]-lon[i,j-1])
                dlon = np.deg2rad(lonr-lonl)
            elif np.isnan(fl):
                df = fr - fc
                lonr = lon[i-1,j+1] + cr*(lon[i,j+1]-lon[i-1,j+1])
                dlon = np.deg2rad(lonr-lon[i,j])
            elif np.isnan(fr):
                df = fc - fl
                lonl = lon[i,j-1] + cl*(lon[i+1,j-1]-lon[i,j-1])
                dlon = np.deg2rad(lon[i,j]-lonl)
            dx = a * np.cos(np.deg2rad(lat[i,j])) * dlon
            dfdx[i,j] = df/dx
    for i in range(1,imax-1):
        for j in range(jmax//2,jmax-1): # lat decreases with j
            cl = (lat[i,j]-lat[i-1,j-1])/(lat[i,j-1]-lat[i-1,j-1])
            cr = (lat[i,j]-lat[i,j+1])/(lat[i+1,j+1]-lat[i,j+1])
            fl = f[i-1,j-1] + cl*(f[i,j-1]-f[i-1,j-1])
            fr = f[i,j+1] + cr*(f[i+1,j+1]-f[i,j+1])
            fc = f[i,j]
            if np.isnan(fc) or (np.isnan(fl) and np.isnan(fr)):
                dfdx[i,j] = np.nan
                continue
            elif (not np.isnan(fl)) and (not np.isnan(fr)):
                df = fr - fl
                lonr = lon[i,j+1] + cr*(lon[i+1,j+1]-lon[i,j+1])
                lonl = lon[i-1,j-1] + cl*(lon[i,j-1]-lon[i-1,j-1])
                dlon = np.deg2rad(lonr-lonl)
            elif np.isnan(fl):
                df = fr - fc
                lonr = lon[i,j+1] + cr*(lon[i+1,j+1]-lon[i,j+1])
                dlon = np.deg2rad(lonr-lon[i,j])
            elif np.isnan(fr):
                df = fc - fl
                lonl = lon[i-1,j-1] + cl*(lon[i,j-1]-lon[i-1,j-1])
                dlon = np.deg2rad(lon[i,j]-lonl)
            dx = a * np.cos(np.deg2rad(lat[i,j])) * dlon
            dfdx[i,j] = df/dx
    for j in range(1,jmax-1):
        if not np.isnan(dfdx[2,j]):
            dfdx[0,j] = 2*dfdx[1,j] - dfdx[2,j]
        else:
            dfdx[0,j] = dfdx[1,j]
        if not np.isnan(dfdx[-3,j]):
            dfdx[-1,j] = 2*dfdx[-2,j] - dfdx[-3,j]
        else:
            dfdx[-1,j] = dfdx[-2,j]
    for i in range(imax):
        if not np.isnan(dfdx[i,2]):
            dfdx[i,0] = 2*dfdx[i,1] - dfdx[i,2]
        else:
            dfdx[i,0] = dfdx[i,1]
        if not np.isnan(dfdx[i,-3]):
            dfdx[i,-1] = 2*dfdx[i,-2] - dfdx[i,-3]
        else:
            dfdx[i,-1] = dfdx[i,-2]

@guvectorize(
    "float32[:,:], float32[:,:], float32[:,:], float32, float32[:,:]",
    "(y,x), (y,x), (y,x), () -> (y,x)", nopython=True)
def gu_ddy2D(f, lat, lon, a, dfdy):
    imax,jmax = lat.shape
    assert f.shape == (imax,jmax)
    for i in range(1,imax-1):
        for j in range(1,jmax//2): # lon decreases with i
            # if (lon[i-1,j]-lon[i-1,j-1]) == 0:
            #     print(i,j)
            # if (lon[i+1,j+1]-lon[i+1,j]) == 0:
            #     print(i,j)
            cb = (lon[i,j]-lon[i-1,j-1])/(lon[i-1,j]-lon[i-1,j-1])
            ct = (lon[i,j]-lon[i+1,j])/(lon[i+1,j+1]-lon[i+1,j])           
            fb = f[i-1,j-1] + cb*(f[i-1,j]-f[i-1,j-1])
            ft = f[i+1,j] + ct*(f[i+1,j+1]-f[i+1,j])
            fc = f[i,j]
            if np.isnan(fc) or (np.isnan(fb) and np.isnan(ft)):
                dfdy[i,j] = np.nan
                continue
            elif (not np.isnan(fb)) and (not np.isnan(ft)):
                df = ft - fb
                latb = lat[i-1,j-1] + cb*(lat[i-1,j]-lat[i-1,j-1])
                latt = lat[i+1,j] + ct*(lat[i+1,j+1]-lat[i+1,j])
                dlat = np.deg2rad(latt-latb)
            elif np.isnan(fb):
                df = ft - fc
                latt = lat[i+1,j] + ct*(lat[i+1,j+1]-lat[i+1,j])
                dlat = np.deg2rad(latt-lat[i,j])
            elif np.isnan(ft):
                df = fc - fb
                latb = lat[i-1,j-1] + cb*(lat[i-1,j]-lat[i-1,j-1])
                dlat = np.deg2rad(lat[i,j]-latb)
            dy = a * dlat
            dfdy[i,j] = df/dy
    for i in range(1,imax-1):
        for j in range(jmax//2,jmax-1): # lat decreases with j
            cb = (lon[i,j]-lon[i-1,j])/(lon[i-1,j+1]-lon[i-1,j])
            ct = (lon[i,j]-lon[i+1,j-1])/(lon[i+1,j]-lon[i+1,j-1])
            fb = f[i-1,j] + cb*(f[i-1,j+1]-f[i-1,j])
            ft = f[i+1,j-1] + ct*(f[i+1,j]-f[i+1,j-1])
            fc = f[i,j]
            if np.isnan(fc) or (np.isnan(fb) and np.isnan(ft)):
                dfdy[i,j] = np.nan
                continue
            elif (not np.isnan(fb)) and (not np.isnan(ft)):
                df = ft - fb
                latb = lat[i-1,j] + cb*(lat[i-1,j+1]-lat[i-1,j])
                latt = lat[i+1,j-1] + ct*(lat[i+1,j]-lat[i+1,j-1])
                dlat = np.deg2rad(latt-latb)
            elif np.isnan(fb):
                df = ft - fc
                latt = lat[i+1,j-1] + ct*(lat[i+1,j]-lat[i+1,j-1])
                dlat = np.deg2rad(latt-lat[i,j])
            elif np.isnan(ft):
                df = fc - fb
                latb = lat[i-1,j] + cb*(lat[i-1,j+1]-lat[i-1,j])
                dlat = np.deg2rad(lat[i,j]-latb)
            dy = a * dlat
            if dy == 0:
                print(i,j)
            dfdy[i,j] = df/dy
    for j in range(1,jmax-1):
        if not np.isnan(dfdy[2,j]):
            dfdy[0,j] = 2*dfdy[1,j] - dfdy[2,j]
        else:
            dfdy[0,j] = dfdy[1,j]
        if not np.isnan(dfdy[-3,j]):
            dfdy[-1,j] = 2*dfdy[-2,j] - dfdy[-3,j]
        else:
            dfdy[-1,j] = dfdy[-2,j]
    for i in range(imax):
        if not np.isnan(dfdy[i,2]):
            dfdy[i,0] = 2*dfdy[i,1] - dfdy[i,2]
        else:
            dfdy[i,0] = dfdy[i,1]
        if not np.isnan(dfdy[i,-3]):
            dfdy[i,-1] = 2*dfdy[i,-2] - dfdy[i,-3]
        else:
            dfdy[i,-1] = dfdy[i,-2]


def ddxND(f, lat=None, lon=None):
    """Calculate derivative with zonal distance on an N-dimensional DataArray"""
    lat = f.latitude if lat is None else lat
    lon = f.longitude if lon is None else lon
    if ('dy' in f.dims) and ('dx' in f.dims):
        f = f.transpose(...,'dy','dx')
        lat = lat.transpose(...,'dy','dx')
        lon = lon.transpose(...,'dy','dx')
    elif ('y' in f.dims) and ('x' in f.dims):
        f = f.transpose(...,'y','x')
        lat = lat.transpose(...,'y','x')
        lon = lon.transpose(...,'y','x')
    else:
        raise ValueError(f'Unknown dims: {f.dims}')
    dims = list(f.dims[-2:])
    dfdx = xr.apply_ufunc(
        gu_ddx2D,
        f.astype('float32'),
        lat.astype('float32'),
        lon.astype('float32'),
        np.float32(a),
        input_core_dims=[dims,dims,dims,[]],
        output_core_dims=[dims],
        dask="parallelized",
        output_dtypes=[f.dtype]
    ).compute()
    return dfdx

def ddyND(f, lat=None, lon=None):
    """Calculate derivative with zonal distance on an N-dimensional DataArray"""
    lat = f.latitude if lat is None else lat
    lon = f.longitude if lon is None else lon
    if ('dy' in f.dims) and ('dx' in f.dims):
        f = f.transpose(...,'dy','dx')
        lat = lat.transpose(...,'dy','dx')
        lon = lon.transpose(...,'dy','dx')
    elif ('y' in f.dims) and ('x' in f.dims):
        f = f.transpose(...,'y','x')
        lat = lat.transpose(...,'y','x')
        lon = lon.transpose(...,'y','x')
    else:
        raise ValueError(f'Unknown dims: {f.dims}')
    dims = list(f.dims[-2:])
    dfdy = xr.apply_ufunc(
        gu_ddy2D,
        f.astype('float32'),
        lat.astype('float32'),
        lon.astype('float32'),
        np.float32(a),
        input_core_dims=[dims,dims,dims,[]],
        output_core_dims=[dims],
        dask="parallelized",
        output_dtypes=[f.dtype]
    ).compute()
    return dfdy