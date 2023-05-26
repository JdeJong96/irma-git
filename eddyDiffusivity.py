import xarray as xr
import matplotlib.pyplot as plt
import ipywidgets as widgets
import cartopy.crs as ccrs
import os
import sys
import glob
import cfgrib
import pygrib
import numpy as np
from numba import jit, float64

# Constants
b = 4
C1 = 0.61       # dimensionless constant virtual temperature
c0 = 3.75 
ch = 0.2
g = 9.81        # gravitational constant [m/s**2]
k = 0.40        # Von Karman constant
Rdry = 287.05   # specific gas constant dry air [J/(kg K)]
p_ref = 100000  # Reference pressure theta [Pa]
cp = 1004
Rdry = 286.9
linf = 75
p = 2
q = 2
cd = c0**(-2)
cn = cd**(1/4)
ac = 3*cn*k
ar = np.pi/2 * cn*k/(ac-cn*k)*b
an = cn*k
Ri_crit = b**(-1) # critical Richardson number
kappa = Rdry/cp

files = sorted(glob.glob("../Data/LambertGrid/629x989/fc2017090512+???.nc"))
for file in files:
    print(f'Adding eddy diffusivity to {file}...')
    with xr.open_dataset(file) as ds:
        ds = ds.assign_coords({'z':ds.height})
        pres = ds.a + ds.b * ds.p0m # pressure [Pa]
        tvir = ds.t * (1 + C1 * ds.q) # virtual temperature [K]
        V = np.sqrt(ds.u**2 + ds.v**2)
        dVdz = V.differentiate('hybrid') / ds.height.differentiate('hybrid')
        theta = (ds.t * (p_ref/pres)**kappa)
        dthetadz = theta.differentiate('hybrid') / ds.height.differentiate('hybrid')
        N = np.sqrt(g/theta * dthetadz.clip(min=0)) # Brunt Vaissala freq.
        Ri = (g/theta * dthetadz / (dVdz**2)) # Richardson number
        Fm = an - (2/np.pi * (ac-an)) * xr.where(Ri>0, ar*Ri, np.arctan(ar*Ri))
        cm = (ch * (1 + 2*Ri)).clip(max=3*ch)
        dz = abs(ds.height.diff('hybrid'))
        Fmi = Fm.rolling(hybrid=2).mean()
        lup = xr.zeros_like(ds.u)
        for h in ds.hybrid[63:0:-1]: # integrate upward (decreasing hybrid)
            Fmidz = Fmi.sel(hybrid=h+1) * dz.sel(hybrid=h+1)
            integrated = (lup.sel(hybrid=h+1).data + Fmidz.data)
            lup.loc[dict(hybrid=h)] = np.where(integrated<0, 0, integrated)
        ldw = xr.zeros_like(ds.u)
        for h in ds.hybrid[1::]: # integrate downward (increasing hybrid)
            Fmidz = Fmi.sel(hybrid=h) * dz.sel(hybrid=h)
            integrated = (ldw.sel(hybrid=h-1).data + Fmidz.data)
            ldw.loc[dict(hybrid=h)] = np.where(integrated<0, 0, integrated)
        lint = (lup**(-1) + ldw**(-1))**(-1)
        ls = cm * np.sqrt(ds.tke) / N
        lmin = (linf**(-1) + (0.5*cn*k*ds.height)**(-1))**(-1)
        lm = ((lint**q+lmin**q)**(-p/q) + ls**(-p))**(-1/p)
        lm = lm.assign_coords(z=ds.height)
        K = lm * np.sqrt(ds.tke)
        ds['Ke'] = K.assign_attrs({'long_name':'eddy diffusivity','units':'m**2 s**-1'})
        del ds['z']
        writeName = file.rstrip('.nc') + 'new.nc'
        ds.to_netcdf(writeName)
        os.replace(writeName, file)
        