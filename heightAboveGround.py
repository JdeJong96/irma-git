#!/usr/bin/python

"""Calculation of height on hybrid model levels.

This module calculates the height above ground level by integrating
the hydrostatic equation dz = -1/(rho*g)dp from the surface upwards,
using virtual temperature to account for vapor effects on density.
"""

import xarray as xr
import os
import glob

files = sorted(glob.glob('../Data/LambertGrid/629x989/fc2017090512+???.nc'))

g = 9.81        # gravitational constant [m/s**2]
Rdry = 287.05   # specific gas constant dry air [J/(kg K)]
C1 = 0.61       # dimensionless constant virtual temperature

for file in files:
    print(f'Adding height to {file}...')
    with xr.open_dataset(file) as ds:
        pres = ds.a + ds.b * ds.p0m # pressure [Pa]
        tvir = ds.t * (1 + C1 * ds.q) # virtual temperature [K]
        rho = pres / (Rdry * tvir) # density [kg/m**3]
        dpair = pres.diff(dim='hybrid',label='lower') # vertical press. difference
        dpsfc = ds.p0m - pres.isel(hybrid=-1) # difference with ground pressure
        dp = xr.concat((dpair, dpsfc), dim='hybrid') # total
        height = ((dp[::-1]/(rho*g)).cumsum('hybrid')[::-1]).astype('float32')
        if height.isnull().sum() > 0: 
            print('NaN values found.')
        height.name = 'height'
        height.attrs['long_name'] = 'height above surface'
        height.attrs['units'] = 'm'
        ds['height'] = height
        writeName = file.rstrip('.nc') + 'new.nc'
        ds.to_netcdf(writeName)
        os.replace(writeName, file)