#!/usr/bin/python

"""Conversion of the HARMONIE output files to netCDF.

This module reads the grib output files, and stores selected variables
in new netCDF files.
"""

import xarray as xr
import os
import glob
import numpy as np

gribLoc = "/Volumes/ELEMENTS/PhD/IrmaHybrid/Data/GribFilesIrma/" # read path
ncdfLoc = "../Data/LambertGrid/629x989/" # write path
files = sorted(glob.glob(gribLoc+"fc2017090512+???grib"))
abFile = gribLoc + "A_B_65.dat" # hybrid coefficients file

# Read grib files
for file in files:
    print(f"Opening {os.path.basename(file)}...")
    ds0m = xr.open_dataset(file, engine='cfgrib',
        filter_by_keys={"typeOfLevel":"heightAboveGround",
                        "stepType":"instant","level":0},
        drop_variables=['sf','aers','aerl','aerc','aerd','ao','bo','co',
                        'rain','snow','grpl','swavr','lwavr','tcc','ccc',
                        'hcc','mcc','lcc','mld','xhail','z'])
    ds2m = xr.open_dataset(file, engine='cfgrib',
        filter_by_keys={"typeOfLevel":"heightAboveGround",
                        "stepType":"instant","level":2})
    ds10m = xr.open_dataset(file, engine='cfgrib',
        filter_by_keys={"typeOfLevel":"heightAboveGround",
                        "stepType":"instant","level":10})
    ds10ma = xr.open_dataset(file, engine='cfgrib',
        filter_by_keys={"typeOfLevel":"heightAboveGround",
                        "stepType":"accum","level":10})
    dsmsl = xr.open_dataset(file, engine='cfgrib',
        filter_by_keys={"typeOfLevel":"heightAboveSea",
                        "stepType":"instant","level":0})
    ds = xr.open_dataset(file, engine='cfgrib',
        filter_by_keys={"typeOfLevel":"hybrid","stepType":"instant"},
        drop_variables=['cwat','ciwc','snow','rain','grpl','tcc','pdep','vdiv'])
    
    # Add surface fields to hybrid level dataset
    ds['psl'] = dsmsl.pres
    ds['p0m'] = ds0m.pres
    ds['t0m'] = ds0m.t
    ds['lsm'] = ds0m.lsm
    ds['t2m'] = ds2m.t
    ds['r2m'] = ds2m.r
    ds['q2m'] = ds2m.q
    ds['u10'] = ds10m.u
    ds['v10'] = ds10m.v
    ds['u10gst'] = ds10ma.ugst
    ds['v10gst'] = ds10ma.vgst
    dsmsl.close()
    ds0m.close()
    ds2m.close()
    ds10m.close()
    ds10ma.close()
    ds = ds.drop('heightAboveSea')
    ds = ds.drop('heightAboveGround')

    # Read hybrid coefficients and add to dataset
    skiprows = 2
    hyb = np.zeros(65); A = np.zeros(65); B = np.zeros(65)
    with open(abFile,"r") as fh:
        for l,line in enumerate(fh):
            if l < skiprows: continue
            i = l - skiprows
            hyb[i], A[i], B[i] = line.strip('\n').split(None)
    ds['a'] = xr.DataArray(A, coords={"hybrid":ds.hybrid.data}, dims="hybrid",
        attrs={"long_name":"hybrid a coefficient",
               "standard_name":"atmosphere_hybrid_sigma_pressure_a_coefficient"})
    ds['b'] = xr.DataArray(B, coords={"hybrid":ds.hybrid.data}, dims="hybrid",
        attrs={"long_name":"hybrid b coefficient",
               "standard_name":"atmosphere_hybrid_sigma_pressure_b_coefficient"})
    
    # Write netCDF file
    writeName = ncdfLoc+os.path.basename(file).rstrip("grib")+".nc"
    print(f"Writing {writeName}")
    ds.to_netcdf(writeName)
    ds.close()
    
# Clean up read directory
for idFile in sorted(glob.glob(gribLoc+"*.idx")):
    print(f"Removing {os.path.basename(idFile)}")
    os.remove(idFile)