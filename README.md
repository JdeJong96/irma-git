# irma-git
A vorticity budget analysis of hurricane Irma (2017) at its most intense and quasi-steady mature stage

The source code is written to read output from the HARMONIE weather forecast model operated by the KNMI, de Bilt, and calculate various vorticity flux components. 

A short summary on the main scripts in this repository:
	ddxddy.py -- calculate horizontal central derivatives efficiently, single derivatives near missing values
	grib2netcdf.py -- read .grib files and write relevant variables in a netCDF file
	findCenter.py -- calculate the cyclone centre based on minimizing azimuthal variance of sea level pressure
	eddyDiffusivity.py -- apply HARATU turbulence scheme to derive eddy diffusivity
	heightAboveGround.py -- add hybrid level height based on balance assumptions
	interpolate.py -- interpolate from hybrid to isentropic levels
	irmaFn.py -- a collection of useful functions
	cimf.py -- calculate cross-isentropic mass flux and other fluxes