#!/usr/bin/python3

"""Finding the TC center based on minimal azimuthal variance.

This module determines the x,y indices of the TC center. A first guess
consists of the location of minimum sea level pressure (psl). For each
member in a set of candidate points around the first guess, the azimuthal
variance of psl is estimated by taking the variance of the means of eight
equal-sized circular segments around the point. The method assumes a
cartesian grid, which is approximately true for inner core-sized regions
in the HARMONIE data.
"""

import xarray as xr
import os
import glob
import numpy as np

ncdfLoc = "../Data/LambertGrid/629x989/" # netCDF file folder
files = sorted(glob.glob(ncdfLoc+"fc2017090512+???.nc"))
n = 10 # search radius (# grid points from slp minimum)
d = 20 # used data radius (# grid points from each candidate)

# Create 8 'pizza slice' index fields
# (assumes rectangular m x m grid (m=2*d+1) for calculating azimuthal variance)
X,Y = np.meshgrid(np.arange(-d,d+1), np.arange(-d,d+1)) 
angle = np.arctan2(Y,X) # azimuth (radians)
r = np.sqrt(X**2 + Y**2) # radius (#grid points)
sliceIds = np.zeros((8,)+angle.shape, dtype=bool) # resulting index array
lims = np.arange(-4,5) * np.pi/4 # angle limits for 8 slices
for i,lim in enumerate(lims[:-1]):
    sliceIds[i] =((angle * (angle<lims[i+1]) * (angle>lim)) * (r<=d)) != 0

for file in files:
    with xr.open_dataset(file) as ds:
        # Calculate azimuthal variance minimum
        locmin = ds.psl.argmin(dim=('x','y')) # location of minimum slp
        xmin, ymin = (v.data for v in locmin.values())
        variances = np.empty((2*n+1,2*n+1))
        for i,dx in enumerate(range(-n,n+1)):
            for j,dy in enumerate(range(-n,n+1)):
                x0, y0 = xmin+dx, ymin+dy
                psl = ds.psl.isel(x=slice(x0-d,x0+d+1), y=slice(y0-d,y0+d+1))
                variances[i,j] = np.var([psl.data[slc].mean() for slc in sliceIds])
        ic,jc = np.unravel_index(np.argmin(variances), variances.shape)
        xc, yc = int(xmin+ic-n), int(ymin+jc-n) # location of minimal azimuthal variance
        print(f"File {os.path.basename(file)} : Minimum slp at (x={xmin},y={ymin})," 
              + f"\n\tminimum slp azimuthal variance at (x={xc},y={yc})")
        
        # Store netCDF
        ds['xc'] = xr.DataArray(xc, coords={}, dims=(),
        attrs={"long_name":"closest x index to TC center by minimizing azimuthal variance of psl",
               "standard_name":"x_of_tc_center"})
        ds['yc'] = xr.DataArray(yc, coords={}, dims=(),
        attrs={"long_name":"closest y index to TC center by minimizing azimuthal variance of psl",
               "standard_name":"y_of_tc_center"})
        writeName = file.rstrip(".nc") + "_new.nc"
        print(f"Writing {writeName}")
        ds.to_netcdf(writeName)
        print(f"Renaming to {file}")
        os.replace(writeName, file)
    ds.close()

# def findCenter(rnames):
#     """Find center of tropical cyclone by minimizing azimuthal variance of 
#     pressure field.
    
#     Parameters
#     ----------
#     rpath: str
#         path to file containing pressure field
    
#     Returns
#     -------
#     latc, lonc: float
#         central latitude and longitude of the cyclone, for which the azimuthal
#         variance of the pressure field is a minimum
#     """
#     def calc_variance(f, lats, lons):
#         """
#         Calculates a measure for the azimuthal variance of field f 
#         within ~65km (20gridpoints) 
#         """
#         assert f.shape[-1] == 41 and f.shape[-2] == 41
#         assert lats.shape == lons.shape and lats.shape == f.shape[-2:]
#         latc = lats[20,20]; lonc = lons[20,20]
#         dist, angle = distance2centre(lats, lons, latc, lonc)
#         dist, angle = np.squeeze(dist), np.squeeze(angle)
#         dphi = 0.2*np.pi
#         means = np.empty(10) #stores means of 'pizza slices' around center
#         for ii, phi in enumerate(np.linspace(0,2*np.pi-dphi,10)):
#             mask = (angle>=phi) * (angle<phi+dphi) * (dist<65000)
#             means[ii] = np.nanmean(f[mask])
#         return np.var(means)

        
#     rnames = sorted(rnames)
#     ext = os.path.splitext(rnames[0])[-1]
    
#     if ext in ['npy','npz']:
#         fig, ax = plt.subplots(3,3, figsize=(20,20))
    
#         for ii, rname in enumerate(rnames[::4]):
#             print(f"\rFinding TC center in file {ii+1}", end=' ')
#             fh = np.load(rname)
#             latc, lonc = fh['latc'][4], fh['lonc'][4]
#             lat, lon = fh['lat'], fh['lon']
#             yc, xc = latlon2idx(lat, lon, latc, lonc) #lat-lon index of center
#             variances = np.empty((21,21))
#             for dx in range(21):
#                 for dy in range(21):
#                     x = xc + dx - 10; y = yc + dy - 10
#                     variances[dy,dx] = calc_variance(
#                         fh['glp'][y-20:y+21, x-20:x+21],
#                         fh['lat'][y-20:y+21, x-20:x+21],
#                         fh['lon'][y-20:y+21, x-20:x+21])
#             dy, dx = np.unravel_index(np.argmin(variances), variances.shape)
#             yc_n = yc + dy - 10; xc_n = xc + dx - 10
        
#             #change y and x
#             dx, dy = 20, 20
#             lat = lat[yc-dy:yc+dy, xc-dx:xc+dx]
#             lon = lon[yc-dy:yc+dy, xc-dx:xc+dx]
#             tlu = fh['tlu'][:, yc-dy:yc+dy, xc-dx:xc+dx]
#             tlv = fh['tlv'][:, yc-dy:yc+dy, xc-dx:xc+dx]
#             tlp = fh['tlp'][:, yc-dy:yc+dy, xc-dx:xc+dx]
#             glp = fh['glp'][yc-dy:yc+dy, xc-dx:xc+dx]
#             cimf = fh['CIMF'][:, yc-dy:yc+dy, xc-dx:xc+dx]
#             sig = fh['sig'][:, yc-dy:yc+dy, xc-dx:xc+dx]
#             lvl = fh['lvl']
#             rv = relativevorticity(tlu, tlv, lat, lon)
#             X, Y = np.meshgrid(lon[yc_n-yc+dy//2,:], lvl)
#             cf = ax.flat[ii].contourf(X, Y, cimf[:,yc_n-yc+dy//2,:], cmap = 'seismic',
#                 levels=25, vmin = -4.25, vmax=4.25)
#     #         ax.flat[ii].quiver(lon, lat, tlu[4], tlv[4])
#             cs = ax.flat[ii].contour(X, Y, sig[:,yc_n-yc+15], colors='k')
#             ax.flat[ii].clabel(cs, inline=True, inline_spacing=0)
#     #         ax.flat[ii].axhline(latc)
#     #         ax.flat[ii].axvline(lonc)
#     #         ax.flat[ii].axhline(fh['lat'][yc_n,xc_n],linestyle='--')
#     #         ax.flat[ii].axvline(fh['lon'][yc_n,xc_n],linestyle='--')
#         fig.colorbar(cf, ax = ax.flat[-1])
#         fig.suptitle("Cross-isentropic mass flux (shading) and isentropic density")
#         plt.savefig("cimf_sig_latc.png")
#         plt.show()
#         print("")

#     elif ext in ['.nc']:
#         for ii,rname in enumerate(rnames):
#             print(f"\rFinding TC center in {os.path.basename(rname)}")
            
#             fh = nc.Dataset(rname,"r+")
#             tlu = fh['u'][:]
#             tlv = fh['v'][:]
#             tlp = fh['p'][:]
#             glp = fh['ps'][:]
#             lat = fh['latitude'][:]
#             lon = fh['longitude'][:]
#             rv = relativevorticity(tlu, tlv, lat, lon)
#             rv[:,:,:20] = 0 # filter out erroneous maxima
#             yc, xc = np.unravel_index(np.argmax(rv[12,:,:]), rv.shape[-2:])
#             print(f"yc: {yc}, xc: {xc} ")
#             variances = np.empty((21,21))
#             for dx in range(21):
#                 for dy in range(21):
#                     x = xc + dx - 10; y = yc + dy - 10
#                     variances[dy,dx] = calc_variance(
#                         glp[y-20:y+21, x-20:x+21],
#                         lat[y-20:y+21, x-20:x+21],
#                         lon[y-20:y+21, x-20:x+21])
#             dy, dx = np.unravel_index(np.argmin(variances), variances.shape)
#             yc_n = yc + dy - 10; xc_n = xc + dx - 10
#             if not 'latc' in fh.variables.keys():
#                 LATC = fh.createVariable('latc','f8',('theta'))
#                 LATC.long_name = "Latitude of cyclone center"
#                 LATC.units = "degrees_north"
#                 LATC.GRIB_shortName = "latc"
#             else:
#                 LATC = fh['latc']
#             if not 'lonc' in fh.variables.keys():
#                 LONC = fh.createVariable('lonc','f8',('theta'))
#                 LONC.long_name = "Longitude of cyclone center"
#                 LONC.units = "degrees_east"
#                 LONC.GRIB_shortName = "lonc"
#             else:
#                 LONC = fh['lonc']
#             LATC[:] = lat[yc_n,xc_n] * np.ones(len(tlu))
#             LONC[:] = lon[yc_n,xc_n] * np.ones(len(tlu))
#             fh.close()
#         print("")
#     else:
#         raise NotImplementedError(f"Cannot load from {ext} file.")
#---------------------------------------------------------------------------------
# # Fix faulty behaviour from old version grib2netcdf.py
# for file in files:
#     with xr.open_dataset(file, decode_times=False) as ds:
#         if 'units' in ds.a.attrs:
#             del ds.a.attrs['units']
#             print(f"Removed a.attrs['units'] from {os.path.basename(file)}")
#         if 'units' in ds.b.attrs:
#             del ds.b.attrs['units']
#             print(f"Removed b.attrs['units'] from {os.path.basename(file)}")
#         ds.to_netcdf(file+"edit", mode='w') # write to new file
#         os.replace(file+"edit", file) # rename to old file name (overwriting it)
#---------------------------------------------------------------------------------