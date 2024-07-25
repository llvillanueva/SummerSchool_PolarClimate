#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For PHY492_1498F-2021
Produce zonal-mean cross sections from NCEP download"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # Deals with netcdf files

# open multifile dataset from ncep using xr.open_dataset
def open_dataset(var_str, verbose=True):
    url_base = 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/'
    #url_base = './'
    url_name = url_base + var_str + '.mon.mean.nc'
    dataset = xr.open_dataset(url_name)
    if verbose:
        print('here is some information about the dataset')
        print(dataset)
    return dataset
    
#open dataset and return zonal and time mean given start and end dates
def get_zm_clim(var_str, start_date, end_date):
    dataset = open_dataset(var_str)
    lat = dataset['lat']
    pressure = dataset['level']
    zm_clim = dataset[var_str].sel(time=slice(start_date,end_date)). \
        mean(dim='time').mean(dim='lon')

    return zm_clim, lat, pressure



T_t = 273.15 # triplepoint in K
# first epoch
start_date_1 = '1981-01-01'
end_date_1   = '1990-12-31'


var_str = 'air'

air_zm_clim_1, lat, pressure = get_zm_clim(var_str, start_date_1, end_date_1)
air_zm_clim_1 += T_t

contour_levels_control = np.arange(190,330, 10)
cmap_control = plt.get_cmap('viridis')
#
plt.figure(1,figsize=(8,10))
plt.clf()
plt.contourf(lat, pressure, air_zm_clim_1, levels=contour_levels_control,\
               cmap = cmap_control)
plt.colorbar()
c=plt.contour(lat, pressure, air_zm_clim_1, levels=contour_levels_control,\
            colors='k')
plt.clabel(c,fmt='%3.0f')

# reverse the pressure axis, label the pressure axis
plt.ylim((pressure[0],pressure[-1]))
plt.ylabel('pressure')
plt.yscale('log')
plt.xlabel('latitude')
# give it a title
plt.title(r'T(K), {0} to {1}'.\
          format(start_date_1, end_date_1))

calc_diff = True
if calc_diff:
    # second epoch
    start_date_2 = '2011-01-01'
    end_date_2   = '2020-12-31'
    air_zm_clim_2, lat, pressure = get_zm_clim(var_str, start_date_2, end_date_2)
    air_zm_clim_2 += T_t
    diff = air_zm_clim_2-air_zm_clim_1
    contour_levels_diff=np.arange(-5, 5.5, 0.5)
    cmap_diff = plt.get_cmap('magma')

    plt.figure(2,figsize=(8,10))
    plt.clf()
    plt.contourf(lat, pressure, diff, levels=contour_levels_diff,\
               cmap = cmap_diff)
    plt.colorbar()
    c=plt.contour(lat, pressure, diff, levels=contour_levels_diff, colors='k')
#    plt.clabel(c,fmt='%2.1f')

    # reverse the pressure axis, label the pressure axis
    plt.ylim((pressure[0],pressure[-1]))
    plt.ylabel('pressure')
    plt.yscale('log')
    plt.xlabel('latitude')
    # give it a title
    plt.title('Temperature change(K),\n{2}-to-{3} minus {0}-to-{1}'. \
          format(start_date_1, end_date_1, start_date_2, end_date_2),\
              fontsize=10)
plt.show()
