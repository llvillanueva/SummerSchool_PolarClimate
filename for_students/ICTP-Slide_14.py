#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Produce temperature cross section from NCEP donwload
@author: Paul J. Kushner for PHY492_1498F
Copyright Paul J. Kushner and  University of Toronto
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # Deals with netcdf files
T_t = 273.15 # triplepoint in K
p_0 = 1000 # reference pressure in hPa
g = 9.81 # m/s^2
R = 287 #J/kg/K
l=2.2e6
c_p = 1004

start_date = '1981-01-01'
#
end_date   = '2020-12-31'
url_base = 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/'

var_str = 'air'


url_name = url_base + var_str + '.mon.mean.nc'

dataset_air = xr.open_dataset(url_name)


var_str = 'hgt'

url_name = url_base + var_str + '.mon.mean.nc'

dataset_hgt = xr.open_dataset(url_name)


var_str = 'shum'

url_name = url_base + var_str + '.mon.mean.nc'

dataset_shum = xr.open_dataset(url_name)

mean_hgt = dataset_hgt.sel(time=slice(start_date,end_date)).mean(dim='time').mean(dim='lon')
mean_air = dataset_air.sel(time=slice(start_date,end_date)).mean(dim='time').mean(dim='lon')+T_t
mean_shum = dataset_shum.sel(time=slice(start_date,end_date)).mean(dim='time').mean(dim='lon')

mean_shum_array  = mean_shum.to_array().squeeze()
mean_hgt_array = mean_hgt.to_array().squeeze()
mean_air_array = mean_air.to_array().squeeze()

n_lev = (mean_shum_array.shape)[0]
theta = mean_air_array[:n_lev,:]*(1000/mean_air.level[:n_lev])**(2/7)# use the appropriate formula for theta here.
theta_e = theta[:n_lev,:]*np.exp(l*mean_shum_array[:n_lev,:]/1000 \
                                 /(c_p*mean_air_array[:,n_lev]))

DSE = c_p*mean_air_array[:n_lev,:]+g*mean_hgt_array[:n_lev,:]
MSE = DSE[:n_lev,:]+l*mean_shum_array[:n_lev,:]/1000
latitude = mean_air.lat
pressure = mean_air.level[:n_lev]
#for plotting purposes
#theta = theta.to_array().squeeze()
#theta_e = theta_e.to_array().squeeze()
#DSE = DSE.to_array().squeeze()
#MSE = MSE.to_array().squeeze()
plt.figure(1,figsize=(12,8))
plt.subplot(121)
c=plt.contour(latitude, pressure, theta, levels=np.arange(240,800,20),colors='k')
plt.clabel(c,fmt='%1.0f')
plt.ylim((pressure[0],pressure[-1]))
plt.title(r'$\theta$')
plt.xlabel('latitude')
plt.ylabel('pressure')
plt.grid(True)
plt.subplot(122)
c=plt.contour(latitude, pressure, theta_e, levels=np.arange(240,800,20),colors='k')
plt.clabel(c,fmt='%1.0f')
plt.ylim((pressure[0],pressure[-1]))
plt.title(r'$\theta_e$')
plt.xlabel('latitude')
plt.ylabel('pressure')
plt.grid(True)
plt.show()

plt.figure(2,figsize=(12,8))
plt.subplot(121)
c=plt.contour(latitude, pressure, DSE/1e4, levels=np.arange(0,5e5,1e4)/1e4,colors='k')
plt.clabel(c,fmt='%1.0f')
plt.ylim((pressure[0],pressure[-1]))
plt.title(r' DSE$/10^4$')
plt.xlabel('latitude')
plt.ylabel('pressure')
plt.grid(True)
plt.subplot(122)
c=plt.contour(latitude, pressure, MSE/1e4, levels=np.arange(0,5e5,1e4)/1e4,colors='k')
plt.clabel(c,fmt='%1.0f')
plt.ylim((pressure[0],pressure[-1]))
plt.title(r' MSE$/10^4$')
plt.xlabel('latitude')
plt.ylabel('pressure')
plt.grid(True)
plt.show()
