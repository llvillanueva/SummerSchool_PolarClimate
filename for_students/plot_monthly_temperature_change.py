import xarray as xr
import numpy as np
import cftime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

root_directory = '/home/esp-shared-a/Distribution/Workshops/PolarClimate_2\
024/atm_oc_seaice_projects/Project_1_2/'
exp1 = 'pa-pdSIC-ext'
exp2 = 'pa-futArcSIC-ext'
mon_to_average = 0 #Jan = 0, Feb = 1, Mar = 2, etc.
var_string = 'tas'
var_rel_path = '/WACCM/atm/Amon/'
mon_strings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
mon_average_string = mon_strings[mon_to_average]
#plot parameters
vmin_1, vmax_1 = 220, 310
vmin_2, vmax_2 = vmin_1, vmax_1
vmin_diff = -7
vmax_diff = -vmin_diff
southern_lat_boundary = 30 #degrees north

def climo_month(exp, var_rel_path, var_string, mon_to_average):
    data_directory = root_directory + exp + var_rel_path + var_string
    print('data_directory = ',data_directory)
    file_pattern = f'{data_directory}/*.nc'
    ds = xr.open_mfdataset(file_pattern, combine='by_coords')
    var = ds[var_string]
    climo = ds[var_string][mon_to_average::12].mean(dim='time')
    return climo

def setup_plot():
    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    # Add coastlines and other features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # Add gridlines with labels for latitude and longitude
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    # Set extent to focus on the North Pole
    ax.set_extent([0, 360, southern_lat_boundary, 90], ccrs.PlateCarree())
    return ax

climo1 = climo_month(exp1, var_rel_path, var_string, mon_to_average)
climo2 = climo_month(exp2, var_rel_path, var_string, mon_to_average)
diff = climo2 - climo1

ax = setup_plot()
climo1.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='OrRd',cbar_kwargs={'label': var_string}, vmin=vmin_1, vmax=vmax_1)
plt.title(f'{mon_average_string} Climatological {var_string}, exp 1')
plt.show()
plt.savefig('fig1.png')

ax = setup_plot()
climo2.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='OrRd',cbar_kwargs={'label': var_string}, vmin=vmin_1, vmax=vmax_1)
plt.title(f'{mon_average_string} Climatological {var_string}, exp 2')
plt.show()
plt.savefig('fig2.png')

ax = setup_plot()
diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',cbar_kwargs={'label': var_string}, vmin=vmin_diff, vmax=vmax_diff)
plt.title(f'Difference of {mon_average_string} Climatological {var_string}')
plt.show()
plt.savefig('fig3.png')


