import xarray as xr
import numpy as np
import cftime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

root_directory = '/home/esp-shared-a/Distribution/Workshops/PolarClimate_2024/atm_oc_seaice_projects/Project_1_2/'
exp1 = 'pa-pdSIC-ext'
exp2 = 'pa-futArcSIC-ext'
mon_to_average = 0 #Jan = 0, Feb = 1, Mar = 2, etc.
var_string = 'ua'
var_rel_path = '/WACCM/atm/Amon/'
mon_strings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
mon_average_string = mon_strings[mon_to_average]
#plot parameters
min_1, max_1 = -50,50
min_2, max_2 = min_1, max_1
dlevel_1 = (max_1-min_1)/20
dlevel_2 = dlevel_1
levels_1=np.arange(min_1,max_1+dlevel_1,dlevel_1)
levels_2=np.arange(min_2,max_2+dlevel_2,dlevel_2)
min_diff = -2
max_diff = -min_diff
dlevel_diff = (max_diff-min_diff)/10
levels_diff=np.arange(min_diff,max_diff+dlevel_diff,dlevel_diff)

def zm_climo_month(exp, var_rel_path, var_string, mon_to_average):
    data_directory = root_directory + exp + var_rel_path + var_string
    file_pattern = f'{data_directory}/*.nc'
    ds = xr.open_mfdataset(file_pattern, combine='by_coords')
    var = ds[var_string]
    lat = ds['lat']
    plev = ds['plev']
    climo = ds[var_string][mon_to_average::12].mean(dim='time')
    zm_climo = climo.mean(dim='lon')
    return zm_climo, lat, plev

def setup_plot():
    plt.figure(figsize=(6, 6))


zm_climo1, lat1, plev1 = zm_climo_month(exp1, var_rel_path, var_string, mon_to_average)
zm_climo2, lat2, plev2 = zm_climo_month(exp2, var_rel_path, var_string, mon_to_average)
diff = zm_climo2 - zm_climo1

ax = setup_plot()
plt.contourf(lat1, plev1, zm_climo1,levels = levels_1, cmap='bwr')
plt.colorbar()
plt.ylim([1000, 100])
plt.title(f'{mon_average_string} Climatological {var_string}, exp 1')
plt.show()
plt.savefig('zm_fig1.png')

ax = setup_plot()
plt.contourf(lat2, plev2, zm_climo2,levels = levels_2,cmap='bwr')
plt.colorbar()
plt.ylim([1000, 100])
plt.title(f'{mon_average_string} Climatological {var_string}, exp 2')
plt.show()
plt.savefig('zm_fig2.png')

ax=setup_plot()
plt.contourf(lat1, plev1, diff, levels = levels_diff,cmap='bwr')
plt.colorbar()
plt.ylim([1000, 100])
plt.title(f'Difference of {mon_average_string} Climatological {var_string}')
plt.show()
plt.savefig('zm_fig3.png')
