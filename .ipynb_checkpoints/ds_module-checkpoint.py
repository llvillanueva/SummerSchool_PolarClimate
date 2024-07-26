import numpy as np
import pandas as pd

from mpl_toolkits.basemap import Basemap

import iris 
import iris.plot as iplt
import iris.quickplot as qplt

from mpl_toolkits.basemap import Basemap, maskoceans, shiftgrid 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec

import scipy
from scipy import stats

from numpy.fft import *
from scipy.ndimage import gaussian_filter



def compute_mean(data_in):
    """
    compute for the mean
    """
    i_sum = 0
    for i in range (0,len(data_in)):
        if np.isnan(data_in[i]):
            continue
        i_sum += data_in[i]
    mean = i_sum/(len(data_in)-np.isnan(data_in).sum())
    return mean

def compute_std(data_in, mean):
    """
    compute for std
    """
    i_std = 0
    for i in range (0, len(data_in)):
        if np.isnan(data_in[i]):
            continue
        i_std += (data_in[i] - mean)**2 
    std = np.sqrt(i_std/(len(data_in)-np.isnan(data_in).sum()-1))
    return std

def my_function_to_detrend(data_in):
    times=np.arange(0,len(data_in.data),1)
  
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(times, data_in.data)
    
    data_in.data = data_in.data - (slope*times + intercept)
    return data_in

def plot_tseries(data_in, mean, stdev, label, linestyle, color, title):
    times=np.arange(0,len(data_in),1)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(times, data_in)
    line_fit= slope*times + intercept
    #plt.plot(data_in)
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Temperature (K)')
    plt.axhline(y=mean, c ='black', linestyle = ':', label=r'$\mu$')
    plt.plot(data_in, label=label)
    plt.plot(line_fit, c=color, linestyle=linestyle,linewidth=2, label = 'regression line')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    #Print p values and etc.
    print ('P_VALUE', p_value)
    print ('SLOPE', slope)
    print ('STD_ERR', std_err)
    print (' ')

def plot_tseriesv2(data_in, label, linestyle, color, xlabel, ylabel, title, date):

    fig=plt.figure(tight_layout=True)
    ax=plt.gca()

    time=np.arange(0, len(data_in) ,1)
    if date: 
        date= pd.date_range(start='1990-01-01', periods=len(time), freq='A')
        plt.plot(date, data_in, c=color, linestyle=linestyle, linewidth=3, label=label)
    else:
        plt.plot(time, data_in, c=color, linestyle=linestyle, linewidth=3, label=label) 

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend()

    return()

def compute_codxy(data_x, data_y, mean_x, mean_y):
    """
    compute for the Cod_xy
    """
    i_cod = 0
    for i in range (0,len(data_y)):
        if np.isnan(data_y[i]):
            continue
        i_cod += (data_y[i] - mean_y)*(data_x[i] - mean_x)
    cod_xy = i_cod
    return cod_xy

def compute_devx(data_in, mean):
    """
    compute for the Cod_xy
    """
    i_dev = 0
    for i in range (0,len(data_in)):
        if np.isnan(data_in[i]):
            continue
        i_dev += (data_in[i] - mean)**2
    dev_x = i_dev
    return dev_x

def compute_errvar(data_x, data_y, mean_x, mean_y):
    s2_err = 0
    t1 = 0
    n1 = 0
    d1 = 0
    for i in range (0,len(data_y)):
        if np.isnan(data_y[i]):
            continue
        t1 += (data_y[i] - mean_y)**2 
        n1 += (((data_y[i] - mean_y)**2)*((data_y[i] - mean_y)**2))
        d1 += (data_x[i] - mean_x)**2
    s2_err = t1 - n1/d1
    s2_err = s2_err/((len(data_y)-np.isnan(data_y).sum())-2)
    return s2_err

def compute_tcrit(df, alpha=0.05):
    """
    Find the critical t-value from the t-distribution table.
    """
    critical_value = scipy.stats.t.ppf(1 - alpha/2, df)
    return critical_value

def compute_annual(data_in):
    """
    Compute annual means from monthly means.
    """
    # Reshape the data to 12 months per year
    data_reshaped = data_in.reshape(-1, 12)
    # Compute the mean along the months axis
    annual_means = np.mean(data_reshaped, axis=1)
    return annual_means

def t_student(data_in):
    #our 'x' is time, let's define the x axis accordingly
    time=np.arange(0,len(data_in),1)
    #compute time mean
    mean_t=compute_mean(time)
    #compute data_in mean
    mean_data=compute_mean(data_in)
    #compute Cod_xy
    Cod_xy=compute_codxy(time, data_in, mean_t, mean_data)
    #Now compute Dev_x
    Dev_x= compute_devx(time, mean_t)
    #call a function to compute the error variance (s_err^2)
    df=len(data_in)-2 #define degrees of freedom as (n_x+n_y-2)
    err_var=compute_errvar(time, data_in, mean_t, mean_data)
    #Compute now the slope (b1)
    b1= Cod_xy/Dev_x
    #and the standard deviation of the residual of the slope (s_b1) -also known as standard error
    s_b1=np.sqrt(err_var/Dev_x)
    #Finally, compue the t-statistics: t=b1-beta_1/s_b1. Note: beta_1=0 because of our null hypothesis
    t_score=b1/s_b1
    t_crit = compute_tcrit(df)
    p_value = scipy.stats.t.sf(np.abs(t_score), df) * 2
    print ('SLOPE my_function:' , b1)
    print ('STD_ERR SLOPE my_function:' , s_b1)
    print ('t-score:' , t_score)
    print ('critical t-value:', t_crit)
    print("p-value:", p_value)
    return (b1, s_b1, t_score, p_value, t_crit)

def regrid(data_in, grid, binary=True):
    if binary:
        data_gridded=data_in.regrid(grid, iris.analysis.Nearest())
    else:
        data_gridded=data_in.regrid(grid, iris.analysis.Linear())
    #print (data_in, data_in_coarse)
    
    #let's check the regridding result
    '''
    plt.figure()
    qplt.contourf(data_in[0], 25)
    plt.figure()
    qplt.contourf(data_coarse[0], 25)
    plt.show()
    '''
    return data_gridded

def create_map(D1, title):
    fig=plt.figure()
    ax=plt.gca()

    bmap= Basemap(projection= 'gall', llcrnrlat= -90, urcrnrlat= 90, llcrnrlon=0, urcrnrlon= 360, resolution='l')
    
    lon= D1.coord('longitude').points
    lat= D1.coord('latitude').points
    
    x,y=bmap(*np.meshgrid(lon,lat))
    
    contours=bmap.contourf(x,y, D1.data, levels=20, cmap='jet')
    bmap.drawcoastlines()
    
    plt.colorbar()
    plt.title(title)
    plt.show()

def interpolationM(data_in, lat, lon):
    data_inter = {}
    data_inter['latitude'] = lat
    data_inter['longitude'] = lon
    data_inter['Sea Level Pressure (hPa) - Linear'] = []
    data_inter['Sea Level Pressure (hPa) - Nearest Neighbor'] = []
    for i in range(len(lat)): #21=number of sites
        site = [('latitude', lat[i] ), ('longitude', lon[i])]    
        data_inter['Sea Level Pressure (hPa) - Linear'].append(data_in.interpolate(site, iris.analysis.Linear()).data)
        data_inter['Sea Level Pressure (hPa) - Nearest Neighbor'].append(data_in.interpolate(site, iris.analysis.Nearest()).data)
    data_inter = pd.DataFrame(data_inter)
    return data_inter


def create_double_map(D1, D2, title1, title2):
    fig = plt.figure(figsize=(12, 6))

    # Define a grid layout with 1 row and 3 columns (2 for maps, 1 for colorbar)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    # Plot the first map
    ax1 = plt.subplot(gs[0])
    bmap1 = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360, resolution='l', ax=ax1)
    lon1 = D1.coord('longitude').points
    lat1 = D1.coord('latitude').points
    x1, y1 = bmap1(*np.meshgrid(lon1, lat1))
    contours1 = bmap1.contourf(x1, y1, D1.data, levels=20, cmap='jet')
    bmap1.drawcoastlines()
    ax1.set_title(title1)

    # Plot the second map
    ax2 = plt.subplot(gs[1])
    bmap2 = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360, resolution='l', ax=ax2)
    lon2 = D2.coord('longitude').points
    lat2 = D2.coord('latitude').points
    x2, y2 = bmap2(*np.meshgrid(lon2, lat2))
    contours2 = bmap2.contourf(x2, y2, D2.data, levels=20, cmap='jet')
    bmap2.drawcoastlines()
    ax2.set_title(title2)

    # Add a shared colorbar
    cbar_ax = plt.subplot(gs[2])
    cbar = fig.colorbar(contours2, cax=cbar_ax, orientation='vertical')

    plt.tight_layout()
    plt.show()

def extract_area(data_in, where):

    if where=='NA':
        lat_min=0
        lat_max=60
        lon_min=280
        lon_max=360

    if where=='NP':
        lat_min=20
        lat_max=70
        lon_min=120
        lon_max=250  
  
    if where=='SO':
        lat_min=-70
        lat_max=-50
        lon_min=0
        lon_max=360  

    if where=='NINO_3_4':
        lat_min=-5
        lat_max=5
        lon_min=190
        lon_max=240 


 #Define a geographical constraint based on the input coordinates 
    R=iris.Constraint(latitude=lambda  lat: lat_min <= lat <= lat_max, longitude= lambda lon: lon_min <= lon <= lon_max )  
    #Extract area 
    data_out=data_in.extract(R) 
 
 #'''
 #Plot selected area for visual inspection
    bmap=Basemap(projection= 'gall', llcrnrlat= lat_min,  urcrnrlat= lat_max, llcrnrlon= lon_min,  urcrnrlon= lon_max, resolution='l')
    lon= data_out.coord('longitude').points     
    lat= data_out.coord('latitude').points
    x,y=bmap(*np.meshgrid(lon,lat))
    contours=bmap.contourf(x,y, data_out[0,:,:].data, levels=80, cmap=matplotlib.cm.RdYlGn)
    bmap.drawcoastlines()
    #plt.figure(figsize=(16, 14))  # Adjust the values as per your requirement
     #plt.show()
     #'''
 
    return(data_out)

def compute_annual_nc(data):

 #compute annual means: 
    months=data.shape[0]
    cubes=iris.cube.CubeList()
    for i in range (12, months+1, 12):
        st=i-12
        cubes.append(data[st:i].collapsed('time', iris.analysis.MEAN))

    data_ann=cubes.merge_cube()
 #print data_ann
    return(data_ann)

def Plot_S_1D(S, nk, title, period):
 
    fig=plt.figure(figsize=(10,5))
    ax=plt.gca()

  
    if period:
        ax.stem(1/nk, S, linefmt='orange', markerfmt='o',label=' ') 
        ax.set_xlabel("Period (year)")
    else: 
        ax.stem(nk, S, linefmt='blue', markerfmt='o', label=' ')
        ax.set_xlabel('Frequency  (year$^{-1}$)')
  

    ax.set_ylabel('PWSD')
    ax.set_title(title)
  
    return() 

def area_weighted(data_in):
 #call function to compute annuals mean from monthly means
    data_in=compute_annual_nc(data_in) 
 
 #On a lat,long grid the grid-spacing reduces near the poles, we need to use area weights in our spatial mean to take into account
 #the irregularity of the grid. To compute the area-weighted spatial mean we get the area of each cell using cartography.area_weights.  
 #This uses the 'cell_area' attribute to calculate the area of each grid-box. 
    data_in.coord('latitude').guess_bounds()
    data_in.coord('longitude').guess_bounds()
    cell_area = iris.analysis.cartography.area_weights(data_in)

    data_out= data_in.collapsed(['latitude', 'longitude'],
                                 iris.analysis.MEAN,
                                  weights=cell_area)
 
    return(data_out)


def area_weighted_ENSO(data_in):
 #call function to compute annuals mean from monthly means
    #data_in=compute_annual_nc(data_in) 
 
 #On a lat,long grid the grid-spacing reduces near the poles, we need to use area weights in our spatial mean to take into account
 #the irregularity of the grid. To compute the area-weighted spatial mean we get the area of each cell using cartography.area_weights.  
 #This uses the 'cell_area' attribute to calculate the area of each grid-box. 
    data_in.coord('latitude').guess_bounds()
    data_in.coord('longitude').guess_bounds()
    cell_area = iris.analysis.cartography.area_weights(data_in)

    data_out= data_in.collapsed(['latitude', 'longitude'],
                                 iris.analysis.MEAN,
                                  weights=cell_area)
 
    return(data_out)

def my_function_to_compute_FFT_1D(data_in):
##Call function to compute the 1D FFT
    n=len(data_in)
    FT1D=fft(data_in, n)
#print (FT1D)
##Call function to compute the associated frequencies
    dt=1. #this is our sampling rate = 1 year, 1 month etc..
    nk=fftfreq(len(FT1D), dt) # Natural frequencies associated to each Fourier coefficient
#print (nk)
##Call function to shift the frequencies so that the zero frequency is in the middle of thearray
    FT1D = fftshift(FT1D)
    nk = fftshift(nk)

##Now select only positive (>0) frequencies. Note in this way we also exclude from the final array the zero-frequency term.
    positives= np.where(nk>0)
    FT1D=FT1D[positives] #take only positive freq (FFT for real input is symmetric)
    nk =nk[positives]
##Compute the power spectrum
    Spec_1D= np.absolute(FT1D)**2
##Find its maximum value
    maximum=np.amax(Spec_1D)
##Normalize the power spectrum relative to its maximum value
    Spec_1D_norm=Spec_1D/maximum
    return(Spec_1D_norm, nk)

def compute_ENSO(data_in):
#Compute area weighted mean
    data_in=area_weighted_ENSO(data_in)
    #compute a climatological mean (and its standard deviation) over the first 30 years
    mean=data_in[:30*12].collapsed('time', iris.analysis.MEAN)
    std_dev=data_in[:30*12].collapsed('time', iris.analysis.STD_DEV)
    
    #calculate anomalies:
    ENSO=data_in-mean
    
    #apply a 5-month running mean:
    months=ENSO.shape[0]
    ENSO_5month=iris.cube.CubeList()
    
    for i in range (0, months-5):
        ENSO_5month.append(ENSO[i:i+5].collapsed('time', iris.analysis.MEAN))
    
    ENSO_5month= ENSO_5month.merge_cube()

    ENSO_norm=ENSO_5month/std_dev
    return(ENSO_norm)


def plot_ENSO(data_set, label, title):
    fig=plt.figure(figsize=(12, 6), tight_layout=True)
    ax=plt.gca()

    months=len(data_set.data)
    times=np.arange(0,months,1)
    date= pd.date_range(start='1970-01-01', periods=len(times), freq='M')  #for monthly means 
 #date= pd.date_range(start='1990-01-01', periods=len(times), freq='A') #for annual means
    plt.plot(date, data_set.data, linewidth=3, label=label)
    #plt.axhline(y=0.4)
    #plt.axhline(y=-0.4)

    plt.fill_between(date, 0.4, np.ma.masked_where(data_set.data <= 0.4, data_set.data) , alpha=0.5, facecolor='red')
    plt.fill_between(date, -0.4, np.ma.masked_where(data_set.data >= -0.4, data_set.data) , alpha=0.5, facecolor='blue')
    
    plt.ylabel('C', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.show()