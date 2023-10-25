# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:55:30 2019

@author: jdt470
"""
import os
import netCDF4
import numpy as np
import pandas as pd
import geopy.distance
import operator
from math import sqrt
import sklearn.metrics as stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import datetime
import xarray as xr

mypath='/gpfs/work3/0/einf4186/gtsm'
os.chdir(mypath)

#cdo.sellonlatbox('-68,-58,13,21',input='maria_ref_surge/output/gtsm_model_0000_his.nc',output='maria_ref_surge/output/gtsm_model_0000_his_select.nc')
#cdo.sellonlatbox('-68,-58,13,21',input='maria_pgw_surge/output/gtsm_model_0000_his.nc',output='maria_pgw_surge/output/gtsm_model_0000_his_select.nc')
#cdo.sellonlatbox('-68,-58,13,21',input='maria_tp2_surge/output/gtsm_model_0000_his.nc',output='maria_tp2_surge/output/gtsm_model_0000_his_select.nc')

import matplotlib.colors as colors
import numpy as np
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('jet')
new_cmap = truncate_colormap(cmap, 0.2, 1.0)

ncREF = netCDF4.Dataset('maria_ref_surge/output/gtsm_model_0000_his_select.nc')
ncPGW = netCDF4.Dataset('maria_pgw_surge/output/gtsm_model_0000_his_select.nc')
ncTP2 = netCDF4.Dataset('maria_tp2_surge/output/gtsm_model_0000_his_select.nc')

dsREF=xr.open_dataset('maria_ref_surge/output/gtsm_model_0000_his_select.nc')
wind_REF = np.sqrt(dsREF.windx**2+dsREF.windy**2)
wind_REF_max = wind_REF.max(dim='time')

dsPGW=xr.open_dataset('maria_pgw_surge/output/gtsm_model_0000_his_select.nc')
wind_PGW = np.sqrt(dsPGW.windx**2+dsPGW.windy**2)
wind_PGW_max = wind_PGW.max(dim='time')

dsTP2=xr.open_dataset('maria_tp2_surge/output/gtsm_model_0000_his_select.nc')
wind_TP2 = np.sqrt(dsTP2.windx**2+dsTP2.windy**2)
wind_TP2_max = wind_TP2.max(dim='time')

patm_REF = ncREF.variables['patm'][:][:]
patm_PGW = ncPGW.variables['patm'][:][:]
patm_TP2 = ncTP2.variables['patm'][:][:]

waterlevel_REF = ncREF.variables['waterlevel'][:][:]
waterlevel_PGW = ncPGW.variables['waterlevel'][:][:]
waterlevel_TP2 = ncTP2.variables['waterlevel'][:][:]

longitude_REF = ncREF.variables['station_x_coordinate'][:]
longitude_PGW = ncPGW.variables['station_x_coordinate'][:]
longitude_TP2 = ncTP2.variables['station_x_coordinate'][:]

latitude_REF = ncREF.variables['station_y_coordinate'][:]
latitude_PGW = ncPGW.variables['station_y_coordinate'][:]
latitude_TP2 = ncTP2.variables['station_y_coordinate'][:]

surge_REF = waterlevel_REF
surge_PGW = waterlevel_PGW
surge_TP2 = waterlevel_TP2

#%%

select_REF = []
for i in range(len(latitude_REF)):
    standard_deviation = np.std(surge_REF[:,i])
    if standard_deviation > 0.001:
        true_false = True
    else:
        true_false = False
    select_REF.append(true_false)

select = np.asarray(select_REF)

surge_REF_select = surge_REF[:,select]
surge_PGW_select = surge_PGW[:,select]
surge_TP2_select = surge_TP2[:,select]

wind_REF_max_select = wind_REF_max.values[select]
wind_PGW_max_select = wind_PGW_max.values[select]
wind_TP2_max_select = wind_TP2_max.values[select]

patm_REF_select = patm_REF[:,select]
patm_PGW_select = patm_PGW[:,select]
patm_TP2_select = patm_TP2[:,select]

longitude_REF_select = longitude_REF[select]
longitude_PGW_select = longitude_PGW[select]
longitude_TP2_select = longitude_TP2[select]

latitude_REF_select = latitude_REF[select]
latitude_PGW_select = latitude_PGW[select]
latitude_TP2_select = latitude_TP2[select]


#%% load track data
#mypath='I://ICY_BOX//PhD//Data//Coordinates//'
#os.chdir(mypath)
#ibtracs = pd.read_excel('ibtracs.xlsx')
#IRMAIB = ibtracs[ibtracs.NAME == 'IRMA']
#data_to_extract = ['NAME','ISO_TIME','LAT','LON','WMO_WIND','WMO_PRES','USA_SSHS','USA_WIND','USA_PRES']
#irma = IRMAIB[data_to_extract].copy()

# cities for spatial reference in maps
#miami = -80.31666,25.78333
#cities = np.asarray([miami])
#%%

max_surge_REF=[]
for i in range(len(latitude_REF_select)):
    max_value = np.max(surge_REF_select[:,i])
    max_surge_REF.append(max_value)

max_surge_PGW=[]
for i in range(len(latitude_PGW_select)):
    max_value = np.max(surge_PGW_select[:,i])
    max_surge_PGW.append(max_value)

max_surge_TP2=[]
for i in range(len(latitude_TP2_select)):
    max_value = np.max(surge_TP2_select[:,i])
    max_surge_TP2.append(max_value)

min_patm_REF=[]
for i in range(len(latitude_REF_select)):
    min_value = np.min(patm_REF_select[:,i])
    min_patm_REF.append(min_value)

min_patm_PGW=[]
for i in range(len(latitude_PGW_select)):
    min_value = np.min(patm_PGW_select[:,i])
    min_patm_PGW.append(min_value)

min_patm_TP2=[]
for i in range(len(latitude_TP2_select)):
    min_value = np.min(patm_TP2_select[:,i])
    min_patm_TP2.append(min_value)

#%% plot erai era5 and difference
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('jet')
new_cmap = truncate_colormap(cmap, 0.2, 1.0)

#Dorian
#cdo.sellonlatbox('-82,-72,23,31',input='dorian_ref_surge/output/gtsm_model_0000_his.nc',output='dorian_ref_surge/output/gtsm_model_0000_his_select.nc')
#cdo.sellonlatbox('-82,-72,23,31',input='dorian_pgw_surge/output/gtsm_model_0000_his.nc',output='dorian_pgw_surge/output/gtsm_model_0000_his_select.nc')
#cdo.sellonlatbox('-82,-72,23,31',input='dorian_tp2_surge/output/gtsm_model_0000_his.nc',output='dorian_tp2_surge/output/gtsm_model_0000_his_select.nc')

ncREF = netCDF4.Dataset('dorian_ref_surge/output/gtsm_model_0000_his_select.nc')
ncPGW = netCDF4.Dataset('dorian_pgw_surge/output/gtsm_model_0000_his_select.nc')
ncTP2 = netCDF4.Dataset('dorian_tp2_surge/output/gtsm_model_0000_his_select.nc')

waterlevel_REF = ncREF.variables['waterlevel'][:][:]
waterlevel_PGW = ncPGW.variables['waterlevel'][:][:]
waterlevel_TP2 = ncTP2.variables['waterlevel'][:][:]

dsREF=xr.open_dataset('dorian_ref_surge/output/gtsm_model_0000_his_select.nc')
wind_REF = np.sqrt(dsREF.windx**2+dsREF.windy**2)
wind_REF_max_dorian = wind_REF.max(dim='time')

dsPGW=xr.open_dataset('dorian_pgw_surge/output/gtsm_model_0000_his_select.nc')
wind_PGW = np.sqrt(dsPGW.windx**2+dsPGW.windy**2)
wind_PGW_max_dorian = wind_PGW.max(dim='time')

dsTP2=xr.open_dataset('dorian_tp2_surge/output/gtsm_model_0000_his_select.nc')
wind_TP2 = np.sqrt(dsTP2.windx**2+dsTP2.windy**2)
wind_TP2_max_dorian = wind_TP2.max(dim='time')

patm_REF = ncREF.variables['patm'][:][:]
patm_PGW = ncPGW.variables['patm'][:][:]
patm_TP2 = ncTP2.variables['patm'][:][:]

longitude_REF = ncREF.variables['station_x_coordinate'][:]
longitude_PGW = ncPGW.variables['station_x_coordinate'][:]
longitude_TP2 = ncTP2.variables['station_x_coordinate'][:]

latitude_REF = ncREF.variables['station_y_coordinate'][:]
latitude_PGW = ncPGW.variables['station_y_coordinate'][:]
latitude_TP2 = ncTP2.variables['station_y_coordinate'][:]

surge_REF = waterlevel_REF
surge_PGW = waterlevel_PGW
surge_TP2 = waterlevel_TP2

#%%

select_REF = []
for i in range(len(latitude_REF)):
    standard_deviation = np.std(surge_REF[:,i])
    if standard_deviation > 0.001:
        true_false = True
    else:
        true_false = False
    select_REF.append(true_false)

select = np.asarray(select_REF)

surge_REF_select = surge_REF[:,select]
surge_PGW_select = surge_PGW[:,select]
surge_TP2_select = surge_TP2[:,select]

wind_REF_max_select_dorian = wind_REF_max_dorian.values[select]
wind_PGW_max_select_dorian = wind_PGW_max_dorian.values[select]
wind_TP2_max_select_dorian = wind_TP2_max_dorian.values[select]

patm_REF_select = patm_REF[:,select]
patm_PGW_select = patm_PGW[:,select]
patm_TP2_select = patm_TP2[:,select]

longitude_REF_select_dorian = longitude_REF[select]
longitude_PGW_select_dorian = longitude_PGW[select]
longitude_TP2_select_dorian = longitude_TP2[select]

latitude_REF_select_dorian = latitude_REF[select]
latitude_PGW_select_dorian = latitude_PGW[select]
latitude_TP2_select_dorian = latitude_TP2[select]


#%% load track data
#mypath='I://ICY_BOX//PhD//Data//Coordinates//'
#os.chdir(mypath)
#ibtracs = pd.read_excel('ibtracs.xlsx')
#IRMAIB = ibtracs[ibtracs.NAME == 'IRMA']
#data_to_extract = ['NAME','ISO_TIME','LAT','LON','WMO_WIND','WMO_PRES','USA_SSHS','USA_WIND','USA_PRES']
#irma = IRMAIB[data_to_extract].copy()

# cities for spatial reference in maps
#miami = -80.31666,25.78333
#cities = np.asarray([miami])
#%%

max_surge_REF_dorian=[]
for i in range(len(latitude_REF_select_dorian)):
    max_value = np.max(surge_REF_select[:,i])
    max_surge_REF_dorian.append(max_value)

max_surge_PGW_dorian=[]
for i in range(len(latitude_PGW_select_dorian)):
    max_value = np.max(surge_PGW_select[:,i])
    max_surge_PGW_dorian.append(max_value)

max_surge_TP2_dorian=[]
for i in range(len(latitude_TP2_select_dorian)):
    max_value = np.max(surge_TP2_select[:,i])
    max_surge_TP2_dorian.append(max_value)

min_patm_REF_dorian=[]
for i in range(len(latitude_REF_select_dorian)):
    min_value = np.min(patm_REF_select[:,i])
    min_patm_REF_dorian.append(min_value)

min_patm_PGW_dorian=[]
for i in range(len(latitude_PGW_select_dorian)):
    min_value = np.min(patm_PGW_select[:,i])
    min_patm_PGW_dorian.append(min_value)

min_patm_TP2_dorian=[]
for i in range(len(latitude_TP2_select_dorian)):
    min_value = np.min(patm_TP2_select[:,i])
    min_patm_TP2_dorian.append(min_value)


#%% plot erai era5 and difference
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('jet')
new_cmap = truncate_colormap(cmap, 0.2, 1.0)


#PLOT MAX SURGE
ax1 = plt.subplot2grid((40,60),(0,0),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax1.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax1.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
a=ax1.tripcolor(longitude_REF_select,latitude_REF_select,max_surge_REF,cmap=new_cmap,vmin=0,vmax=2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),['14N','17N','20N'],fontsize=6)
plt.title('IBTrACS',fontsize=6)
plt.ylabel('Hurricane Maria',fontsize=6)
ax1.text(-68.5,20.3,'a)',fontweight='bold',fontsize=6)

difference_PGW = np.asarray(max_surge_PGW)-np.asarray(max_surge_REF)

ax2 = plt.subplot2grid((40,60),(0,20),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax2.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax2.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
b=ax2.tripcolor(longitude_REF_select,latitude_REF_select,difference_PGW,cmap='hot_r',vmin=0,vmax=0.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),[],fontsize=6)
plt.title('?PGW - IBTrACS',fontsize=6)
ax2.text(-68.5,20.3,'b)',fontweight='bold',fontsize=6)

#ax4 = plt.subplot2grid((40,60),(38,0),colspan=15,rowspan=1)
#cbar4=plt.colorbar(a,cax=ax4,orientation='horizontal')
#cbar4.ax.tick_params(labelsize=7)
#plt.xlabel('Max surge height [m]',fontsize=7)

difference_TP2 = np.asarray(max_surge_TP2)-np.asarray(max_surge_REF)

ax3 = plt.subplot2grid((40,60),(0,40),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax3.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax3.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
ax3.tripcolor(longitude_REF_select,latitude_REF_select,difference_TP2,cmap='hot_r',vmin=0,vmax=0.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),[],fontsize=6)
plt.title('?TP2 - IBTrACS',fontsize=6)
ax3.text(-68.5,20.3,'c)',fontweight='bold',fontsize=6)

#ax5 = plt.subplot2grid((40,60),(38,30),colspan=15,rowspan=1)
#cbar5=plt.colorbar(b,cax=ax5,orientation='horizontal')
#cbar5.ax.tick_params(labelsize=7)
#plt.xlabel('? [m]',fontsize=7)

#plt.suptitle('Maximum surge height Hurricane Maria',fontsize=8)
#plt.savefig("/gpfs/work3/0/einf4186/gtsm/figures/GTSM_Maria_surge.png",bbox_inches='tight',dpi=600)
#plt.close()


ax4 = plt.subplot2grid((40,60),(15,0),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax4.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax4.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
a=ax4.tripcolor(longitude_REF_select_dorian,latitude_REF_select_dorian,max_surge_REF_dorian,cmap=new_cmap,vmin=0,vmax=2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),['25N','28N','31N'],fontsize=6)
plt.ylabel('Hurricane Dorian',fontsize=6)
#plt.title('IBTrACS',fontsize=7)
ax4.text(-82.5,31.3,'d)',fontweight='bold',fontsize=6)

difference_PGW_dorian = np.asarray(max_surge_PGW_dorian)-np.asarray(max_surge_REF_dorian)

ax5 = plt.subplot2grid((40,60),(15,20),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax5.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax5.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
b=ax5.tripcolor(longitude_REF_select_dorian,latitude_REF_select_dorian,difference_PGW_dorian,cmap='hot_r',vmin=0,vmax=0.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),[],fontsize=6)
#plt.title('?(RACMO-PGW - IBTrACS)',fontsize=7)
ax5.text(-82.5,31.3,'e)',fontweight='bold',fontsize=6)

ax6 = plt.subplot2grid((40,60),(31,0),colspan=17,rowspan=1)
cbar6=plt.colorbar(a,cax=ax6,orientation='horizontal',extend='max')
cbar6.ax.tick_params(labelsize=6)
plt.xlabel('Maximum surge height [m]',fontsize=6)

difference_TP2_dorian = np.asarray(max_surge_TP2_dorian)-np.asarray(max_surge_REF_dorian)

ax7 = plt.subplot2grid((40,60),(15,40),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax7.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax7.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
ax7.tripcolor(longitude_REF_select_dorian,latitude_REF_select_dorian,difference_TP2_dorian,cmap='hot_r',vmin=0,vmax=0.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),[],fontsize=6)
#plt.title('?(RACMO-TP2 - IBTrACS)',fontsize=7)
ax7.text(-82.5,31.3,'f)',fontweight='bold',fontsize=6)

ax8 = plt.subplot2grid((40,60),(31,30),colspan=17,rowspan=1)
cbar8=plt.colorbar(b,cax=ax8,orientation='horizontal',extend='max')
cbar8.ax.tick_params(labelsize=6)
plt.xlabel('Difference maximum surge height [m]',fontsize=6)

#plt.suptitle('Maximum surge height Hurricane Dorian',fontsize=6)
plt.savefig("/gpfs/work3/0/einf4186/eucp_knmi/scripts/tracking/Fig5_surge_gtsm_maria_dorain.png",dpi=600,bbox_inches='tight')
plt.close()

#PLOT MAX WIND
ax1 = plt.subplot2grid((40,60),(0,0),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax1.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax1.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
a=ax1.tripcolor(longitude_REF_select,latitude_REF_select,wind_REF_max_select,cmap=new_cmap,vmin=0,vmax=70)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),['14N','17N','20N'],fontsize=6)
plt.title('IBTrACS',fontsize=6)
plt.ylabel('Hurricane Maria',fontsize=6)
ax1.text(-68.5,20.3,'a)',fontweight='bold',fontsize=6)

difference_PGW = np.asarray(wind_PGW_max_select)-np.asarray(wind_REF_max_select)

ax2 = plt.subplot2grid((40,60),(0,20),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax2.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax2.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
b=ax2.tripcolor(longitude_REF_select,latitude_REF_select,difference_PGW,cmap='hot_r',vmin=0,vmax=5)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),[],fontsize=6)
plt.title('?PGW - IBTrACS',fontsize=6)
ax2.text(-68.5,20.3,'b)',fontweight='bold',fontsize=6)

difference_TP2 = np.asarray(wind_TP2_max_select)-np.asarray(wind_REF_max_select)

ax3 = plt.subplot2grid((40,60),(0,40),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax3.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax3.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
ax3.tripcolor(longitude_REF_select,latitude_REF_select,difference_TP2,cmap='hot_r',vmin=0,vmax=5)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),[],fontsize=6)
plt.title('?TP2 - IBTrACS',fontsize=6)
ax3.text(-68.5,20.3,'c)',fontweight='bold',fontsize=6)

ax4 = plt.subplot2grid((40,60),(15,0),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax4.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax4.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
a=ax4.tripcolor(longitude_REF_select_dorian,latitude_REF_select_dorian,wind_REF_max_select_dorian,cmap=new_cmap,vmin=0,vmax=70)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),['25N','28N','31N'],fontsize=6)
plt.ylabel('Hurricane Dorian',fontsize=6)
#plt.title('IBTrACS',fontsize=7)
ax4.text(-82.5,31.3,'d)',fontweight='bold',fontsize=6)

difference_PGW_dorian = np.asarray(wind_PGW_max_select_dorian)-np.asarray(wind_REF_max_select_dorian)

ax5 = plt.subplot2grid((40,60),(15,20),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax5.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax5.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
b=ax5.tripcolor(longitude_REF_select_dorian,latitude_REF_select_dorian,difference_PGW_dorian,cmap='hot_r',vmin=0,vmax=5)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),[],fontsize=6)
#plt.title('?(RACMO-PGW - IBTrACS)',fontsize=7)
ax5.text(-82.5,31.3,'e)',fontweight='bold',fontsize=6)

ax6 = plt.subplot2grid((40,60),(31,0),colspan=17,rowspan=1)
cbar6=plt.colorbar(a,cax=ax6,orientation='horizontal',extend='max')
cbar6.ax.tick_params(labelsize=6)
plt.xlabel('Maximum windspeed [m/s]',fontsize=6)

difference_TP2_dorian = np.asarray(wind_TP2_max_select_dorian)-np.asarray(wind_REF_max_select_dorian)

ax7 = plt.subplot2grid((40,60),(15,40),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax7.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax7.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
ax7.tripcolor(longitude_REF_select_dorian,latitude_REF_select_dorian,difference_TP2_dorian,cmap='hot_r',vmin=0,vmax=5)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),[],fontsize=6)
#plt.title('?(RACMO-TP2 - IBTrACS)',fontsize=7)
ax7.text(-82.5,31.3,'f)',fontweight='bold',fontsize=6)

ax8 = plt.subplot2grid((40,60),(31,30),colspan=17,rowspan=1)
cbar8=plt.colorbar(b,cax=ax8,orientation='horizontal',extend='max')
cbar8.ax.tick_params(labelsize=6)
plt.xlabel('Difference maximum windspeed [m/s]',fontsize=6)

#plt.suptitle('Maximum surge height Hurricane Dorian',fontsize=6)
plt.savefig("/gpfs/work3/0/einf4186/eucp_knmi/scripts/tracking/FigSx_wind_maria_dorian.png",dpi=600,bbox_inches='tight')
plt.close()

#PLOT MIN PATM
ax1 = plt.subplot2grid((40,60),(0,0),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax1.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax1.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
a=ax1.tripcolor(longitude_REF_select[np.asarray(min_patm_REF)>0],latitude_REF_select[np.asarray(min_patm_REF)>0],(np.asarray(min_patm_REF)/100)[np.asarray(min_patm_REF)>0],cmap='terrain',vmin=920,vmax=1020)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),['14N','17N','20N'],fontsize=6)
plt.title('IBTrACS',fontsize=6)
plt.ylabel('Hurricane Maria',fontsize=6)
ax1.text(-68.5,20.3,'a)',fontweight='bold',fontsize=6)

difference_PGW = np.asarray(min_patm_PGW)-np.asarray(min_patm_REF)

ax2 = plt.subplot2grid((40,60),(0,20),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax2.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax2.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
b=ax2.tripcolor(longitude_REF_select[np.asarray(min_patm_REF)>0],latitude_REF_select[np.asarray(min_patm_REF)>0],(np.asarray(difference_PGW)/100)[np.asarray(min_patm_REF)>0],cmap='hot',vmin=-10,vmax=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),[],fontsize=6)
plt.title('?PGW - IBTrACS',fontsize=6)
ax2.text(-68.5,20.3,'b)',fontweight='bold',fontsize=6)

difference_TP2 = np.asarray(min_patm_TP2) - np.asarray(min_patm_REF)

ax3 = plt.subplot2grid((40,60),(0,40),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax3.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax3.set_extent([-68, -60, 14, 20], ccrs.PlateCarree())
ax3.tripcolor(longitude_REF_select[np.asarray(min_patm_REF)>0],latitude_REF_select[np.asarray(min_patm_REF)>0],(np.asarray(difference_TP2)/100)[np.asarray(min_patm_REF)>0],cmap='hot',vmin=-10,vmax=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-68,-64,-60),['68W','64W','60W'],fontsize=6)
plt.yticks((14,17,20),[],fontsize=6)
plt.title('?TP2 - IBTrACS',fontsize=6)
ax3.text(-68.5,20.3,'c)',fontweight='bold',fontsize=6)

ax4 = plt.subplot2grid((40,60),(15,0),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax4.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax4.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
a=ax4.tripcolor(longitude_REF_select_dorian[np.asarray(min_patm_REF_dorian)>0],latitude_REF_select_dorian[np.asarray(min_patm_REF_dorian)>0],(np.asarray(min_patm_REF_dorian)/100)[np.asarray(min_patm_REF_dorian)>0],cmap='terrain',vmin=920,vmax=1020)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),['25N','28N','31N'],fontsize=6)
plt.ylabel('Hurricane Dorian',fontsize=6)
#plt.title('IBTrACS',fontsize=7)
ax4.text(-82.5,31.3,'d)',fontweight='bold',fontsize=6)

difference_PGW_dorian = np.asarray(min_patm_PGW_dorian) - np.asarray(min_patm_REF_dorian)

ax5 = plt.subplot2grid((40,60),(15,20),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax5.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax5.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
b=ax5.tripcolor(longitude_REF_select_dorian[np.asarray(min_patm_REF_dorian)>0],latitude_REF_select_dorian[np.asarray(min_patm_REF_dorian)>0],(np.asarray(difference_PGW_dorian)/100)[np.asarray(min_patm_REF_dorian)>0],cmap='hot',vmin=-10,vmax=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),[],fontsize=6)
#plt.title('?(RACMO-PGW - IBTrACS)',fontsize=7)
ax5.text(-82.5,31.3,'e)',fontweight='bold',fontsize=6)

ax6 = plt.subplot2grid((40,60),(31,0),colspan=17,rowspan=1)
cbar6=plt.colorbar(a,cax=ax6,orientation='horizontal',extend='min')
cbar6.ax.tick_params(labelsize=6)
plt.xlabel('Minimum MSLP [hPa]',fontsize=6)

difference_TP2_dorian = np.asarray(min_patm_TP2_dorian)-np.asarray(min_patm_REF_dorian)

ax7 = plt.subplot2grid((40,60),(15,40),colspan=17,rowspan=15,projection=ccrs.PlateCarree())
ax7.add_feature(cfeature.NaturalEarthFeature('physical','land','10m',facecolor='grey',edgecolor='black',linewidth=0.7))#,zorder=100,edgecolor='k')
ax7.set_extent([-82, -74, 25, 31], ccrs.PlateCarree())
ax7.tripcolor(longitude_REF_select_dorian[np.asarray(min_patm_REF_dorian)>0],latitude_REF_select_dorian[np.asarray(min_patm_REF_dorian)>0],(np.asarray(difference_TP2_dorian)/100)[np.asarray(min_patm_REF_dorian)>0],cmap='hot',vmin=-10,vmax=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xticks((-82,-78,-74),['82W','78W','74W'],fontsize=6)
plt.yticks((25,28,31),[],fontsize=6)
#plt.title('?(RACMO-TP2 - IBTrACS)',fontsize=7)
ax7.text(-82.5,31.3,'f)',fontweight='bold',fontsize=6)

ax8 = plt.subplot2grid((40,60),(31,30),colspan=17,rowspan=1)
cbar8=plt.colorbar(b,cax=ax8,orientation='horizontal',extend='min')
cbar8.ax.tick_params(labelsize=6)
plt.xlabel('Difference minimum MSLP [hPa]',fontsize=6)

#plt.suptitle('Maximum surge height Hurricane Dorian',fontsize=6)
plt.savefig("/gpfs/work3/0/einf4186/eucp_knmi/scripts/tracking/FigSx_patm_maria_dorian.png",dpi=600,bbox_inches='tight')
plt.close()
