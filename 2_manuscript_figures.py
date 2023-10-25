import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
import os
import glob
#from cdo import Cdo
#cdo=Cdo()
from scipy.ndimage import minimum_filter
from itertools import *
from operator import itemgetter
import pandas as pd              
from pathlib import Path
import cartopy.crs as ccrs
import cartopy
import matplotlib.cm as cmplt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes

#define folder paths
ref_folder='/gpfs/work3/0/einf4186/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-REF-fERA5/1Hourly_data/'
tp2_folder='/gpfs/work3/0/einf4186/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-TP2-fERA5/1Hourly_data/'
pgw_folder='/gpfs/work3/0/einf4186/eucp_knmi/data/RACMO/LCARIB12/wCY33-v563-JJASON-PGW-CMIP5-19models-fERA5/1Hourly_data/'

ref_folder_out='/gpfs/work3/0/einf4186/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-REF-fERA5/1Hourly_data_regridded/'
tp2_folder_out='/gpfs/work3/0/einf4186/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-TP2-fERA5/1Hourly_data_regridded/'
pgw_folder_out='/gpfs/work3/0/einf4186/eucp_knmi/data/RACMO/LCARIB12/wCY33-v563-JJASON-PGW-CMIP5-19models-fERA5/1Hourly_data_regridded/'


df_ref=pd.read_pickle('df_ref.pkl')
df_tp2=pd.read_pickle('df_tp2.pkl')
df_pgw=pd.read_pickle('df_pgw.pkl')
df_ibt=pd.read_pickle('df_ibtracs.pkl')

##divide by 1.11 because it seems that racmo data is most comparable to wind gusts. Here we transform to 10 meter 1-minute averaged wind speed m/s
#df_ref['w10m_max']=df_ref['w10m_max']/1.11
#df_pgw['w10m_max']=df_pgw['w10m_max']/1.11
#df_tp2['w10m_max']=df_tp2['w10m_max']/1.11

#only needed for minimum certain TC strength
TS=18; TC1=33.06; TC2=42.78; TC3=49.44; TC4=58.06; TC5=70
TC=TC1
ids=df_ref['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_ref.loc[df_ref['id']==storm_id]
    if df_storm.w10m_max.max()>TC:
        dfs.append(df_storm)

df_ref = pd.concat(dfs).sort_values(by=['id','time']).reset_index(drop=True)

ids=df_pgw['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_pgw.loc[df_pgw['id']==storm_id]
    if df_storm.w10m_max.max()>TC:
        dfs.append(df_storm)

df_pgw = pd.concat(dfs).sort_values(by=['id','time']).reset_index(drop=True)

ids=df_tp2['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_tp2.loc[df_tp2['id']==storm_id]
    if df_storm.w10m_max.max()>TC:
        dfs.append(df_storm)

df_tp2 = pd.concat(dfs).sort_values(by=['id','time']).reset_index(drop=True)

ids=df_ibt['SID'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibt.loc[df_ibt['SID']==storm_id]
    if df_storm.USA_WIND.max()>TC:
        dfs.append(df_storm)

df_ibt = pd.concat(dfs).sort_values(by=['SID','ISO_TIME']).reset_index(drop=True)

##############
#FIGURE 1#
##############
#1a: absolute frequency tracks and hours per decade
#1b: relative frequency tracks per category
#1c: cdf
#1d: relative frequency hours per category

#1a tracks per decade
year_freq_ref_tracks=[]
for year in range(1979,2021):
    year_freq_ref_tracks.append(len(df_ref.loc[df_ref['year']==year]['id'].unique()))

df_ibt_year_tracks=[]
for i in range(len(df_ibt)):
    df_ibt_year_tracks.append(df_ibt.loc[i]['ISO_TIME'].year)

df_ibt_tracks=df_ibt.copy(deep=True)
df_ibt_tracks['year']=df_ibt_year_tracks

year_freq_ibt_tracks=[]
for year in range(1979,2021):
    year_freq_ibt_tracks.append(len(df_ibt_tracks.loc[df_ibt_tracks['year']==year]['SID'].unique()))

import random
labels = ['Tracks/decade','Hours/decade']
r1_tracks=np.mean(year_freq_ref_tracks)*10
i1_tracks=np.mean(year_freq_ibt_tracks)*10
data=year_freq_ibt_tracks
def bootstrap(data, R=10000):
    means = []
    n = len(data)
    for i in range(R):
        sampled_data = random.choices(data,k=n)
        mean = np.mean(sampled_data)*10
        means.append(mean)
    
    return pd.DataFrame(means, columns=['means'])

bootstrap_means_tracks=bootstrap(data=year_freq_ibt_tracks)
i1_yer_tracks=np.std(bootstrap_means_tracks.means)

data=year_freq_ref_tracks
def bootstrap(data, R=10000):
    means = []
    n = len(data)
    for i in range(R):
        sampled_data = random.choices(data,k=n)
        mean = np.mean(sampled_data)*10
        means.append(mean)
    
    return pd.DataFrame(means, columns=['means'])

bootstrap_means_tracks=bootstrap(data=year_freq_ref_tracks)
r1_yer_tracks=np.std(bootstrap_means_tracks.means)

year_freq_ref_hours=[]
for year in range(1979,2021):
    year_freq_ref_hours.append(len(df_ref.loc[df_ref['year']==year]))

df_ibt_year_hours=[]
for i in range(len(df_ibt)):
    df_ibt_year_hours.append(df_ibt.loc[i]['ISO_TIME'].year)

df_ibt_hours=df_ibt.copy(deep=True)
df_ibt_hours['year']=df_ibt_year_hours

year_freq_ibt_hours=[]
for year in range(1979,2021):
    year_freq_ibt_hours.append(len(df_ibt_hours.loc[df_ibt_hours['year']==year]))

r1_hours=np.mean(year_freq_ref_hours)*10
i1_hours=np.mean(year_freq_ibt_hours)*10

data=year_freq_ibt_hours
def bootstrap(data, R=10000):
    means = []
    n = len(data)
    for i in range(R):
        sampled_data = random.choices(data,k=n)
        mean = np.mean(sampled_data)*10
        means.append(mean)
    
    return pd.DataFrame(means, columns=['means'])

bootstrap_means_hours=bootstrap(data=year_freq_ibt_hours)
i1_yer_hours=np.std(bootstrap_means_hours.means)

data=year_freq_ref_hours
def bootstrap(data, R=10000):
    means = []
    n = len(data)
    for i in range(R):
        sampled_data = random.choices(data,k=n)
        mean = np.mean(sampled_data)*10
        means.append(mean)
    
    return pd.DataFrame(means, columns=['means'])

bootstrap_means_hours=bootstrap(data=year_freq_ref_hours)
r1_yer_hours=np.std(bootstrap_means_hours.means)

IBTrACS = [i1_tracks,i1_hours/100]
RACMO_REF = [r1_tracks,r1_hours/100]
IBTrACS_YER = [i1_yer_tracks,i1_yer_hours/100]
RACMO_REF_YER = [r1_yer_tracks,r1_yer_hours/100]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots()
ax2=ax1.twinx()

rects1 = ax1.bar(x - width/2, IBTrACS, width, yerr=IBTrACS_YER, capsize=7, label='IBTrACS',color='tab:blue')
rects2 = ax1.bar(x + width/2, RACMO_REF, width, yerr=RACMO_REF_YER, capsize=7, label='RACMO-REF',color='tab:orange')

ax1.set_ylabel('Frequency [tracks/decade]',fontsize=13)
ax2.set_ylabel('Frequency [hours/decade]',fontsize=13)
ax1.set_title('Absolute frequency',fontsize=14)#: tracks per decade',fontsize=12)
ax1.set_xticks(x, labels,fontsize=13)
ax1.set_yticks([0,5,10,15,20,25,30,35],['0','5','10','15','20','25','30','35'],fontsize=13)
ax2.set_yticks([0,500,1000,1500,2000,2500,3000,3500],['0','500','1000','1500','2000','2500','3000','3500'],fontsize=13)
ax1.legend(fontsize=13)
ax1.text(-0.65,35.8,'a)',fontweight='bold',fontsize=13)

fig.tight_layout()
plt.savefig('Fig2a_absolute_frequency.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 1b: Absolute frequency plot hours
ibt_w10ms=[]
ibt_mslps=[]
for storm_id in df_ibt.SID.unique():
    storm_df=df_ibt.loc[df_ibt['SID'] == storm_id]
    ibt_w10ms.append(storm_df['USA_WIND'].max())
    ibt_mslps.append(storm_df['USA_PRES'].min())

ref_w10ms=[]
ref_mslps=[]
for storm_id in df_ref.id.unique():
    storm_df=df_ref.loc[df_ref['id'] == storm_id]
    ref_w10ms.append(storm_df['w10m_max'].max())
    ref_mslps.append(storm_df['mslp_min'].min())

bars_ibt = [np.nansum(np.asarray(ibt_w10ms)<=17)/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>17)&(np.asarray(ibt_w10ms)<=32))/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>32)&(np.asarray(ibt_w10ms)<=43))/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>43)&(np.asarray(ibt_w10ms)<=50))/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>50)&(np.asarray(ibt_w10ms)<=58))/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>58)&(np.asarray(ibt_w10ms)<=70))/len(ibt_w10ms),np.nansum(np.asarray(ibt_w10ms)>70)/len(ibt_w10ms)]
bars_ref = [np.nansum(np.asarray(ref_w10ms)<=17)/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>17)&(np.asarray(ref_w10ms)<=32))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>32)&(np.asarray(ref_w10ms)<=43))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>43)&(np.asarray(ref_w10ms)<=50))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>50)&(np.asarray(ref_w10ms)<=58))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>58)&(np.asarray(ref_w10ms)<=70))/len(ref_w10ms),np.nansum(np.asarray(ref_w10ms)>70)/len(ref_w10ms)]

barWidth=0.2
r1 = np.arange(len(bars_ref))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars_ibt, width = barWidth, edgecolor = 'black', capsize=7, label='IBTrACS', color='tab:blue')#color = 'blue', 
plt.bar(r2, bars_ref, width = barWidth, edgecolor = 'black', capsize=7, label='RACMO-REF', color='tab:orange')#color = 'cyan', 

plt.xticks([r + barWidth for r in range(len(bars_ibt))],['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'],fontsize=13)
plt.ylabel('Relative frequency',fontsize=13)
plt.xlabel('Track maximum intensity [TC category]',fontsize=13)
plt.ylim([0,0.80])
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.title('Relative frequency: tracks per category',fontsize=13)
plt.text(-1.3,0.815,'b)',fontweight='bold',fontsize=13)

plt.legend(fontsize=13)
plt.savefig('Fig2b_relative_frequency_tracks_per_category.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 1c: Relative frequency TC1+ categories; tracks per decade
a=df_ibt.USA_WIND.values
b=df_ref.w10m_max.values
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='IBTrACS (n=%s)'%(f"{len(a):,}"),zorder=9)
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='RACMO-REF (n=%s)'%(f"{len(b):,}"),zorder=10)
plt.title('CDF maximum windspeed',fontsize=13)
plt.ylabel('Probability',fontsize=13)
plt.xlabel('Windspeed [m/s]',fontsize=13)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.axvline(x = TS, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC1, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC2, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC3, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC4, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC5, color= 'black', linestyle='dashed', zorder=1)
plt.legend(fontsize=13)
plt.text(2.5,1.08,'c)',fontweight='bold',fontsize=13)
plt.text((TS+TC1)/2-2.5,-0.04,' TS ',fontsize=10)
plt.text((TC1+TC2)/2-2.5,-0.04,'CAT1',fontsize=10)
plt.text((TC2+TC3)/2-2.5,-0.04,'CAT2',fontsize=10)
plt.text((TC3+TC4)/2-2.5,-0.04,'CAT3',fontsize=10)
plt.text((TC4+TC5)/2-2.5,-0.04,'CAT4',fontsize=10)
plt.text((TC5+TC5+9)/2-2.5,-0.04,'CAT5',fontsize=10)
plt.savefig('Fig2c_cdf_windspeed_timesteps.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 1d: Relative frequency TC1+ categories; hours per decade
bars_ibt = [np.nansum(df_ibt.USA_WIND<=17)/len(df_ibt),np.nansum((df_ibt.USA_WIND>17)&(df_ibt.USA_WIND<=32))/len(df_ibt),np.nansum((df_ibt.USA_WIND>32)&(df_ibt.USA_WIND<=43))/len(df_ibt),np.nansum((df_ibt.USA_WIND>43)&(df_ibt.USA_WIND<=50))/len(df_ibt),np.nansum((df_ibt.USA_WIND>50)&(df_ibt.USA_WIND<=58))/len(df_ibt),np.nansum((df_ibt.USA_WIND>58)&(df_ibt.USA_WIND<=70))/len(df_ibt),np.nansum(df_ibt.USA_WIND>70)/len(df_ibt)]
bars_ref = [np.nansum(df_ref.w10m_max<=17)/len(df_ref),np.nansum((df_ref.w10m_max>17)&(df_ref.w10m_max<=32))/len(df_ref),np.nansum((df_ref.w10m_max>32)&(df_ref.w10m_max<=43))/len(df_ref),np.nansum((df_ref.w10m_max>43)&(df_ref.w10m_max<=50))/len(df_ref),np.nansum((df_ref.w10m_max>50)&(df_ref.w10m_max<=58))/len(df_ref),np.nansum((df_ref.w10m_max>58)&(df_ref.w10m_max<=70))/len(df_ref),np.nansum(df_ref.w10m_max>70)/len(df_ref)]

barWidth=0.2
r1 = np.arange(len(bars_ibt))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars_ibt, width = barWidth, edgecolor = 'black', capsize=7, label='IBTrACS',color = 'tab:blue')
plt.bar(r2, bars_ref, width = barWidth, edgecolor = 'black', capsize=7, label='RACMO-REF',color = 'tab:orange') 

plt.xticks([r + barWidth for r in range(len(bars_ibt))],['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'],fontsize=13)
plt.ylabel('Relative frequency',fontsize=13)
plt.xlabel('Hourly maximum intensity [TC category]',fontsize=13)
plt.ylim([0,0.4])
plt.yticks([0,0.1,0.2,0.3,0.4],['0.0','0.1','0.2','0.3','0.4'],fontsize=13)
plt.xticks(fontsize=13)
plt.title('Relative frequency: hours per category',fontsize=13)
plt.text(-1.3,0.41,'d)',fontweight='bold',fontsize=13)
plt.legend(fontsize=13)
plt.savefig('Fig2d_relative_frequency_hours_per_category.png',dpi=600,bbox_inches='tight')
plt.close()

#extra info about rapid intensification
ref_rapid_ids=[]
for storm_id in df_ref.id.unique():
    storm_df=df_ref.loc[df_ref['id'] == storm_id]
    storm_wind=storm_df['w10m_max'].values
    windspeed_difs=[]
    for i in range(len(storm_wind)-24):
        windspeed_difs.append(storm_wind[i+24]-storm_wind[i])
    if max(windspeed_difs)>15.433:
        ref_rapid_ids.append(storm_id)

df_ref = df_ref.loc[df_ref['id'].isin(ref_rapid_ids)].reset_index(drop=True)

ibt_rapid_ids=[]
for storm_id in df_ibt.SID.unique():
    storm_df=df_ibt.loc[df_ibt['SID'] == storm_id]
    storm_wind=storm_df['USA_WIND'].values
    windspeed_difs=[]
    for i in range(len(storm_wind)-24):
        windspeed_difs.append(storm_wind[i+24]-storm_wind[i])
    if max(windspeed_difs)>15.433:
        ibt_rapid_ids.append(storm_id)

df_ibt = df_ibt.loc[df_ibt['SID'].isin(ibt_rapid_ids)].reset_index(drop=True)

#Figure Appendix S1: IBTrACS TC track-point density 1979-2020
ibt_lons = df_ibt.USA_LON.values
ibt_lats = df_ibt.USA_LAT.values

crg = ccrs.PlateCarree()
        
plt.close('all')
fig = plt.figure()
ax = plt.axes(projection=crg,zorder=6)
ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='gainsboro',zorder=4,edgecolor='black',linewidth=0.001)#,alpha=0.5)
ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), zorder=5,linewidth=0.2)
ax.set_global()
ax.set_extent([-85, -4, 8, 28], crg)
cmap = cmplt.get_cmap('jet')

x=ibt_lons
y=ibt_lats
bs1,xedges,yedges,image=plt.hist2d(x, y,bins=[45,20], range=[[-85,-40],[8,28]],vmin=0,vmax=50,transform=crg,cmap=cmap)#,alpha=0.5)#
plt.title('IBTrACS TC track-point density 1979-2020',fontsize=6)
plt.ylabel('latitude [degrees]',fontsize=6)
plt.xlabel('longitude [degrees]',fontsize=6)
plt.xticks(np.arange(-85,-39,step=5),fontsize=6)
plt.yticks(np.arange(8,29,step=5),fontsize=6)
#plt.text(-88.5,29,'a)',weight='bold',fontsize=8) #a,b 0.42   c 0.49

gl = ax.gridlines(crs=crg, draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',zorder=6)
gl.top_lables = False
gl.left_labels = False
gl.xlines = True
gl.ylocator = mticker.FixedLocator(np.arange(8.,28.,5))
gl.xlocator = mticker.FixedLocator(np.arange(-85.,-40.,5))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

cbaxes=fig.add_axes([0.935,0.265,0.020,0.45])
cbar= plt.colorbar(ax=ax,cax=cbaxes,cmap=cmap,spacing='uniform', orientation='vertical',shrink=0.35,extend='max')#,ticks=[5,10,15,20,30,40,50,100,250,500,1000])
cbar.set_label('TC track-point density',fontsize=6)
cbar.ax.tick_params(labelsize=6)

plt.savefig('FigS1_track_point_density_IBTrACS.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix S2: RACMO-REF TC track-point density 1979-2020
ref_lons = df_ref.lon.values
ref_lats = df_ref.lat.values

crg = ccrs.PlateCarree()
        
plt.close('all')
fig = plt.figure()
ax = plt.axes(projection=crg,zorder=6)
ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='gainsboro',zorder=4,edgecolor='black',linewidth=0.001)#,alpha=0.5)
ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), zorder=5,linewidth=0.2)
ax.set_global()
ax.set_extent([-85, -4, 8, 28], crg)
cmap = cmplt.get_cmap('jet')

x=ref_lons-360
y=ref_lats
bs2,xedges1,yedges1,image1=plt.hist2d(x, y,bins=[45,20], range=[[-85,-40],[8,28]],vmin=0,vmax=50,transform=crg,cmap=cmap)#,alpha=0.5)#
plt.title('RACMO-REF TC track-point density 1979-2020',fontsize=6)
plt.ylabel('latitude [degrees]',fontsize=6)
plt.xlabel('longitude [degrees]',fontsize=6)
plt.xticks(np.arange(-85,-39,step=5),fontsize=6)
plt.yticks(np.arange(8,29,step=5),fontsize=6)
#plt.text(-88.5,29,'a)',weight='bold',fontsize=8) #a,b 0.42   c 0.49

gl = ax.gridlines(crs=crg, draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',zorder=6)
gl.top_lables = False
gl.left_labels = False
gl.xlines = True
gl.ylocator = mticker.FixedLocator(np.arange(8.,28.,5))
gl.xlocator = mticker.FixedLocator(np.arange(-85.,-40.,5))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

cbaxes=fig.add_axes([0.935,0.265,0.020,0.45])
cbar= plt.colorbar(ax=ax,cax=cbaxes,cmap=cmap,spacing='uniform', orientation='vertical',shrink=0.35,extend='max')#,ticks=[5,10,15,20,30,40,50,100,250,500,1000])
cbar.set_label('TC track-point density',fontsize=6)
cbar.ax.tick_params(labelsize=6)

plt.savefig('FigS2_track_point_density_RACMO_REF.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix S5: RACMO-PGW TC track-point density 1979-2020
pgw_lons = df_pgw.lon.values
pgw_lats = df_pgw.lat.values

crg = ccrs.PlateCarree()
        
plt.close('all')
fig = plt.figure()
ax = plt.axes(projection=crg,zorder=6)
ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='gainsboro',zorder=4,edgecolor='black',linewidth=0.001)#,alpha=0.5)
ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), zorder=5,linewidth=0.2)
ax.set_global()
ax.set_extent([-85, -4, 8, 28], crg)
cmap = cmplt.get_cmap('jet')

x=pgw_lons-360
y=pgw_lats
bs2,xedges1,yedges1,image1=plt.hist2d(x, y,bins=[45,20], range=[[-85,-40],[8,28]],vmin=0,vmax=50,transform=crg,cmap=cmap)#,alpha=0.5)#
plt.title('RACMO-PGW TC track-point density 1979-2020',fontsize=6)
plt.ylabel('latitude [degrees]',fontsize=6)
plt.xlabel('longitude [degrees]',fontsize=6)
plt.xticks(np.arange(-85,-39,step=5),fontsize=6)
plt.yticks(np.arange(8,29,step=5),fontsize=6)
#plt.text(-88.5,29,'a)',weight='bold',fontsize=8) #a,b 0.42   c 0.49

gl = ax.gridlines(crs=crg, draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',zorder=6)
gl.top_lables = False
gl.left_labels = False
gl.xlines = True
gl.ylocator = mticker.FixedLocator(np.arange(8.,28.,5))
gl.xlocator = mticker.FixedLocator(np.arange(-85.,-40.,5))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

cbaxes=fig.add_axes([0.935,0.265,0.020,0.45])
cbar= plt.colorbar(ax=ax,cax=cbaxes,cmap=cmap,spacing='uniform', orientation='vertical',shrink=0.35,extend='max')#,ticks=[5,10,15,20,30,40,50,100,250,500,1000])
cbar.set_label('TC track-point density',fontsize=6)
cbar.ax.tick_params(labelsize=6)

plt.savefig('FigS5_track_point_density_RACMO_PGW.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix S6: RACMO-TP2 TC track-point density 1979-2020
tp2_lons = df_tp2.lon.values
tp2_lats = df_tp2.lat.values

crg = ccrs.PlateCarree()
        
plt.close('all')
fig = plt.figure()
ax = plt.axes(projection=crg,zorder=6)
ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='gainsboro',zorder=4,edgecolor='black',linewidth=0.001)#,alpha=0.5)
ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), zorder=5,linewidth=0.2)
ax.set_global()
ax.set_extent([-85, -4, 8, 28], crg)
cmap = cmplt.get_cmap('jet')

x=tp2_lons-360
y=tp2_lats
bs2,xedges1,yedges1,image1=plt.hist2d(x, y,bins=[45,20], range=[[-85,-40],[8,28]],vmin=0,vmax=100,transform=crg,cmap=cmap)#,alpha=0.5)#
plt.title('RACMO-TP2 TC track-point density 1979-2020',fontsize=6)
plt.ylabel('latitude [degrees]',fontsize=6)
plt.xlabel('longitude [degrees]',fontsize=6)
plt.xticks(np.arange(-85,-39,step=5),fontsize=6)
plt.yticks(np.arange(8,29,step=5),fontsize=6)
#plt.text(-88.5,29,'a)',weight='bold',fontsize=8) #a,b 0.42   c 0.49

gl = ax.gridlines(crs=crg, draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',zorder=6)
gl.top_lables = False
gl.left_labels = False
gl.xlines = True
gl.ylocator = mticker.FixedLocator(np.arange(8.,28.,5))
gl.xlocator = mticker.FixedLocator(np.arange(-85.,-40.,5))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

cbaxes=fig.add_axes([0.935,0.265,0.020,0.45])
cbar= plt.colorbar(ax=ax,cax=cbaxes,cmap=cmap,spacing='uniform', orientation='vertical',shrink=0.35,extend='max')#,ticks=[5,10,15,20,30,40,50,100,250,500,1000])
cbar.set_label('TC track-point density',fontsize=6)
cbar.ax.tick_params(labelsize=6)

plt.savefig('FigS6_track_point_density_RACMO_TP2.png',dpi=600,bbox_inches='tight')
plt.close()

#plt.pcolormesh(xedges,yedges,(bs2-bs1).T)
#plt.colorbar()
#plt.show()

#Figure 2:  Absolute frequency RACMO-REF vs RACMO-PGW & RACMO-TP2 (categorey 1+ TCs & 3+ TCs)
#Figure 2a: Absolute frequency plot tracks
year_freq_ref_tracks=[]
for year in range(1979,2021):
    year_freq_ref_tracks.append(len(df_ref.loc[df_ref['year']==year]['id'].unique()))

year_freq_pgw_tracks=[]
for year in range(1979,2021):
    year_freq_pgw_tracks.append(len(df_pgw.loc[df_pgw['year']==year]['id'].unique()))

year_freq_tp2_tracks=[]
for year in range(1979,2021):
    year_freq_tp2_tracks.append(len(df_tp2.loc[df_tp2['year']==year]['id'].unique()))

import random
labels = ['Tracks/decade','Hours/decade']
r1_tracks=np.mean(year_freq_ref_tracks)*10
p1_tracks=np.mean(year_freq_pgw_tracks)*10
t1_tracks=np.mean(year_freq_tp2_tracks)*10

def bootstrap(data, R=10000):
    means = []
    n = len(data)
    for i in range(R):
        sampled_data = random.choices(data,k=n)
        mean = np.mean(sampled_data)*10
        means.append(mean)
    
    return pd.DataFrame(means, columns=['means'])

data=year_freq_ref_tracks
bootstrap_means_tracks=bootstrap(data=year_freq_ref_tracks)
r1_yer_tracks=np.std(bootstrap_means_tracks.means)

data=year_freq_pgw_tracks
bootstrap_means_tracks=bootstrap(data=year_freq_pgw_tracks)
p1_yer_tracks=np.std(bootstrap_means_tracks.means)

data=year_freq_tp2_tracks
bootstrap_means_tracks=bootstrap(data=year_freq_tp2_tracks)
t1_yer_tracks=np.std(bootstrap_means_tracks.means)

year_freq_ref_hours=[]
for year in range(1979,2021):
    year_freq_ref_hours.append(len(df_ref.loc[df_ref['year']==year]))

year_freq_pgw_hours=[]
for year in range(1979,2021):
    year_freq_pgw_hours.append(len(df_pgw.loc[df_pgw['year']==year]))

year_freq_tp2_hours=[]
for year in range(1979,2021):
    year_freq_tp2_hours.append(len(df_tp2.loc[df_tp2['year']==year]))

r1_hours=np.mean(year_freq_ref)*10
p1_hours=np.mean(year_freq_pgw)*10
t1_hours=np.mean(year_freq_tp2)*10

def bootstrap(data, R=10000):
    means = []
    n = len(data)
    for i in range(R):
        sampled_data = random.choices(data,k=n)
        mean = np.mean(sampled_data)*10
        means.append(mean)
    
    return pd.DataFrame(means, columns=['means'])

data=year_freq_ref_hours
bootstrap_means_hours=bootstrap(data=year_freq_ref_hours)
r1_yer_hours=np.std(bootstrap_means_hours.means)

data=year_freq_pgw_hours
bootstrap_means_hours=bootstrap(data=year_freq_pgw_hours)
p1_yer_hours=np.std(bootstrap_means_hours.means)

data=year_freq_tp2_hours
bootstrap_means_hours=bootstrap(data=year_freq_tp2_hours)
t1_yer_hours=np.std(bootstrap_means_hours.means)

RACMO_REF = [r1_tracks,r1_hours/100]
RACMO_PGW = [p1_tracks,p1_hours/100]
RACMO_TP2 = [t1_tracks,t1_hours/100]
RACMO_REF_YER = [r1_yer_tracks,r1_yer_hours/100]
RACMO_PGW_YER = [p1_yer_tracks,p1_yer_hours/100]
RACMO_TP2_YER = [t1_yer_tracks,t1_yer_hours/100]

x = np.arange(len(labels))
width = 0.25

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
rects1 = ax1.bar(x - width, RACMO_REF, width, yerr=RACMO_REF_YER, capsize=7, label='RACMO-REF',color='tab:orange')
rects2 = ax1.bar(x, RACMO_PGW, width, yerr=RACMO_PGW_YER, capsize=7, label='RACMO-PGW',color='tab:green')
rects3 = ax1.bar(x + width, RACMO_TP2, width, yerr=RACMO_TP2_YER, capsize=7, label='RACMO-TP2',color='tab:grey')

ax1.set_ylabel('Frequency [tracks/decade]',fontsize=13)
ax2.set_ylabel('Frequency [hours/decade]',fontsize=13)
ax1.set_title('Absolute frequency',fontsize=13)
ax1.set_xticks(x, labels,fontsize=13)
ax1.set_yticks([0,10,20,30,40,50,60,70],['0','10','20','30','40','50','60','70'],fontsize=13)
ax2.set_yticks([0,1000,2000,3000,4000,5000,6000,7000],['0','1000','2000','3000','4000','5000','6000','7000'],fontsize=13)
ax1.legend(fontsize=13)
ax1.text(-0.69,71.5,'a)',fontweight='bold',fontsize=13)

fig.tight_layout()
plt.savefig('Fig3a_absolute_frequency.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 2b: Relative frequency tracks per category
ref_w10ms=[]
ref_mslps=[]
for storm_id in df_ref.id.unique():
    storm_df=df_ref.loc[df_ref['id'] == storm_id]
    ref_w10ms.append(storm_df['w10m_max'].max())
    ref_mslps.append(storm_df['mslp_min'].min())

pgw_w10ms=[]
pgw_mslps=[]
for storm_id in df_pgw.id.unique():
    storm_df=df_pgw.loc[df_pgw['id'] == storm_id]
    pgw_w10ms.append(storm_df['w10m_max'].max())
    pgw_mslps.append(storm_df['mslp_min'].min())

tp2_w10ms=[]
tp2_mslps=[]
for storm_id in df_tp2.id.unique():
    storm_df=df_tp2.loc[df_tp2['id'] == storm_id]
    tp2_w10ms.append(storm_df['w10m_max'].max())
    tp2_mslps.append(storm_df['mslp_min'].min())

bars_ref = [np.nansum(np.asarray(ref_w10ms)<=17)/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>17)&(np.asarray(ref_w10ms)<=32))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>32)&(np.asarray(ref_w10ms)<=43))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>43)&(np.asarray(ref_w10ms)<=50))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>50)&(np.asarray(ref_w10ms)<=58))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>58)&(np.asarray(ref_w10ms)<=70))/len(ref_w10ms),np.nansum(np.asarray(ref_w10ms)>70)/len(ref_w10ms)]
bars_pgw = [np.nansum(np.asarray(pgw_w10ms)<=17)/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>17)&(np.asarray(pgw_w10ms)<=32))/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>32)&(np.asarray(pgw_w10ms)<=43))/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>43)&(np.asarray(pgw_w10ms)<=50))/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>50)&(np.asarray(pgw_w10ms)<=58))/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>58)&(np.asarray(pgw_w10ms)<=70))/len(pgw_w10ms),np.nansum(np.asarray(pgw_w10ms)>70)/len(pgw_w10ms)]
bars_tp2 = [np.nansum(np.asarray(tp2_w10ms)<=17)/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>17)&(np.asarray(tp2_w10ms)<=32))/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>32)&(np.asarray(tp2_w10ms)<=43))/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>43)&(np.asarray(tp2_w10ms)<=50))/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>50)&(np.asarray(tp2_w10ms)<=58))/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>58)&(np.asarray(tp2_w10ms)<=70))/len(tp2_w10ms),np.nansum(np.asarray(tp2_w10ms)>70)/len(tp2_w10ms)]

barWidth=0.25
r1 = np.arange(len(bars_ref))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, bars_ref, width = barWidth, edgecolor = 'black', capsize=7, label='RACMO-REF',color='tab:orange')#color = 'blue', 
plt.bar(r2, bars_pgw, width = barWidth, edgecolor = 'black', capsize=7, label='RACMO-PGW',color='tab:green')#color = 'cyan', 
plt.bar(r3, bars_tp2, width = barWidth, edgecolor = 'black', capsize=7, label='RACMO-TP2',color='tab:grey')#color = 'cyan', 

plt.xticks([r + barWidth for r in range(len(bars_ref))],['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'],fontsize=13)
plt.ylabel('Relative frequency',fontsize=13)
plt.xlabel('Track maximum intensity [TC category]',fontsize=13)
plt.ylim([0,0.8])
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.title('Relative frequency: tracks per category',fontsize=13)
plt.text(-1.35,0.82,'b)',fontweight='bold',fontsize=13)
plt.legend(fontsize=13)
plt.savefig('Fig3b_relative_frequency_tracks_per_category.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 2c: cdf
a=df_ref.w10m_max.values
b=df_pgw.w10m_max.values
c=df_tp2.w10m_max.values
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='RACMO-REF (n=%s)'%(f"{len(a):,}"),color='tab:orange',zorder=8)
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='RACMO-PGW (n=%s)'%(f"{len(b):,}"),color='tab:green',zorder=9)
plt.plot(np.sort(c), np.linspace(0, 1, len(c), endpoint=False),label='RACMO-TP2 (n=%s)'%(f"{len(c):,}"),color='tab:grey',zorder=10)
plt.title('CDF maximum windspeed',fontsize=13)
plt.ylabel('Probability',fontsize=13)
plt.xlabel('Windspeed [m/s]',fontsize=13)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.axvline(x = TS, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC1, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC2, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC3, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC4, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC5, color= 'black', linestyle='dashed', zorder=1)
plt.legend(fontsize=13)
plt.text(2.5,1.08,'c)',fontweight='bold',fontsize=13)
plt.text((TS+TC1)/2-2.5,-0.04,' TS ',fontsize=10)
plt.text((TC1+TC2)/2-2.5,-0.04,'CAT1',fontsize=10)
plt.text((TC2+TC3)/2-2.5,-0.04,'CAT2',fontsize=10)
plt.text((TC3+TC4)/2-2.5,-0.04,'CAT3',fontsize=10)
plt.text((TC4+TC5)/2-2.5,-0.04,'CAT4',fontsize=10)
plt.text((TC5+TC5+9)/2-2.5,-0.04,'CAT5',fontsize=10)
plt.savefig('Fig3c_cdf_windspeed_timesteps.png',dpi=600,bbox_inches='tight')
plt.close()

plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='IBTrACS (n=%s)'%(f"{len(a):,}"),zorder=9)
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='RACMO-REF (n=%s)'%(f"{len(b):,}"),zorder=10)
plt.title('CDF maximum windspeed',fontsize=13)
plt.ylabel('Probability',fontsize=13)
plt.xlabel('Windspeed [m/s]',fontsize=13)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.axvline(x = TS, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC1, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC2, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC3, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC4, color= 'black', linestyle='dashed', zorder=1)
plt.axvline(x = TC5, color= 'black', linestyle='dashed', zorder=1)
plt.legend(fontsize=13)
plt.text(2.5,1.08,'c)',fontweight='bold',fontsize=13)
plt.text((TS+TC1)/2-2.5,-0.04,' TS ',fontsize=10)
plt.text((TC1+TC2)/2-2.5,-0.04,'CAT1',fontsize=10)
plt.text((TC2+TC3)/2-2.5,-0.04,'CAT2',fontsize=10)
plt.text((TC3+TC4)/2-2.5,-0.04,'CAT3',fontsize=10)
plt.text((TC4+TC5)/2-2.5,-0.04,'CAT4',fontsize=10)
plt.text((TC5+TC5+9)/2-2.5,-0.04,'CAT5',fontsize=10)
plt.savefig('Fig2c_cdf_windspeed_timesteps.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 2d: Relative frequency TC1+ categories; hours per decade
bars_ref = [np.nansum(df_ref.w10m_max<=17)/len(df_ref),np.nansum((df_ref.w10m_max>17)&(df_ref.w10m_max<=32))/len(df_ref),np.nansum((df_ref.w10m_max>32)&(df_ref.w10m_max<=43))/len(df_ref),np.nansum((df_ref.w10m_max>43)&(df_ref.w10m_max<=50))/len(df_ref),np.nansum((df_ref.w10m_max>50)&(df_ref.w10m_max<=58))/len(df_ref),np.nansum((df_ref.w10m_max>58)&(df_ref.w10m_max<=70))/len(df_ref),np.nansum(df_ref.w10m_max>70)/len(df_ref)]
bars_pgw = [np.nansum(df_pgw.w10m_max<=17)/len(df_pgw),np.nansum((df_pgw.w10m_max>17)&(df_pgw.w10m_max<=32))/len(df_pgw),np.nansum((df_pgw.w10m_max>32)&(df_pgw.w10m_max<=43))/len(df_pgw),np.nansum((df_pgw.w10m_max>43)&(df_pgw.w10m_max<=50))/len(df_pgw),np.nansum((df_pgw.w10m_max>50)&(df_pgw.w10m_max<=58))/len(df_pgw),np.nansum((df_pgw.w10m_max>58)&(df_pgw.w10m_max<=70))/len(df_pgw),np.nansum(df_pgw.w10m_max>70)/len(df_pgw)]
bars_tp2 = [np.nansum(df_tp2.w10m_max<=17)/len(df_tp2),np.nansum((df_tp2.w10m_max>17)&(df_tp2.w10m_max<=32))/len(df_tp2),np.nansum((df_tp2.w10m_max>32)&(df_tp2.w10m_max<=43))/len(df_tp2),np.nansum((df_tp2.w10m_max>43)&(df_tp2.w10m_max<=50))/len(df_tp2),np.nansum((df_tp2.w10m_max>50)&(df_tp2.w10m_max<=58))/len(df_tp2),np.nansum((df_tp2.w10m_max>58)&(df_tp2.w10m_max<=70))/len(df_tp2),np.nansum(df_tp2.w10m_max>70)/len(df_tp2)]

barWidth=0.25
r1 = np.arange(len(bars_ref))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, bars_ref, width = barWidth, edgecolor = 'black', capsize=7, label='RACMO-REF',color='tab:orange')#color = 'blue', 
plt.bar(r2, bars_pgw, width = barWidth, edgecolor = 'black', capsize=7, label='RACMO-PGW',color='tab:green')#color = 'cyan', 
plt.bar(r3, bars_tp2, width = barWidth, edgecolor = 'black', capsize=7, label='RACMO-TP2',color='tab:grey')#color = 'cyan', 

plt.xticks([r + barWidth for r in range(len(bars_ref))],['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'],fontsize=13)
plt.ylabel('Relative frequency',fontsize=13)
plt.xlabel('Hourly maximum intensity [TC category]',fontsize=13)
plt.ylim([0,0.4])
plt.yticks([0,0.1,0.2,0.3,0.4],['0.0','0.1','0.2','0.3','0.4'],fontsize=13)
plt.xticks(fontsize=13)
plt.title('Relative frequency: hours per category',fontsize=13)
plt.text(-1.35,0.41,'d)',fontweight='bold',fontsize=13)
plt.legend(fontsize=13)
plt.savefig('Fig3d_relative_frequency_hours_per_category.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix S3: CDF max wind speed TC1+ of TCs that are both in IBTrACS AND RACMO-REF

radius= 6371
ids=df_ibt['SID'].unique()
dfs_ref_ids=[]
dfs_ibt_ids=[]
for i in range(len(df_ibt)):
    print(i)
    df_ibt_step = df_ibt.loc[i]
    if not df_ref.loc[df_ref['time']==df_ibt_step['ISO_TIME']].empty:
        search_ref=df_ref.loc[df_ref['time']==df_ibt_step['ISO_TIME']]
        lon_ibt=df_ibt_step.USA_LON
        lat_ibt=df_ibt_step.USA_LAT
        lons_sim=search_ref.lon.values
        lats_sim=search_ref.lat.values
        id_sim=search_ref.id.values
        distances=[]
        ids_sim=[]
        for xy in range(len(lons_sim)):
            lon_sim=lons_sim[xy]-360
            lat_sim=lats_sim[xy]
            latR1,latR2,lonR1,lonR2 = radians(lat_ibt),radians(lat_sim),radians(lon_ibt),radians(lon_sim)
            dlon = lonR2 - lonR1
            dlat = latR2 - latR1
            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
            if a>1:
                a=1
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distances.append(radius * c)
            ids_sim.append(id_sim[xy])
        if min(distances)<300:
            dfs_ref_ids.append(ids_sim[np.argmin(distances)])
            dfs_ibt_ids.append(df_ibt_step.SID)

dfs_ref_ids_unique=list(set(dfs_ref_ids))
dfs_ibt_ids_unique=list(set(dfs_ibt_ids))
#nr. of storms ref / timesteps: 76/10,749
#nr. of storms ibt / timesteps: 92/12,229
df_ref = df_ref.loc[df_ref['id'].isin(dfs_ref_ids_unique)].reset_index(drop=True)
df_ibt = df_ibt.loc[df_ibt['SID'].isin(dfs_ibt_ids_unique)].reset_index(drop=True)
#nr. of storms ref / timesteps: 42/5,884
#nr. of storms ibt / timesteps: 42/6,273

a=df_ibt.USA_WIND.values
b=df_ref.w10m_max.values
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='IBTrACS (n=%s)'%(f"{len(a):,}"))
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='RACMO-REF (n=%s)'%(f"{len(b):,}"))
plt.title('CDF maximum windspeed TC1+ of matching TCs')
plt.ylabel('Probability')
plt.xlabel('Windspeed [m/s]')
plt.legend()
#plt.text(4.5,1.08,'c)',fontweight='bold')
plt.savefig('FigS3_cdf_windspeed_timesteps_TC1_matching_TCs.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix S4: CDF max wind speed TC3+ of TCs that are both in IBTrACS AND RACMO-REF
radius= 6371
ids=df_ibt['SID'].unique()
dfs_ref_ids=[]
dfs_ibt_ids=[]
for i in range(len(df_ibt)):
    print(i)
    df_ibt_step = df_ibt.loc[i]
    if not df_ref.loc[df_ref['time']==df_ibt_step['ISO_TIME']].empty:
        search_ref=df_ref.loc[df_ref['time']==df_ibt_step['ISO_TIME']]
        lon_ibt=df_ibt_step.USA_LON
        lat_ibt=df_ibt_step.USA_LAT
        lons_sim=search_ref.lon.values
        lats_sim=search_ref.lat.values
        id_sim=search_ref.id.values
        distances=[]
        ids_sim=[]
        for xy in range(len(lons_sim)):
            lon_sim=lons_sim[xy]-360
            lat_sim=lats_sim[xy]
            latR1,latR2,lonR1,lonR2 = radians(lat_ibt),radians(lat_sim),radians(lon_ibt),radians(lon_sim)
            dlon = lonR2 - lonR1
            dlat = latR2 - latR1
            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
            if a>1:
                a=1
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distances.append(radius * c)
            ids_sim.append(id_sim[xy])
        if min(distances)<300:
            dfs_ref_ids.append(ids_sim[np.argmin(distances)])
            dfs_ibt_ids.append(df_ibt_step.SID)

dfs_ref_ids_unique=list(set(dfs_ref_ids))
dfs_ibt_ids_unique=list(set(dfs_ibt_ids))
#nr. of storms ref / timesteps: 74/10,601
#nr. of storms ibt / timesteps: 64/8,419
df_ref = df_ref.loc[df_ref['id'].isin(dfs_ref_ids_unique)].reset_index(drop=True)
df_ibt = df_ibt.loc[df_ibt['SID'].isin(dfs_ibt_ids_unique)].reset_index(drop=True)
#nr. of storms ref / timesteps: 30/4,372
#nr. of storms ibt / timesteps: 30/4,350

a=df_ibt.USA_WIND.values
b=df_ref.w10m_max.values
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='IBTrACS (n=%s)'%(f"{len(a):,}"))
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='RACMO-REF (n=%s)'%(f"{len(b):,}"))
plt.title('CDF maximum windspeed TC3+ of matching TCs')
plt.ylabel('Probability')
plt.xlabel('Windspeed [m/s]')
plt.legend()
#plt.text(4.5,1.08,'d)',fontweight='bold')
plt.savefig('FigS4_cdf_windspeed_timesteps_TC3_matching_TCs.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix S7: CDF max wind speed TC1+ of TCs that are in RACMO-REF, RACMO-PGW, and RACMO-TP2
radius= 6371
ids=df_ref['id'].unique()
dfs_pgw_ids=[]
dfs_ref_ids=[]
dfs_tp2_ids=[]
for i in range(len(df_ref)):
    print(i)
    df_ref_step = df_ref.loc[i]
    if not df_pgw.loc[df_pgw['time']==df_ref_step['time']].empty:
        search_pgw=df_pgw.loc[df_pgw['time']==df_ref_step['time']]
        lon_ref=df_ref_step.lon
        lat_ref=df_ref_step.lat
        lons_pgw=search_pgw.lon.values
        lats_pgw=search_pgw.lat.values
        id_pgw=search_pgw.id.values
        distances1=[]
        ids_pgw=[]
        for xy in range(len(lons_pgw)):
            lon_pgw=lons_pgw[xy]-360
            lat_pgw=lats_pgw[xy]
            latR1,latR2,lonR1,lonR2 = radians(lat_ref),radians(lat_pgw),radians(lon_ref),radians(lon_pgw)
            dlon = lonR2 - lonR1
            dlat = latR2 - latR1
            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
            if a>1:
                a=1
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distances1.append(radius * c)
            ids_pgw.append(id_pgw[xy])
        if min(distances1)<300:
            if not df_tp2.loc[df_tp2['time']==df_ref_step['time']].empty:
                search_tp2=df_tp2.loc[df_tp2['time']==df_ref_step['time']]
                lon_ref=df_ref_step.lon
                lat_ref=df_ref_step.lat
                lons_tp2=search_tp2.lon.values
                lats_tp2=search_tp2.lat.values
                id_tp2=search_tp2.id.values
                distances2=[]
                ids_tp2=[]
                for xy in range(len(lons_tp2)):
                    lon_tp2=lons_tp2[xy]-360
                    lat_tp2=lats_tp2[xy]
                    latR1,latR2,lonR1,lonR2 = radians(lat_ref),radians(lat_tp2),radians(lon_ref),radians(lon_tp2)
                    dlon = lonR2 - lonR1
                    dlat = latR2 - latR1
                    a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                    if a>1:
                        a=1
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    distances2.append(radius * c)
                    ids_tp2.append(id_tp2[xy])
                if min(distances2)<300:
                    dfs_pgw_ids.append(ids_pgw[np.argmin(distances1)])
                    dfs_tp2_ids.append(ids_tp2[np.argmin(distances2)])
                    dfs_ref_ids.append(df_ref_step.id)

dfs_pgw_ids_unique=list(set(dfs_pgw_ids))
dfs_tp2_ids_unique=list(set(dfs_tp2_ids))
dfs_ref_ids_unique=list(set(dfs_ref_ids))
#nr. of storms ref / timesteps: 76/10,749
df_pgw = df_pgw.loc[df_pgw['id'].isin(dfs_pgw_ids_unique)].reset_index(drop=True)
df_tp2 = df_tp2.loc[df_tp2['id'].isin(dfs_tp2_ids_unique)].reset_index(drop=True)
df_ref = df_ref.loc[df_ref['id'].isin(dfs_ref_ids_unique)].reset_index(drop=True)
#nr. of storms ref / timesteps: 31/4,778
#nr. of storms pgw / timesteps: 30/5,169
#nr. of storms tp2 / timesteps: 31/5,696

a=df_ref.w10m_max.values
b=df_pgw.w10m_max.values
c=df_tp2.w10m_max.values
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='RACMO-REF (n=%s)'%(f"{len(a):,}"),color='tab:orange')
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='RACMO-PGW (n=%s)'%(f"{len(b):,}"),color='tab:green')
plt.plot(np.sort(c), np.linspace(0, 1, len(c), endpoint=False),label='RACMO-TP2 (n=%s)'%(f"{len(c):,}"),color='tab:grey')
plt.title('CDF maximum windspeed TC1+ of matching TCs')
plt.ylabel('Probability')
plt.xlabel('Windspeed [m/s]')
plt.legend()
#plt.text(4.5,1.08,'c)',fontweight='bold')
plt.savefig('FigS7_cdf_windspeed_timesteps_TC1_matching_TCs.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix S8: CDF max wind speed TC3+ of TCs that are in RACMO-REF, RACMO-PGW, and RACMO-TP2
radius= 6371
ids=df_ref['id'].unique()
dfs_pgw_ids=[]
dfs_ref_ids=[]
dfs_tp2_ids=[]
for i in range(len(df_ref)):
    print(i)
    df_ref_step = df_ref.loc[i]
    if not df_pgw.loc[df_pgw['time']==df_ref_step['time']].empty:
        search_pgw=df_pgw.loc[df_pgw['time']==df_ref_step['time']]
        lon_ref=df_ref_step.lon
        lat_ref=df_ref_step.lat
        lons_pgw=search_pgw.lon.values
        lats_pgw=search_pgw.lat.values
        id_pgw=search_pgw.id.values
        distances1=[]
        ids_pgw=[]
        for xy in range(len(lons_pgw)):
            lon_pgw=lons_pgw[xy]-360
            lat_pgw=lats_pgw[xy]
            latR1,latR2,lonR1,lonR2 = radians(lat_ref),radians(lat_pgw),radians(lon_ref),radians(lon_pgw)
            dlon = lonR2 - lonR1
            dlat = latR2 - latR1
            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
            if a>1:
                a=1
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distances1.append(radius * c)
            ids_pgw.append(id_pgw[xy])
        if min(distances1)<300:
            if not df_tp2.loc[df_tp2['time']==df_ref_step['time']].empty:
                search_tp2=df_tp2.loc[df_tp2['time']==df_ref_step['time']]
                lon_ref=df_ref_step.lon
                lat_ref=df_ref_step.lat
                lons_tp2=search_tp2.lon.values
                lats_tp2=search_tp2.lat.values
                id_tp2=search_tp2.id.values
                distances2=[]
                ids_tp2=[]
                for xy in range(len(lons_tp2)):
                    lon_tp2=lons_tp2[xy]-360
                    lat_tp2=lats_tp2[xy]
                    latR1,latR2,lonR1,lonR2 = radians(lat_ref),radians(lat_tp2),radians(lon_ref),radians(lon_tp2)
                    dlon = lonR2 - lonR1
                    dlat = latR2 - latR1
                    a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                    if a>1:
                        a=1
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    distances2.append(radius * c)
                    ids_tp2.append(id_tp2[xy])
                if min(distances2)<300:
                    dfs_pgw_ids.append(ids_pgw[np.argmin(distances1)])
                    dfs_tp2_ids.append(ids_tp2[np.argmin(distances2)])
                    dfs_ref_ids.append(df_ref_step.id)

dfs_pgw_ids_unique=list(set(dfs_pgw_ids))
dfs_tp2_ids_unique=list(set(dfs_tp2_ids))
dfs_ref_ids_unique=list(set(dfs_ref_ids))
#nr. of storms ref / timesteps: 76/10,749
df_pgw = df_pgw.loc[df_pgw['id'].isin(dfs_pgw_ids_unique)].reset_index(drop=True)
df_tp2 = df_tp2.loc[df_tp2['id'].isin(dfs_tp2_ids_unique)].reset_index(drop=True)
df_ref = df_ref.loc[df_ref['id'].isin(dfs_ref_ids_unique)].reset_index(drop=True)
#nr. of storms ref / timesteps: 31/4,778
#nr. of storms pgw / timesteps: 30/5,169
#nr. of storms tp2 / timesteps: 31/5,696

a=df_ref.w10m_max.values
b=df_pgw.w10m_max.values
c=df_tp2.w10m_max.values
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='RACMO-REF (n=%s)'%(f"{len(a):,}"),color='tab:orange')
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='RACMO-PGW (n=%s)'%(f"{len(b):,}"),color='tab:green')
plt.plot(np.sort(c), np.linspace(0, 1, len(c), endpoint=False),label='RACMO-TP2 (n=%s)'%(f"{len(c):,}"),color='tab:grey')
plt.title('CDF maximum windspeed TC3+ of matching TCs')
plt.ylabel('Probability')
plt.xlabel('Windspeed [m/s]')
plt.legend()
#plt.text(4.5,1.08,'d)',fontweight='bold')
plt.savefig('FigS8_cdf_windspeed_timesteps_TC3_matching_TCs.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 3: Delta's
ibt_w10ms=[]
ids=df_ibt['SID'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibt.loc[df_ibt['SID']==storm_id]
    ibt_w10ms.append(df_storm.USA_WIND.max())

ref_w10ms=[]
ids=df_ref['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ref.loc[df_ref['id']==storm_id]
    ref_w10ms.append(df_storm.w10m_max.max())

tp2_w10ms=[]
ids=df_tp2['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_tp2.loc[df_tp2['id']==storm_id]
    tp2_w10ms.append(df_storm.w10m_max.max())

pgw_w10ms=[]
ids=df_pgw['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_pgw.loc[df_pgw['id']==storm_id]
    pgw_w10ms.append(df_storm.w10m_max.max())

#ibt_mean_w10m = np.mean(ibt_w10ms)
#ref_mean_w10m = np.mean(ref_w10ms)
#tp2_mean_w10m = np.mean(tp2_w10ms)
#pgw_mean_w10m = np.mean(pgw_w10ms)

#pgw_mean_w10m_change_abs=pgw_mean_w10m-ref_mean_w10m                     #TS=0.50; TC1=-0.28; TC2=0.78; TC3=1.50; TC4=2.08; TC5=2.26
#pgw_mean_w10m_change_rel=(pgw_mean_w10m-ref_mean_w10m)/ref_mean_w10m*100 #TS=0.82; TC1=-0.46: TC2=1.23; TC3=2.38; TC4=3.24; TC5=3.19
#tp2_mean_w10m_change_abs=tp2_mean_w10m-ref_mean_w10m                     #TS=2.10; TC1=1.73;  TC2=2.21; TC3=3.13; TC4=4.16; TC5=3.32
#tp2_mean_w10m_change_rel=(tp2_mean_w10m-ref_mean_w10m)/tp2_mean_w10m*100 #TS=3.31; TC1=2.71;  TC2=3.40; TC3=4.74; TC4=6.07; TC5=4.50

#pgw_mean_w10m_change_abs=df_pgw.w10m_max.mean()-df_ref.w10m_max.mean()                              #TS=1.35;  TC1=0.83;  TC2=1.50;  TC3=2.72;  TC4=2.44; TC5=2.94
#pgw_mean_w10m_change_rel=(df_pgw.w10m_max.mean()-df_ref.w10m_max.mean())/df_ref.w10m_max.mean()*100 #TS=2.83;  TC1=1.72;  TC2=3.09;  TC3=4.70;  TC4=4.88; TC5=5.28
#tp2_mean_w10m_change_abs=df_tp2.w10m_max.mean()-df_ref.w10m_max.mean()                              #TS=-0.65; TC1=-0.99; TC2=-0.65; TC3=-0.02; TC4=0.54; TC5=-1.89
#tp2_mean_w10m_change_rel=(df_tp2.w10m_max.mean()-df_ref.w10m_max.mean())/df_ref.w10m_max.mean()*100 #TS=-1.37; TC1=-2.05; TC2=-1.34; TC3=-0.03; TC4=1.07; TC5=-3.31

#Figure 3: Intensity dependent delta
#Figure 3a: Intensity change per 10% quantile windspeed
means_ref=[]
df_ref_sorted=df_ref.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_ref=df_ref_sorted[int(np.around(len(df_ref_sorted)/10*(i))):int(np.around(len(df_ref_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_ref.append(mean_ref)

means_pgw=[]
df_pgw_sorted=df_pgw.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_pgw=df_pgw_sorted[int(np.around(len(df_pgw_sorted)/10*(i))):int(np.around(len(df_pgw_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_pgw.append(mean_pgw)

means_tp2=[]
df_tp2_sorted=df_tp2.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_tp2=df_tp2_sorted[int(np.around(len(df_tp2_sorted)/10*(i))):int(np.around(len(df_tp2_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_tp2.append(mean_tp2)

data={'ref':means_ref,'pgw':means_pgw,'tp2':means_tp2}
df_boxes_10=pd.DataFrame(data=data)
df_boxes_10['pgw_abs']=df_boxes_10['pgw']-df_boxes_10['ref']
df_boxes_10['tp2_abs']=df_boxes_10['tp2']-df_boxes_10['ref']
df_boxes_10['pgw_rel']=(df_boxes_10['pgw']-df_boxes_10['ref'])/df_boxes_10['ref']*100
df_boxes_10['tp2_rel']=(df_boxes_10['tp2']-df_boxes_10['ref'])/df_boxes_10['ref']*100
df_boxes_10.index=df_boxes_10.index*10+5

bars_pgw=df_boxes_10.pgw_rel.values
bars_tp2=df_boxes_10.tp2_rel.values

barWidth=0.2
r1 = np.arange(len(df_boxes_10.index.values))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars_pgw, width = barWidth, color = 'tab:green', edgecolor = 'black', capsize=7, label='?PGW')
plt.bar(r2, bars_tp2, width = barWidth, color = 'tab:grey', edgecolor = 'black', capsize=7, label='?TP2')
plt.axhline(y=0,linewidth=1,color='black')
plt.axvline(x=-0.4,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=0.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=1.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=2.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=3.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=4.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=5.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=6.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=7.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=8.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=9.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
#plt.xticks([r + barWidth/2 for r in range(len(df_boxes_10.index.values))],['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'],fontsize=13,rotation=60)#
plt.xticks([-0.4,0.6,1.6,2.6,3.6,4.6,5.6,6.6,7.6,8.6,9.6],['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],fontsize=13)
plt.ylabel('Relative change (%)',fontsize=13)
plt.xlabel('Windspeed quantile boundary',fontsize=13)
plt.ylim((-17.5,7.5))
plt.yticks([-15,-10,-5,0,5],['-15','-10','-5','0','5'],fontsize=13)
plt.title('Relative windspeed change per 10% quantile',fontsize=13)
plt.legend(fontsize=13)
plt.text(-1.85,8.1,'a)',fontweight='bold',fontsize=13)
plt.savefig('Fig4a_windspeed_delta_quantile_relative.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 3b: Intensity change per 10% quantile mslp
means_ref=[]
df_ref_sorted=df_ref.sort_values(by=['mslp_min']).reset_index(drop=True)
for i in range(10):
    mean_ref=df_ref_sorted[int(np.around(len(df_ref_sorted)/10*(i))):int(np.around(len(df_ref_sorted)/10*(i+1)))]['mslp_min'].mean()
    means_ref.append(mean_ref)

means_pgw=[]
df_pgw_sorted=df_pgw.sort_values(by=['mslp_min']).reset_index(drop=True)
for i in range(10):
    mean_pgw=df_pgw_sorted[int(np.around(len(df_pgw_sorted)/10*(i))):int(np.around(len(df_pgw_sorted)/10*(i+1)))]['mslp_min'].mean()
    means_pgw.append(mean_pgw)

means_tp2=[]
df_tp2_sorted=df_tp2.sort_values(by=['mslp_min']).reset_index(drop=True)
for i in range(10):
    mean_tp2=df_tp2_sorted[int(np.around(len(df_tp2_sorted)/10*(i))):int(np.around(len(df_tp2_sorted)/10*(i+1)))]['mslp_min'].mean()
    means_tp2.append(mean_tp2)

data={'ref':means_ref,'pgw':means_pgw,'tp2':means_tp2}
df_boxes_10_mslp=pd.DataFrame(data=data)
df_boxes_10_mslp['pgw_abs']=df_boxes_10_mslp['pgw']-df_boxes_10_mslp['ref']
df_boxes_10_mslp['tp2_abs']=df_boxes_10_mslp['tp2']-df_boxes_10_mslp['ref']
df_boxes_10_mslp['pgw_rel']=(df_boxes_10_mslp['pgw']-df_boxes_10_mslp['ref'])/df_boxes_10_mslp['ref']*100
df_boxes_10_mslp['tp2_rel']=(df_boxes_10_mslp['tp2']-df_boxes_10_mslp['ref'])/df_boxes_10_mslp['ref']*100
df_boxes_10_mslp.index=df_boxes_10_mslp.index*10+5

#pgw_delta_mslp = df_boxes_10_mslp.pgw_rel.values/100
#tp2_delta_mslp = df_boxes_10_mslp.tp2_rel.values/100

bars_pgw=df_boxes_10_mslp.pgw_rel.values
bars_tp2=df_boxes_10_mslp.tp2_rel.values

barWidth=0.2
r1 = np.arange(len(df_boxes_10_mslp.index.values))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars_pgw, width = barWidth, color = 'tab:green', edgecolor = 'black', capsize=7, label='?PGW')
plt.bar(r2, bars_tp2, width = barWidth, color = 'tab:grey', edgecolor = 'black', capsize=7, label='?TP2')
plt.axhline(y=0,linewidth=1,color='black')
plt.axvline(x=-0.4,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=0.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=1.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=2.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=3.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=4.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=5.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=6.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=7.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=8.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.axvline(x=9.6,color='black',alpha=0.5,linestyle='--',lw=0.75)
plt.xticks([-0.4,0.6,1.6,2.6,3.6,4.6,5.6,6.6,7.6,8.6,9.6],['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],fontsize=13)
plt.ylabel('Relative change (%)',fontsize=13)
plt.xlabel('MSLP quantile boundary',fontsize=13)
plt.ylim((-1.2,1.2))
plt.yticks([-1.2,-0.6,0,0.6,1.2],['-1.2','-0.6','0.0','0.6','1.2'],fontsize=13)
plt.title('Relative MSLP change per 10% quantile',fontsize=13)
plt.legend(fontsize=13)
plt.text(-1.9,1.25,'b)',fontweight='bold',fontsize=13)
plt.text(-2.5,1.25,'x',color='white')
plt.savefig('Fig4b_MSLP_delta_quantile_relative.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 3c: TC Maria
df_ibt_sorted=df_ibt.sort_values(by=['USA_WIND']).reset_index(drop=True)
ibtracs = xr.open_dataset('/gpfs/work3/0/einf4186/eucp_knmi/data/IBTrACS/IBTrACS.NA.v04r00.nc')
ibtracs = pd.read_csv('/gpfs/work3/0/einf4186/eucp_knmi/data/IBTrACS/ibtracs.NA.list.v04r00.csv')
df = ibtracs.loc[ibtracs['SEASON'].isin((range(1979,2021)))]
df_keep = df[['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW']]
df_keep['USA_WIND']=pd.to_numeric(df_keep['USA_WIND'],errors='coerce')
df_keep['USA_PRES']=pd.to_numeric(df_keep['USA_PRES'],errors='coerce')
df_keep['USA_WIND']=df_keep['USA_WIND']*0.5144444
df_keep['USA_LON']=pd.to_numeric(df_keep['USA_LON'],errors='coerce')
df_keep['USA_LAT']=pd.to_numeric(df_keep['USA_LAT'],errors='coerce')
df_keep['USA_RMW']=pd.to_numeric(df_keep['USA_RMW'],errors='coerce')
df_keep['USA_RMW']=df_keep['USA_RMW']*1.852
df_keep=df_keep.dropna(subset=['USA_WIND']).reset_index(drop=True)
df_keep=df_keep.dropna(subset=['USA_PRES']).reset_index(drop=True)
df_keep['ISO_TIME']=pd.to_datetime(df_keep['ISO_TIME'])

df_maria = df_keep[df_keep.SID=='2017260N12310'].reset_index(drop=True)

i=0
df_maria['PGW_WIND']=np.where(df_maria['USA_WIND'].values<df_ibt.USA_WIND.quantile(1/10),df_maria['USA_WIND']+df_maria['USA_WIND']*pgw_delta[i],df_maria['USA_WIND'])
df_maria['TP2_WIND']=np.where(df_maria['USA_WIND'].values<df_ibt.USA_WIND.quantile(1/10),df_maria['USA_WIND']+df_maria['USA_WIND']*tp2_delta[i],df_maria['USA_WIND'])
for i in range(1,9):
    df_maria['PGW_WIND']=np.where((df_maria['USA_WIND'].values>=df_ibt.USA_WIND.quantile(i/10))&(df_maria['USA_WIND'].values<df_ibt.USA_WIND.quantile((i+1)/10)),df_maria['USA_WIND']+df_maria['USA_WIND']*pgw_delta[i],df_maria['PGW_WIND'])
    df_maria['TP2_WIND']=np.where((df_maria['USA_WIND'].values>=df_ibt.USA_WIND.quantile(i/10))&(df_maria['USA_WIND'].values<df_ibt.USA_WIND.quantile((i+1)/10)),df_maria['USA_WIND']+df_maria['USA_WIND']*tp2_delta[i],df_maria['TP2_WIND'])

i=9
df_maria['PGW_WIND']=np.where(df_maria['USA_WIND'].values>=df_ibt.USA_WIND.quantile(9/10),df_maria['USA_WIND']+df_maria['USA_WIND']*pgw_delta[i],df_maria['PGW_WIND'])
df_maria['TP2_WIND']=np.where(df_maria['USA_WIND'].values>=df_ibt.USA_WIND.quantile(9/10),df_maria['USA_WIND']+df_maria['USA_WIND']*tp2_delta[i],df_maria['TP2_WIND'])

#Maria
#T33=landfall Puerto Rico
#T21=landfall Dominica
fig, ax1 = plt.subplots()
ax2=ax1.twinx()

ax1.plot(df_maria.TP2_WIND.values,label='IBTrACS + ?TP2',color='tab:grey')
ax1.plot(df_maria.PGW_WIND.values,label='IBTrACS + ?PGW',color='tab:green')
ax1.plot(df_maria.USA_WIND.values,label='IBTrACS',color='tab:blue')

for i in range(1,10):
    ibt_q=df_ibt_sorted.USA_WIND.quantile(i/10)#   [int(np.around(len(df_tp2_sorted)/10*(i))):int(np.around(len(df_tp2_sorted)/10*(i+1)))]['w10m_max'].mean()
    ax2.axhline(ibt_q,linewidth=1,color='black',ls='--',alpha=0.5,lw=0.75)

ax2.axhline(df_ibt_sorted.USA_WIND.min(),linewidth=1,color='black')
ax2.axhline(df_ibt_sorted.USA_WIND.max(),linewidth=1,color='black')
ax2.set_ylabel('IBTrACS quantile boundaries',fontsize=13)
ax1.set_ylabel('Windspeed [m/s]',fontsize=13)
ax1.set_xlabel('Time [hr]',fontsize=13)
plt.title('TC Maria windspeed',fontsize=13)
ax1.set_ylim((10,90))
ax2.set_ylim((10,90))
ax1.set_xlim((0,len(df_maria)))
ax2.set_xlim((0,len(df_maria)))
ax1.set_xticks([0,20,40,60,80,100,120],['0','20','40','60','80','100','120'],fontsize=13)
ax1.set_yticks([10,20,30,40,50,60,70,80,90],['10','20','30','40','50','60','70','80','90'],fontsize=13)
ax2.set_yticks(([df_ibt_sorted.USA_WIND.min(),df_ibt_sorted.USA_WIND.quantile(0.1),df_ibt_sorted.USA_WIND.quantile(0.2),df_ibt_sorted.USA_WIND.quantile(0.3),df_ibt_sorted.USA_WIND.quantile(0.4),df_ibt_sorted.USA_WIND.quantile(0.5),df_ibt_sorted.USA_WIND.quantile(0.6),df_ibt_sorted.USA_WIND.quantile(0.7),df_ibt_sorted.USA_WIND.quantile(0.8),df_ibt_sorted.USA_WIND.quantile(0.9),df_ibt_sorted.USA_WIND.max()]),(np.round(np.arange(0,1.05,0.1),1)),fontsize=13)
lines, labels = ax1.get_legend_handles_labels()
ax2.legend(lines, labels, loc='upper right',fontsize=13)
plt.text(-15,92.5,'c)',fontweight='bold',fontsize=13)
plt.savefig('Fig4c_Maria.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure 3d: TC Dorian
df_ibt_sorted=df_ibt.sort_values(by=['USA_WIND']).reset_index(drop=True)
ibtracs = xr.open_dataset('/gpfs/work3/0/einf4186/eucp_knmi/data/IBTrACS/IBTrACS.NA.v04r00.nc')
ibtracs = pd.read_csv('/gpfs/work3/0/einf4186/eucp_knmi/data/IBTrACS/ibtracs.NA.list.v04r00.csv')
df = ibtracs.loc[ibtracs['SEASON'].isin((range(1979,2021)))]
df_keep = df[['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW']]
df_keep['USA_WIND']=pd.to_numeric(df_keep['USA_WIND'],errors='coerce')
df_keep['USA_PRES']=pd.to_numeric(df_keep['USA_PRES'],errors='coerce')
df_keep['USA_WIND']=df_keep['USA_WIND']*0.5144444
df_keep['USA_LON']=pd.to_numeric(df_keep['USA_LON'],errors='coerce')
df_keep['USA_LAT']=pd.to_numeric(df_keep['USA_LAT'],errors='coerce')
df_keep['USA_RMW']=pd.to_numeric(df_keep['USA_RMW'],errors='coerce')
df_keep['USA_RMW']=df_keep['USA_RMW']*1.852
df_keep=df_keep.dropna(subset=['USA_WIND']).reset_index(drop=True)
df_keep=df_keep.dropna(subset=['USA_PRES']).reset_index(drop=True)
df_keep['ISO_TIME']=pd.to_datetime(df_keep['ISO_TIME'])
df_dorian = df_keep[df_keep.SID=='2019236N10314'].reset_index(drop=True)

pgw_delta = df_boxes_10.pgw_rel.values/100
tp2_delta = df_boxes_10.tp2_rel.values/100

i=0
df_dorian['PGW_WIND']=np.where(df_dorian['USA_WIND'].values<df_ibt.USA_WIND.quantile(1/10),df_dorian['USA_WIND']+df_dorian['USA_WIND']*pgw_delta[i],df_dorian['USA_WIND'])
df_dorian['TP2_WIND']=np.where(df_dorian['USA_WIND'].values<df_ibt.USA_WIND.quantile(1/10),df_dorian['USA_WIND']+df_dorian['USA_WIND']*tp2_delta[i],df_dorian['USA_WIND'])
for i in range(1,9):
    df_dorian['PGW_WIND']=np.where((df_dorian['USA_WIND'].values>=df_ibt.USA_WIND.quantile(i/10))&(df_dorian['USA_WIND'].values<df_ibt.USA_WIND.quantile((i+1)/10)),df_dorian['USA_WIND']+df_dorian['USA_WIND']*pgw_delta[i],df_dorian['PGW_WIND'])
    df_dorian['TP2_WIND']=np.where((df_dorian['USA_WIND'].values>=df_ibt.USA_WIND.quantile(i/10))&(df_dorian['USA_WIND'].values<df_ibt.USA_WIND.quantile((i+1)/10)),df_dorian['USA_WIND']+df_dorian['USA_WIND']*tp2_delta[i],df_dorian['TP2_WIND'])

i=9
df_dorian['PGW_WIND']=np.where(df_dorian['USA_WIND'].values>=df_ibt.USA_WIND.quantile(9/10),df_dorian['USA_WIND']+df_dorian['USA_WIND']*pgw_delta[i],df_dorian['PGW_WIND'])
df_dorian['TP2_WIND']=np.where(df_dorian['USA_WIND'].values>=df_ibt.USA_WIND.quantile(9/10),df_dorian['USA_WIND']+df_dorian['USA_WIND']*tp2_delta[i],df_dorian['TP2_WIND'])

fig, ax1 = plt.subplots()
ax2=ax1.twinx()

ax1.plot(df_dorian.TP2_WIND.values[2:],label='IBTrACS + ?TP2',color='tab:grey')
ax1.plot(df_dorian.PGW_WIND.values[2:],label='IBTrACS + ?PGW',color='tab:green')
ax1.plot(df_dorian.USA_WIND.values[2:],label='IBTrACS',color='tab:blue')

for i in range(1,10):
    ibt_q=df_ibt_sorted.USA_WIND.quantile(i/10)#   [int(np.around(len(df_tp2_sorted)/10*(i))):int(np.around(len(df_tp2_sorted)/10*(i+1)))]['w10m_max'].mean()
    ax2.axhline(ibt_q,linewidth=1,color='black',ls='--',alpha=0.5,lw=0.75)

ax2.axhline(df_ibt_sorted.USA_WIND.min(),linewidth=1,color='black')
ax2.axhline(df_ibt_sorted.USA_WIND.max(),linewidth=1,color='black')
ax2.set_ylabel('IBTrACS quantile boundaries',fontsize=13)
ax1.set_ylabel('Windspeed [m/s]',fontsize=13)
ax1.set_xlabel('Time [hr]',fontsize=13)
plt.title('TC Dorian windspeed',fontsize=13)
ax1.set_xlim((0,len(df_dorian)-2))
ax2.set_xlim((0,len(df_dorian)-2))
ax1.set_ylim((10,90))
ax2.set_ylim((10,90))
ax1.set_xticks([0,20,40,60,80,100,120],['0','20','40','60','80','100','120'],fontsize=13)
ax1.set_yticks([10,20,30,40,50,60,70,80,90],['10','20','30','40','50','60','70','80','90'],fontsize=13)
ax2.set_yticks(([df_ibt_sorted.USA_WIND.min(),df_ibt_sorted.USA_WIND.quantile(0.1),df_ibt_sorted.USA_WIND.quantile(0.2),df_ibt_sorted.USA_WIND.quantile(0.3),df_ibt_sorted.USA_WIND.quantile(0.4),df_ibt_sorted.USA_WIND.quantile(0.5),df_ibt_sorted.USA_WIND.quantile(0.6),df_ibt_sorted.USA_WIND.quantile(0.7),df_ibt_sorted.USA_WIND.quantile(0.8),df_ibt_sorted.USA_WIND.quantile(0.9),df_ibt_sorted.USA_WIND.max()]),(np.round(np.arange(0,1.05,0.1),1)),fontsize=13)
lines, labels = ax1.get_legend_handles_labels()
ax2.legend(lines, labels, loc='upper left',fontsize=13)
plt.text(-15,92.5,'d)',fontweight='bold',fontsize=13)
plt.savefig('Fig4d_Dorian.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix 9: Intensity dependent delta TC1+ in REF, PGW & TP2 
radius= 6371
ids=df_ref['id'].unique()
dfs_pgw_ids=[]
dfs_ref_ids=[]
dfs_tp2_ids=[]
for i in range(len(df_ref)):
    print(i)
    df_ref_step = df_ref.loc[i]
    if not df_pgw.loc[df_pgw['time']==df_ref_step['time']].empty:
        search_pgw=df_pgw.loc[df_pgw['time']==df_ref_step['time']]
        lon_ref=df_ref_step.lon
        lat_ref=df_ref_step.lat
        lons_pgw=search_pgw.lon.values
        lats_pgw=search_pgw.lat.values
        id_pgw=search_pgw.id.values
        distances1=[]
        ids_pgw=[]
        for xy in range(len(lons_pgw)):
            lon_pgw=lons_pgw[xy]-360
            lat_pgw=lats_pgw[xy]
            latR1,latR2,lonR1,lonR2 = radians(lat_ref),radians(lat_pgw),radians(lon_ref),radians(lon_pgw)
            dlon = lonR2 - lonR1
            dlat = latR2 - latR1
            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
            if a>1:
                a=1
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distances1.append(radius * c)
            ids_pgw.append(id_pgw[xy])
        if min(distances1)<300:
            if not df_tp2.loc[df_tp2['time']==df_ref_step['time']].empty:
                search_tp2=df_tp2.loc[df_tp2['time']==df_ref_step['time']]
                lon_ref=df_ref_step.lon
                lat_ref=df_ref_step.lat
                lons_tp2=search_tp2.lon.values
                lats_tp2=search_tp2.lat.values
                id_tp2=search_tp2.id.values
                distances2=[]
                ids_tp2=[]
                for xy in range(len(lons_tp2)):
                    lon_tp2=lons_tp2[xy]-360
                    lat_tp2=lats_tp2[xy]
                    latR1,latR2,lonR1,lonR2 = radians(lat_ref),radians(lat_tp2),radians(lon_ref),radians(lon_tp2)
                    dlon = lonR2 - lonR1
                    dlat = latR2 - latR1
                    a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                    if a>1:
                        a=1
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    distances2.append(radius * c)
                    ids_tp2.append(id_tp2[xy])
                if min(distances2)<300:
                    dfs_pgw_ids.append(ids_pgw[np.argmin(distances1)])
                    dfs_tp2_ids.append(ids_tp2[np.argmin(distances2)])
                    dfs_ref_ids.append(df_ref_step.id)

dfs_pgw_ids_unique=list(set(dfs_pgw_ids))
dfs_tp2_ids_unique=list(set(dfs_tp2_ids))
dfs_ref_ids_unique=list(set(dfs_ref_ids))
#nr. of storms ref / timesteps: 76/10,749
df_pgw = df_pgw.loc[df_pgw['id'].isin(dfs_pgw_ids_unique)].reset_index(drop=True)
df_tp2 = df_tp2.loc[df_tp2['id'].isin(dfs_tp2_ids_unique)].reset_index(drop=True)
df_ref = df_ref.loc[df_ref['id'].isin(dfs_ref_ids_unique)].reset_index(drop=True)
#nr. of storms ref / timesteps: 31/4,778
#nr. of storms pgw / timesteps: 30/5,169
#nr. of storms tp2 / timesteps: 31/5,696

a=df_ref.w10m_max.values
b=df_pgw.w10m_max.values
c=df_tp2.w10m_max.values

means_ref=[]
df_ref_sorted=df_ref.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_ref=df_ref_sorted[int(np.around(len(df_ref_sorted)/10*(i))):int(np.around(len(df_ref_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_ref.append(mean_ref)

means_pgw=[]
df_pgw_sorted=df_pgw.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_pgw=df_pgw_sorted[int(np.around(len(df_pgw_sorted)/10*(i))):int(np.around(len(df_pgw_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_pgw.append(mean_pgw)

means_tp2=[]
df_tp2_sorted=df_tp2.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_tp2=df_tp2_sorted[int(np.around(len(df_tp2_sorted)/10*(i))):int(np.around(len(df_tp2_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_tp2.append(mean_tp2)

data={'ref':means_ref,'pgw':means_pgw,'tp2':means_tp2}
df_boxes_10=pd.DataFrame(data=data)
df_boxes_10['pgw_abs']=df_boxes_10['pgw']-df_boxes_10['ref']
df_boxes_10['tp2_abs']=df_boxes_10['tp2']-df_boxes_10['ref']
df_boxes_10['pgw_rel']=(df_boxes_10['pgw']-df_boxes_10['ref'])/df_boxes_10['ref']*100
df_boxes_10['tp2_rel']=(df_boxes_10['tp2']-df_boxes_10['ref'])/df_boxes_10['ref']*100
df_boxes_10.index=df_boxes_10.index*10+5

bars_pgw=df_boxes_10.pgw_rel.values
bars_tp2=df_boxes_10.tp2_rel.values

barWidth=0.2
r1 = np.arange(len(df_boxes_10.index.values))
r2 = [x + barWidth for x in r1]

#Figure Appendix 9: Intensity change per 10% quantile
plt.bar(r1, bars_pgw, width = barWidth, color = 'tab:green', edgecolor = 'black', capsize=7, label='pgw')
plt.bar(r2, bars_tp2, width = barWidth, color = 'tab:grey', edgecolor = 'black', capsize=7, label='tp2')
plt.axhline(y=0,linewidth=1,color='black')
plt.xticks([r + barWidth for r in range(len(df_boxes_10.index.values))],['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
plt.ylabel('Relative change (%)')
plt.xlabel('Windspeed quantile')
plt.title('Relative windspeed change per quantile for matching TCs CAT1+')
plt.legend()
#plt.text(-2,8.45,'b)',fontweight='bold')
plt.savefig('FigS9_delta_quantile_relative_matching_TC1+.png',dpi=600,bbox_inches='tight')
plt.close()

#Figure Appendix 10: Intensity change per 10% quantile
plt.bar(r1, bars_pgw, width = barWidth, color = 'tab:green', edgecolor = 'black', capsize=7, label='pgw')
plt.bar(r2, bars_tp2, width = barWidth, color = 'tab:grey', edgecolor = 'black', capsize=7, label='tp2')
plt.axhline(y=0,linewidth=1,color='black')
plt.xticks([r + barWidth for r in range(len(df_boxes_10.index.values))],['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
plt.ylabel('Relative change (%)')
plt.xlabel('Windspeed quantile')
plt.title('Relative windspeed change per quantile for matching TCs CAT3+')
plt.legend()
#plt.text(-1.75,8.2,'a)',fontweight='bold')
plt.savefig('FigS10_delta_quantile_relative_matching_TC3+.png',dpi=600,bbox_inches='tight')
plt.close()

#prepare TCs for input to holland model
#find fulll dorian and maria tracks
df_ibt_sorted=df_ibt.sort_values(by=['USA_WIND']).reset_index(drop=True)
ibtracs = xr.open_dataset('/gpfs/work3/0/einf4186/eucp_knmi/data/IBTrACS/IBTrACS.NA.v04r00.nc')
ibtracs = pd.read_csv('/gpfs/work3/0/einf4186/eucp_knmi/data/IBTrACS/ibtracs.NA.list.v04r00.csv')
df = ibtracs.loc[ibtracs['SEASON'].isin((range(1979,2021)))]
df_keep = df[['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW']]
df_keep['USA_WIND']=pd.to_numeric(df_keep['USA_WIND'],errors='coerce')
df_keep['USA_PRES']=pd.to_numeric(df_keep['USA_PRES'],errors='coerce')
df_keep['USA_WIND']=df_keep['USA_WIND']*0.5144444
df_keep['USA_LON']=pd.to_numeric(df_keep['USA_LON'],errors='coerce')
df_keep['USA_LAT']=pd.to_numeric(df_keep['USA_LAT'],errors='coerce')
df_keep['USA_RMW']=pd.to_numeric(df_keep['USA_RMW'],errors='coerce')
df_keep['USA_RMW']=df_keep['USA_RMW']*1.852
#df_keep['
df_keep=df_keep.dropna(subset=['USA_WIND']).reset_index(drop=True)
df_keep=df_keep.dropna(subset=['USA_PRES']).reset_index(drop=True)
df_keep['ISO_TIME']=pd.to_datetime(df_keep['ISO_TIME'])
df_dorian = df_keep[df_keep.SID=='2019236N10314'].reset_index(drop=True)

pgw_delta = df_boxes_10.pgw_rel.values/100
tp2_delta = df_boxes_10.tp2_rel.values/100

i=0
df_dorian['PGW_WIND']=np.where(df_dorian['USA_WIND'].values<df_ibt.USA_WIND.quantile(1/10),df_dorian['USA_WIND']+df_dorian['USA_WIND']*pgw_delta[i],df_dorian['USA_WIND'])
df_dorian['TP2_WIND']=np.where(df_dorian['USA_WIND'].values<df_ibt.USA_WIND.quantile(1/10),df_dorian['USA_WIND']+df_dorian['USA_WIND']*tp2_delta[i],df_dorian['USA_WIND'])
for i in range(1,9):
    df_dorian['PGW_WIND']=np.where((df_dorian['USA_WIND'].values>=df_ibt.USA_WIND.quantile(i/10))&(df_dorian['USA_WIND'].values<df_ibt.USA_WIND.quantile((i+1)/10)),df_dorian['USA_WIND']+df_dorian['USA_WIND']*pgw_delta[i],df_dorian['PGW_WIND'])
    df_dorian['TP2_WIND']=np.where((df_dorian['USA_WIND'].values>=df_ibt.USA_WIND.quantile(i/10))&(df_dorian['USA_WIND'].values<df_ibt.USA_WIND.quantile((i+1)/10)),df_dorian['USA_WIND']+df_dorian['USA_WIND']*tp2_delta[i],df_dorian['TP2_WIND'])

i=9
df_dorian['PGW_WIND']=np.where(df_dorian['USA_WIND'].values>=df_ibt.USA_WIND.quantile(9/10),df_dorian['USA_WIND']+df_dorian['USA_WIND']*pgw_delta[i],df_dorian['PGW_WIND'])
df_dorian['TP2_WIND']=np.where(df_dorian['USA_WIND'].values>=df_ibt.USA_WIND.quantile(9/10),df_dorian['USA_WIND']+df_dorian['USA_WIND']*tp2_delta[i],df_dorian['TP2_WIND'])

df_maria = df_keep[df_keep.SID=='2017260N12310'].reset_index(drop=True)

i=0
df_maria['PGW_WIND']=np.where(df_maria['USA_WIND'].values<df_ibt.USA_WIND.quantile(1/10),df_maria['USA_WIND']+df_maria['USA_WIND']*pgw_delta[i],df_maria['USA_WIND'])
df_maria['TP2_WIND']=np.where(df_maria['USA_WIND'].values<df_ibt.USA_WIND.quantile(1/10),df_maria['USA_WIND']+df_maria['USA_WIND']*tp2_delta[i],df_maria['USA_WIND'])
for i in range(1,9):
    df_maria['PGW_WIND']=np.where((df_maria['USA_WIND'].values>=df_ibt.USA_WIND.quantile(i/10))&(df_maria['USA_WIND'].values<df_ibt.USA_WIND.quantile((i+1)/10)),df_maria['USA_WIND']+df_maria['USA_WIND']*pgw_delta[i],df_maria['PGW_WIND'])
    df_maria['TP2_WIND']=np.where((df_maria['USA_WIND'].values>=df_ibt.USA_WIND.quantile(i/10))&(df_maria['USA_WIND'].values<df_ibt.USA_WIND.quantile((i+1)/10)),df_maria['USA_WIND']+df_maria['USA_WIND']*tp2_delta[i],df_maria['TP2_WIND'])

i=9
df_maria['PGW_WIND']=np.where(df_maria['USA_WIND'].values>=df_ibt.USA_WIND.quantile(9/10),df_maria['USA_WIND']+df_maria['USA_WIND']*pgw_delta[i],df_maria['PGW_WIND'])
df_maria['TP2_WIND']=np.where(df_maria['USA_WIND'].values>=df_ibt.USA_WIND.quantile(9/10),df_maria['USA_WIND']+df_maria['USA_WIND']*tp2_delta[i],df_maria['TP2_WIND'])

#1. Maria
df_maria_Holland=df_maria.copy(deep=True)
df_maria_Holland['Year']=np.full(len(df_maria),2017)
df_maria_Holland['Month']=np.full(len(df_maria),9)
df_maria_Holland['TC number']=np.full(len(df_maria),0)
df_maria_Holland['Time step']=np.arange(0,(len(df_maria)*3),step=3)#      np.full(len(df_maria),0)
df_maria_Holland['Basin ID']=np.full(len(df_maria),1)
df_maria_Holland['Latitude']=df_maria['USA_LAT']
df_maria_Holland['Longitude']=df_maria['USA_LON']
df_maria_Holland['Minimum pressure']=df_maria['USA_PRES']
df_maria_Holland['Maximum wind speed']=df_maria['USA_WIND']
df_maria_Holland['Radius to maximum winds']=df_maria['USA_RMW']
df_maria_Holland['Category']=np.full(len(df_maria),0)
df_maria_Holland['Landfall']=np.full(len(df_maria),0)
df_maria_Holland['Distance to land']=np.full(len(df_maria),0)
df_maria_Holland=df_maria_Holland.drop(columns=['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW','PGW_WIND','TP2_WIND'])
df_maria_Holland.to_csv('Maria_REF.csv',header=False, index=False)

plt.plot((df_dorian.USA_PRES-1000)*-1)
plt.plot(df_dorian.USA_WIND)
plt.plot(df_dorian.PGW_WIND)
plt.plot(df_dorian.TP2_WIND)
plt.show()

#also need pressure change
means_ref=[]
df_ref_sorted=df_ref.sort_values(by=['mslp_min']).reset_index(drop=True)
for i in range(10):
    mean_ref=df_ref_sorted[int(np.around(len(df_ref_sorted)/10*(i))):int(np.around(len(df_ref_sorted)/10*(i+1)))]['mslp_min'].mean()
    means_ref.append(mean_ref)

means_pgw=[]
df_pgw_sorted=df_pgw.sort_values(by=['mslp_min']).reset_index(drop=True)
for i in range(10):
    mean_pgw=df_pgw_sorted[int(np.around(len(df_pgw_sorted)/10*(i))):int(np.around(len(df_pgw_sorted)/10*(i+1)))]['mslp_min'].mean()
    means_pgw.append(mean_pgw)

means_tp2=[]
df_tp2_sorted=df_tp2.sort_values(by=['mslp_min']).reset_index(drop=True)
for i in range(10):
    mean_tp2=df_tp2_sorted[int(np.around(len(df_tp2_sorted)/10*(i))):int(np.around(len(df_tp2_sorted)/10*(i+1)))]['mslp_min'].mean()
    means_tp2.append(mean_tp2)

data={'ref':means_ref,'pgw':means_pgw,'tp2':means_tp2}
df_boxes_10_mslp=pd.DataFrame(data=data)
df_boxes_10_mslp['pgw_abs']=df_boxes_10_mslp['pgw']-df_boxes_10_mslp['ref']
df_boxes_10_mslp['tp2_abs']=df_boxes_10_mslp['tp2']-df_boxes_10_mslp['ref']
df_boxes_10_mslp['pgw_rel']=(df_boxes_10_mslp['pgw']-df_boxes_10_mslp['ref'])/df_boxes_10_mslp['ref']*100
df_boxes_10_mslp['tp2_rel']=(df_boxes_10_mslp['tp2']-df_boxes_10_mslp['ref'])/df_boxes_10_mslp['ref']*100
df_boxes_10_mslp.index=df_boxes_10_mslp.index*10+5

pgw_delta_mslp = df_boxes_10_mslp.pgw_rel.values/100
tp2_delta_mslp = df_boxes_10_mslp.tp2_rel.values/100

#add column to dataframe with delta adjusted mslp
i=0
df_dorian['PGW_PRES']=np.where(df_dorian['USA_PRES'].values<df_ibt.USA_PRES.quantile(1/10),df_dorian['USA_PRES']+df_dorian['USA_PRES']*pgw_delta_mslp[i],df_dorian['USA_PRES'])
df_dorian['TP2_PRES']=np.where(df_dorian['USA_PRES'].values<df_ibt.USA_PRES.quantile(1/10),df_dorian['USA_PRES']+df_dorian['USA_PRES']*tp2_delta_mslp[i],df_dorian['USA_PRES'])
for i in range(1,9):
    df_dorian['PGW_PRES']=np.where((df_dorian['USA_PRES'].values>=df_ibt.USA_PRES.quantile(i/10))&(df_dorian['USA_PRES'].values<df_ibt.USA_PRES.quantile((i+1)/10)),df_dorian['USA_PRES']+df_dorian['USA_PRES']*pgw_delta_mslp[i],df_dorian['PGW_PRES'])
    df_dorian['TP2_PRES']=np.where((df_dorian['USA_PRES'].values>=df_ibt.USA_PRES.quantile(i/10))&(df_dorian['USA_PRES'].values<df_ibt.USA_PRES.quantile((i+1)/10)),df_dorian['USA_PRES']+df_dorian['USA_PRES']*tp2_delta_mslp[i],df_dorian['TP2_PRES'])

i=9
df_dorian['PGW_PRES']=np.where(df_dorian['USA_PRES'].values>=df_ibt.USA_PRES.quantile(9/10),df_dorian['USA_PRES']+df_dorian['USA_PRES']*pgw_delta_mslp[i],df_dorian['PGW_PRES'])
df_dorian['TP2_PRES']=np.where(df_dorian['USA_PRES'].values>=df_ibt.USA_PRES.quantile(9/10),df_dorian['USA_PRES']+df_dorian['USA_PRES']*tp2_delta_mslp[i],df_dorian['TP2_PRES'])

plt.plot((df_dorian.USA_PRES-1000))
plt.plot((df_dorian.PGW_PRES-1000))
plt.plot((df_dorian.TP2_PRES-1000))
plt.plot(df_dorian.USA_WIND)
plt.plot(df_dorian.PGW_WIND)
plt.plot(df_dorian.TP2_WIND)
plt.show()

i=0
df_maria['PGW_PRES']=np.where(df_maria['USA_PRES'].values<df_ibt.USA_PRES.quantile(1/10),df_maria['USA_PRES']+df_maria['USA_PRES']*pgw_delta_mslp[i],df_maria['USA_PRES'])
df_maria['TP2_PRES']=np.where(df_maria['USA_PRES'].values<df_ibt.USA_PRES.quantile(1/10),df_maria['USA_PRES']+df_maria['USA_PRES']*tp2_delta_mslp[i],df_maria['USA_PRES'])
for i in range(1,9):
    df_maria['PGW_PRES']=np.where((df_maria['USA_PRES'].values>=df_ibt.USA_PRES.quantile(i/10))&(df_maria['USA_PRES'].values<df_ibt.USA_PRES.quantile((i+1)/10)),df_maria['USA_PRES']+df_maria['USA_PRES']*pgw_delta_mslp[i],df_maria['PGW_PRES'])
    df_maria['TP2_PRES']=np.where((df_maria['USA_PRES'].values>=df_ibt.USA_PRES.quantile(i/10))&(df_maria['USA_PRES'].values<df_ibt.USA_PRES.quantile((i+1)/10)),df_maria['USA_PRES']+df_maria['USA_PRES']*tp2_delta_mslp[i],df_maria['TP2_PRES'])

i=9
df_maria['PGW_PRES']=np.where(df_maria['USA_PRES'].values>=df_ibt.USA_PRES.quantile(9/10),df_maria['USA_PRES']+df_maria['USA_PRES']*pgw_delta_mslp[i],df_maria['PGW_PRES'])
df_maria['TP2_PRES']=np.where(df_maria['USA_PRES'].values>=df_ibt.USA_PRES.quantile(9/10),df_maria['USA_PRES']+df_maria['USA_PRES']*tp2_delta_mslp[i],df_maria['TP2_PRES'])

plt.plot((df_maria.USA_PRES-1000))
plt.plot((df_maria.PGW_PRES-1000))
plt.plot((df_maria.TP2_PRES-1000))
plt.plot(df_maria.USA_WIND)
plt.plot(df_maria.PGW_WIND)
plt.plot(df_maria.TP2_WIND)
plt.show()

#prepare TCs for input to holland model
#1. Maria
df_maria_Holland=df_maria.copy(deep=True)
df_maria_Holland['Year']=np.full(len(df_maria),2017)
df_maria_Holland['Month']=np.full(len(df_maria),9)
df_maria_Holland['TC number']=np.full(len(df_maria),0)
df_maria_Holland['Time step']=np.arange(0,(len(df_maria)*3),step=3)#      np.full(len(df_maria),0)
df_maria_Holland['Basin ID']=np.full(len(df_maria),1)
df_maria_Holland['Latitude']=df_maria['USA_LAT']
df_maria_Holland['Longitude']=df_maria['USA_LON']
df_maria_Holland['Minimum pressure']=df_maria['USA_PRES']
df_maria_Holland['Maximum wind speed']=df_maria['USA_WIND']
df_maria_Holland['Radius to maximum winds']=df_maria['USA_RMW']
df_maria_Holland['Category']=np.full(len(df_maria),0)
df_maria_Holland['Landfall']=np.full(len(df_maria),0)
df_maria_Holland['Distance to land']=np.full(len(df_maria),0)
df_maria_Holland=df_maria_Holland.drop(columns=['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW','PGW_WIND','TP2_WIND','PGW_PRES','TP2_PRES'])
df_maria_Holland.to_csv('Maria_REF.csv',header=False, index=False)

df_maria_Holland=df_maria.copy(deep=True)
df_maria_Holland['Year']=np.full(len(df_maria),2017)
df_maria_Holland['Month']=np.full(len(df_maria),9)
df_maria_Holland['TC number']=np.full(len(df_maria),0)
df_maria_Holland['Time step']=np.arange(0,(len(df_maria)*3),step=3)#      np.full(len(df_maria),0)
df_maria_Holland['Basin ID']=np.full(len(df_maria),1)
df_maria_Holland['Latitude']=df_maria['USA_LAT']
df_maria_Holland['Longitude']=df_maria['USA_LON']
df_maria_Holland['Minimum pressure']=df_maria['PGW_PRES']
df_maria_Holland['Maximum wind speed']=df_maria['PGW_WIND']
df_maria_Holland['Radius to maximum winds']=df_maria['USA_RMW']
df_maria_Holland['Category']=np.full(len(df_maria),0)
df_maria_Holland['Landfall']=np.full(len(df_maria),0)
df_maria_Holland['Distance to land']=np.full(len(df_maria),0)
df_maria_Holland=df_maria_Holland.drop(columns=['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW','PGW_WIND','TP2_WIND','PGW_PRES','TP2_PRES'])
df_maria_Holland.to_csv('Maria_PGW.csv',header=False, index=False)

df_maria_Holland=df_maria.copy(deep=True)
df_maria_Holland['Year']=np.full(len(df_maria),2017)
df_maria_Holland['Month']=np.full(len(df_maria),9)
df_maria_Holland['TC number']=np.full(len(df_maria),0)
df_maria_Holland['Time step']=np.arange(0,(len(df_maria)*3),step=3)#      np.full(len(df_maria),0)
df_maria_Holland['Basin ID']=np.full(len(df_maria),1)
df_maria_Holland['Latitude']=df_maria['USA_LAT']
df_maria_Holland['Longitude']=df_maria['USA_LON']
df_maria_Holland['Minimum pressure']=df_maria['TP2_PRES']
df_maria_Holland['Maximum wind speed']=df_maria['TP2_WIND']
df_maria_Holland['Radius to maximum winds']=df_maria['USA_RMW']
df_maria_Holland['Category']=np.full(len(df_maria),0)
df_maria_Holland['Landfall']=np.full(len(df_maria),0)
df_maria_Holland['Distance to land']=np.full(len(df_maria),0)
df_maria_Holland=df_maria_Holland.drop(columns=['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW','PGW_WIND','TP2_WIND','PGW_PRES','TP2_PRES'])
df_maria_Holland.to_csv('Maria_TP2.csv',header=False, index=False)

#2. Dorian
df_dorian_Holland=df_dorian.copy(deep=True)
df_dorian_Holland['Year']=np.full(len(df_dorian),2017)
df_dorian_Holland['Month']=np.full(len(df_dorian),9)
df_dorian_Holland['TC number']=np.full(len(df_dorian),0)
df_dorian_Holland['Time step']=np.arange(0,(len(df_dorian)*3),step=3)#      np.full(len(df_dorian),0)
df_dorian_Holland['Basin ID']=np.full(len(df_dorian),1)
df_dorian_Holland['Latitude']=df_dorian['USA_LAT']
df_dorian_Holland['Longitude']=df_dorian['USA_LON']
df_dorian_Holland['Minimum pressure']=df_dorian['USA_PRES']
df_dorian_Holland['Maximum wind speed']=df_dorian['USA_WIND']
df_dorian_Holland['Radius to maximum winds']=df_dorian['USA_RMW']
df_dorian_Holland['Category']=np.full(len(df_dorian),0)
df_dorian_Holland['Landfall']=np.full(len(df_dorian),0)
df_dorian_Holland['Distance to land']=np.full(len(df_dorian),0)
df_dorian_Holland=df_dorian_Holland.drop(columns=['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW','PGW_WIND','TP2_WIND','PGW_PRES','TP2_PRES'])
df_dorian_Holland.to_csv('Dorian_REF.csv',header=False, index=False)

df_dorian_Holland=df_dorian.copy(deep=True)
df_dorian_Holland['Year']=np.full(len(df_dorian),2017)
df_dorian_Holland['Month']=np.full(len(df_dorian),9)
df_dorian_Holland['TC number']=np.full(len(df_dorian),0)
df_dorian_Holland['Time step']=np.arange(0,(len(df_dorian)*3),step=3)#      np.full(len(df_dorian),0)
df_dorian_Holland['Basin ID']=np.full(len(df_dorian),1)
df_dorian_Holland['Latitude']=df_dorian['USA_LAT']
df_dorian_Holland['Longitude']=df_dorian['USA_LON']
df_dorian_Holland['Minimum pressure']=df_dorian['PGW_PRES']
df_dorian_Holland['Maximum wind speed']=df_dorian['PGW_WIND']
df_dorian_Holland['Radius to maximum winds']=df_dorian['USA_RMW']
df_dorian_Holland['Category']=np.full(len(df_dorian),0)
df_dorian_Holland['Landfall']=np.full(len(df_dorian),0)
df_dorian_Holland['Distance to land']=np.full(len(df_dorian),0)
df_dorian_Holland=df_dorian_Holland.drop(columns=['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW','PGW_WIND','TP2_WIND','PGW_PRES','TP2_PRES'])
df_dorian_Holland.to_csv('Dorian_PGW.csv',header=False, index=False)

df_dorian_Holland=df_dorian.copy(deep=True)
df_dorian_Holland['Year']=np.full(len(df_dorian),2017)
df_dorian_Holland['Month']=np.full(len(df_dorian),9)
df_dorian_Holland['TC number']=np.full(len(df_dorian),0)
df_dorian_Holland['Time step']=np.arange(0,(len(df_dorian)*3),step=3)#      np.full(len(df_dorian),0)
df_dorian_Holland['Basin ID']=np.full(len(df_dorian),1)
df_dorian_Holland['Latitude']=df_dorian['USA_LAT']
df_dorian_Holland['Longitude']=df_dorian['USA_LON']
df_dorian_Holland['Minimum pressure']=df_dorian['TP2_PRES']
df_dorian_Holland['Maximum wind speed']=df_dorian['TP2_WIND']
df_dorian_Holland['Radius to maximum winds']=df_dorian['USA_RMW']
df_dorian_Holland['Category']=np.full(len(df_dorian),0)
df_dorian_Holland['Landfall']=np.full(len(df_dorian),0)
df_dorian_Holland['Distance to land']=np.full(len(df_dorian),0)
df_dorian_Holland=df_dorian_Holland.drop(columns=['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND','USA_RMW','PGW_WIND','TP2_WIND','PGW_PRES','TP2_PRES'])
df_dorian_Holland.to_csv('Dorian_TP2.csv',header=False, index=False)
























#only needed for maximum certain TC strength (quantile mapping)
TS=18; TC1=33.06; TC2=42.78; TC3=49.44; TC4=58.06; TC5=70
TC=TC4 #with TC4 meaning, TC should not reach Cat 4

ids=df_ref['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_ref.loc[df_ref['id']==storm_id]
    if df_storm.w10m_max.max()<TC:
        dfs.append(df_storm)

df_ref = pd.concat(dfs).sort_values(by=['id','time']).reset_index(drop=True)

ids=df_pgw['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_pgw.loc[df_pgw['id']==storm_id]
    if df_storm.w10m_max.max()<TC:
        dfs.append(df_storm)

df_pgw = pd.concat(dfs).sort_values(by=['id','time']).reset_index(drop=True)

ids=df_tp2['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_tp2.loc[df_tp2['id']==storm_id]
    if df_storm.w10m_max.max()<TC:
        dfs.append(df_storm)

df_tp2 = pd.concat(dfs).sort_values(by=['id','time']).reset_index(drop=True)


ids=df_ibt['SID'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibt.loc[df_ibt['SID']==storm_id]
    if df_storm.USA_WIND.max()>TC:
        dfs.append(df_storm)

df_ibt = pd.concat(dfs).sort_values(by=['SID','ISO_TIME']).reset_index(drop=True)

ibt_w10ms=[]
ids=df_ibtracs['SID'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibtracs.loc[df_ibtracs['SID']==storm_id]
    ibt_w10ms.append(df_storm.USA_WIND.max())

ref_w10ms=[]
ids=df_ref['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ref.loc[df_ref['id']==storm_id]
    ref_w10ms.append(df_storm.w10m_max.max())

tp2_w10ms=[]
ids=df_tp2['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_tp2.loc[df_tp2['id']==storm_id]
    tp2_w10ms.append(df_storm.w10m_max.max())

pgw_w10ms=[]
ids=df_pgw['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_pgw.loc[df_pgw['id']==storm_id]
    pgw_w10ms.append(df_storm.w10m_max.max())

ibt_mean_w10m = np.mean(ibt_w10ms)
ref_mean_w10m = np.mean(ref_w10ms)
tp2_mean_w10m = np.mean(tp2_w10ms)
pgw_mean_w10m = np.mean(pgw_w10ms)

pgw_mean_w10m_change_abs=pgw_mean_w10m-ref_mean_w10m                     #TS=0.50; TC1=-0.28; TC2=0.78; TC3=1.50; TC4=2.08; TC5=2.26
pgw_mean_w10m_change_rel=(pgw_mean_w10m-ref_mean_w10m)/ref_mean_w10m*100 #TS=0.82; TC1=-0.46: TC2=1.23; TC3=2.38; TC4=3.24; TC5=3.19
tp2_mean_w10m_change_abs=tp2_mean_w10m-ref_mean_w10m                     #TS=2.10; TC1=1.73;  TC2=2.21; TC3=3.13; TC4=4.16; TC5=3.32
tp2_mean_w10m_change_rel=(tp2_mean_w10m-ref_mean_w10m)/tp2_mean_w10m*100 #TS=3.31; TC1=2.71;  TC2=3.40; TC3=4.74; TC4=6.07; TC5=4.50

pgw_mean_w10m_change_abs=df_pgw.w10m_max.mean()-df_ref.w10m_max.mean()                              #TS=1.35;  TC1=0.83;  TC2=1.50;  TC3=2.72;  TC4=2.44; TC5=2.94
pgw_mean_w10m_change_rel=(df_pgw.w10m_max.mean()-df_ref.w10m_max.mean())/df_ref.w10m_max.mean()*100 #TS=2.83;  TC1=1.72;  TC2=3.09;  TC3=4.70;  TC4=4.88; TC5=5.28
tp2_mean_w10m_change_abs=df_tp2.w10m_max.mean()-df_ref.w10m_max.mean()                              #TS=-0.65; TC1=-0.99; TC2=-0.65; TC3=-0.02; TC4=0.54; TC5=-1.89
tp2_mean_w10m_change_rel=(df_tp2.w10m_max.mean()-df_ref.w10m_max.mean())/df_ref.w10m_max.mean()*100 #TS=-1.37; TC1=-2.05; TC2=-1.34; TC3=-0.03; TC4=1.07; TC5=-3.31

#10% boxes
means_ref=[]
df_ref_sorted=df_ref.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_ref=df_ref_sorted[int(np.around(len(df_ref_sorted)/10*(i))):int(np.around(len(df_ref_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_ref.append(mean_ref)

means_pgw=[]
df_pgw_sorted=df_pgw.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_pgw=df_pgw_sorted[int(np.around(len(df_pgw_sorted)/10*(i))):int(np.around(len(df_pgw_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_pgw.append(mean_pgw)

means_tp2=[]
df_tp2_sorted=df_tp2.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(10):
    mean_tp2=df_tp2_sorted[int(np.around(len(df_tp2_sorted)/10*(i))):int(np.around(len(df_tp2_sorted)/10*(i+1)))]['w10m_max'].mean()
    means_tp2.append(mean_tp2)

data={'ref':means_ref,'pgw':means_pgw,'tp2':means_tp2}
df_boxes_10=pd.DataFrame(data=data)
df_boxes_10['pgw_abs']=df_boxes_10['pgw']-df_boxes_10['ref']
df_boxes_10['tp2_abs']=df_boxes_10['tp2']-df_boxes_10['ref']
df_boxes_10['pgw_rel']=(df_boxes_10['pgw']-df_boxes_10['ref'])/df_boxes_10['ref']*100
df_boxes_10['tp2_rel']=(df_boxes_10['tp2']-df_boxes_10['ref'])/df_boxes_10['ref']*100
df_boxes_10.index=df_boxes_10.index*10+5

plt.rcParams['figure.figsize']=(10,8)
df_boxes_10[['pgw_abs','tp2_abs']].plot()
plt.ylabel('absolute change (m/s)')
plt.xlabel('wind speed quantile')
plt.xticks(np.arange(5,100,step=10),['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
plt.savefig('windspeed_delta_absolute_TC1_boxes_10.png',dpi=600)
plt.close()

plt.rcParams['figure.figsize']=(10,8)
#df_boxes_10[['pgw_rel','tp2_rel']].plot()
plt.bar(df_boxes_10.index.values,df_boxes_10.pgw_rel.values,label='pgw')
plt.bar(df_boxes_10.index.values,df_boxes_10.tp2_rel.values,label='tp2')
plt.ylabel('relative change (%)')
plt.xlabel('wind speed quantile')
plt.title('relative windspeed change per quantile (10%) for TCs CAT3')
plt.xticks(np.arange(5,100,step=10),['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
plt.legend()
plt.savefig('windspeed_delta_relative_TC3_boxes_10.png',dpi=600)
plt.close()

bars_pgw=df_boxes_10.pgw_rel.values
bars_tp2=df_boxes_10.tp2_rel.values

barWidth=0.2
r1 = np.arange(len(df_boxes_10.index.values))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars_pgw, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='pgw')
plt.bar(r2, bars_tp2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='tp2')

plt.xticks([r + barWidth for r in range(len(df_boxes_10.index.values))],['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
plt.ylabel('relative change (m/s)')
plt.xlabel('windspeed quantile')
plt.title('relative windspeed change per quantile for TCs CAT1+')
plt.legend()
plt.savefig('windspeed_delta_relative_TC1+_boxes_10.png',dpi=600)
plt.close()

#5% boxes
means_ref=[]
df_ref_sorted=df_ref.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(20):
    mean_ref=df_ref_sorted[int(np.around(len(df_ref_sorted)/20*(i))):int(np.around(len(df_ref_sorted)/20*(i+1)))]['w10m_max'].mean()
    means_ref.append(mean_ref)

means_pgw=[]
df_pgw_sorted=df_pgw.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(20):
    mean_pgw=df_pgw_sorted[int(np.around(len(df_pgw_sorted)/20*(i))):int(np.around(len(df_pgw_sorted)/20*(i+1)))]['w10m_max'].mean()
    means_pgw.append(mean_pgw)

means_tp2=[]
df_tp2_sorted=df_tp2.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(20):
    mean_tp2=df_tp2_sorted[int(np.around(len(df_tp2_sorted)/20*(i))):int(np.around(len(df_tp2_sorted)/20*(i+1)))]['w10m_max'].mean()
    means_tp2.append(mean_tp2)

data={'ref':means_ref,'pgw':means_pgw,'tp2':means_tp2}
df_boxes_5=pd.DataFrame(data=data)
df_boxes_5['pgw_abs']=df_boxes_5['pgw']-df_boxes_5['ref']
df_boxes_5['tp2_abs']=df_boxes_5['tp2']-df_boxes_5['ref']
df_boxes_5['pgw_rel']=(df_boxes_5['pgw']-df_boxes_5['ref'])/df_boxes_5['ref']*100
df_boxes_5['tp2_rel']=(df_boxes_5['tp2']-df_boxes_5['ref'])/df_boxes_5['ref']*100

df_boxes_5['constant_pgw_abs']=-0.28
df_boxes_5['constant_tp2_abs']=1.73
df_boxes_5['constant_pgw_rel']=-0.46
df_boxes_5['constant_tp2_rel']=2.71
df_boxes_5.index=df_boxes_5.index*5+2.5

plt.rcParams['figure.figsize']=(20,15)
df_boxes_5[['pgw_abs','tp2_abs','constant_pgw_abs','constant_tp2_abs']].plot()
plt.ylabel('absolute change (m/s)')
plt.xlabel('wind speed quantile')
plt.xticks(np.arange(2.5,100,step=5),['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'])
plt.savefig('windspeed_delta_absolute_TC1.png',dpi=600)
plt.close()

plt.rcParams['figure.figsize']=(20,15)
df_boxes_5[['pgw_rel','tp2_rel','constant_pgw_rel','constant_tp2_rel']].plot()
plt.ylabel('relative change (%)')
plt.xlabel('wind speed quantile')
plt.xticks(np.arange(2.5,100,step=5),['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'])
plt.savefig('windspeed_delta_relative_TC1.png',dpi=600)
plt.close()

bars_pgw=df_boxes_5.pgw_rel.values
bars_tp2=df_boxes_5.tp2_rel.values

barWidth=0.4
r1 = np.arange(len(df_boxes_5.index.values))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars_pgw, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='pgw')
plt.bar(r2, bars_tp2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='tp2')

plt.xticks([r + barWidth for r in range(len(df_boxes_5.index.values))],['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'],fontsize=6)
plt.ylabel('relative change (m/s)')
plt.xlabel('windspeed quantile')
plt.title('relative windspeed change per quantile')
plt.legend()
plt.savefig('windspeed_delta_relative_TS_boxes_5_in_ibtracs.png',dpi=600)
plt.close()

#would be cool to select those tracks that are also in ibtracs
#first only selection criteria that tracks are in both sim and obs at the RACMO model boundary
#loop over tracks in ibtracs
#per track search for corresponding track id in each of the sims
#keep that track id and create df of those tracks
#ref
radius= 6371
ids=df_ibt['SID'].unique()
dfs_ref_ids=[]
for i in range(len(df_ibt)):
    print(i)
    df_ibt_step = df_ibt.loc[i]
    if not df_ref.loc[df_ref['time']==df_ibt_step['ISO_TIME']].empty:
        search_ref=df_ref.loc[df_ref['time']==df_ibt_step['ISO_TIME']]
        lon_ibt=df_ibt_step.USA_LON
        lat_ibt=df_ibt_step.USA_LAT
        lons_sim=search_ref.lon.values
        lats_sim=search_ref.lat.values
        id_sim=search_ref.id.values
        distances=[]
        ids_sim=[]
        for xy in range(len(lons_sim)):
            lon_sim=lons_sim[xy]-360
            lat_sim=lats_sim[xy]
            latR1,latR2,lonR1,lonR2 = radians(lat_ibt),radians(lat_sim),radians(lon_ibt),radians(lon_sim)
            dlon = lonR2 - lonR1
            dlat = latR2 - latR1
            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
            if a>1:
                a=1
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distances.append(radius * c)
            ids_sim.append(id_sim[xy])
        if min(distances)<300:
            dfs_ref_ids.append(ids_sim[np.argmin(distances)])

dfs_ref_ids_unique=list(set(dfs_ref_ids))
df_ref = df_ref.loc[df_ref['id'].isin(dfs_ref_ids_unique)].reset_index(drop=True)

#pgw
ids=df_ibt['SID'].unique()
dfs_pgw_ids=[]
for i in range(len(df_ibt)):
    print(i)
    df_ibt_step = df_ibt.loc[i]
    if not df_pgw.loc[df_pgw['time']==df_ibt_step['ISO_TIME']].empty:
        search_pgw=df_pgw.loc[df_pgw['time']==df_ibt_step['ISO_TIME']]
        lon_ibt=df_ibt_step.USA_LON
        lat_ibt=df_ibt_step.USA_LAT
        lons_sim=search_pgw.lon.values
        lats_sim=search_pgw.lat.values
        id_sim=search_pgw.id.values
        distances=[]
        ids_sim=[]
        for xy in range(len(lons_sim)):
            lon_sim=lons_sim[xy]-360
            lat_sim=lats_sim[xy]
            latR1,latR2,lonR1,lonR2 = radians(lat_ibt),radians(lat_sim),radians(lon_ibt),radians(lon_sim)
            dlon = lonR2 - lonR1
            dlat = latR2 - latR1
            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
            if a>1:
                a=1
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distances.append(radius * c)
            ids_sim.append(id_sim[xy])
        if min(distances)<300:
            dfs_pgw_ids.append(ids_sim[np.argmin(distances)])

dfs_pgw_ids_unique=list(set(dfs_pgw_ids))
df_pgw = df_pgw.loc[df_pgw['id'].isin(dfs_pgw_ids_unique)].reset_index(drop=True)
    
#tp2
ids=df_ibt['SID'].unique()
dfs_tp2_ids=[]
for i in range(len(df_ibt)):
    print(i)
    df_ibt_step = df_ibt.loc[i]
    if not df_tp2.loc[df_tp2['time']==df_ibt_step['ISO_TIME']].empty:
        search_tp2=df_tp2.loc[df_tp2['time']==df_ibt_step['ISO_TIME']]
        lon_ibt=df_ibt_step.USA_LON
        lat_ibt=df_ibt_step.USA_LAT
        lons_sim=search_tp2.lon.values
        lats_sim=search_tp2.lat.values
        id_sim=search_tp2.id.values
        distances=[]
        ids_sim=[]
        for xy in range(len(lons_sim)):
            lon_sim=lons_sim[xy]-360
            lat_sim=lats_sim[xy]
            latR1,latR2,lonR1,lonR2 = radians(lat_ibt),radians(lat_sim),radians(lon_ibt),radians(lon_sim)
            dlon = lonR2 - lonR1
            dlat = latR2 - latR1
            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
            if a>1:
                a=1
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distances.append(radius * c)
            ids_sim.append(id_sim[xy])
        if min(distances)<300:
            dfs_tp2_ids.append(ids_sim[np.argmin(distances)])

dfs_tp2_ids_unique=list(set(dfs_tp2_ids))
df_tp2 = df_tp2.loc[df_tp2['id'].isin(dfs_tp2_ids_unique)].reset_index(drop=True)

#boxes 5 with selection of tracks that are also in ibtracs
means_ref=[]
df_ref_sorted=df_ref.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(20):
    mean_ref=df_ref_sorted[int(np.around(len(df_ref_sorted)/20*(i))):int(np.around(len(df_ref_sorted)/20*(i+1)))]['w10m_max'].mean()
    means_ref.append(mean_ref)

means_pgw=[]
df_pgw_sorted=df_pgw.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(20):
    mean_pgw=df_pgw_sorted[int(np.around(len(df_pgw_sorted)/20*(i))):int(np.around(len(df_pgw_sorted)/20*(i+1)))]['w10m_max'].mean()
    means_pgw.append(mean_pgw)

means_tp2=[]
df_tp2_sorted=df_tp2.sort_values(by=['w10m_max']).reset_index(drop=True)
for i in range(20):
    mean_tp2=df_tp2_sorted[int(np.around(len(df_tp2_sorted)/20*(i))):int(np.around(len(df_tp2_sorted)/20*(i+1)))]['w10m_max'].mean()
    means_tp2.append(mean_tp2)

data={'ref':means_ref,'pgw':means_pgw,'tp2':means_tp2}
df_boxes_5=pd.DataFrame(data=data)
df_boxes_5['pgw_abs']=df_boxes_5['pgw']-df_boxes_5['ref']
df_boxes_5['tp2_abs']=df_boxes_5['tp2']-df_boxes_5['ref']
df_boxes_5['pgw_rel']=(df_boxes_5['pgw']-df_boxes_5['ref'])/df_boxes_5['ref']*100
df_boxes_5['tp2_rel']=(df_boxes_5['tp2']-df_boxes_5['ref'])/df_boxes_5['ref']*100

df_boxes_5['constant_pgw_abs']=#-0.28
df_boxes_5['constant_tp2_abs']=#1.73
df_boxes_5['constant_pgw_rel']=#-0.46
df_boxes_5['constant_tp2_rel']=#2.71
df_boxes_5.index=df_boxes_5.index*5+2.5

plt.rcParams['figure.figsize']=(20,15)
df_boxes_5[['pgw_abs','tp2_abs','constant_pgw_abs','constant_tp2_abs']].plot()
plt.ylabel('absolute change (m/s)')
plt.xlabel('wind speed quantile')
plt.xticks(np.arange(2.5,100,step=5),['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'])
plt.savefig('windspeed_delta_absolute_TS_ibtracks_sel.png',dpi=600)
plt.close()

plt.rcParams['figure.figsize']=(20,15)
df_boxes_5[['pgw_rel','tp2_rel','constant_pgw_rel','constant_tp2_rel']].plot()
plt.ylabel('relative change (%)')
plt.xlabel('wind speed quantile')
plt.xticks(np.arange(2.5,100,step=5),['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'])
plt.savefig('windspeed_delta_relative_TS_ibtracs_sel.png',dpi=600)
plt.close()

ibt_mean_w10m = np.mean(ibt_w10ms)
ref_mean_w10m = np.mean(ref_w10ms)
tp2_mean_w10m = np.mean(tp2_w10ms)
pgw_mean_w10m = np.mean(pgw_w10ms)

pgw_mean_w10m_change_abs=pgw_mean_w10m-ref_mean_w10m                     #TS=0.50; TC1=-0.28; TC2=0.78; TC3=1.50; TC4=2.08; TC5=2.26
pgw_mean_w10m_change_rel=(pgw_mean_w10m-ref_mean_w10m)/ref_mean_w10m*100 #TS=0.82; TC1=-0.46: TC2=1.23; TC3=2.38; TC4=3.24; TC5=3.19
tp2_mean_w10m_change_abs=tp2_mean_w10m-ref_mean_w10m                     #TS=2.10; TC1=1.73;  TC2=2.21; TC3=3.13; TC4=4.16; TC5=3.32
tp2_mean_w10m_change_rel=(tp2_mean_w10m-ref_mean_w10m)/tp2_mean_w10m*100 #TS=3.31; TC1=2.71;  TC2=3.40; TC3=4.74; TC4=6.07; TC5=4.50

pgw_mean_w10m_change_abs=df_pgw.w10m_max.mean()-df_ref.w10m_max.mean()                              #TS=1.35;  TC1=0.83;  TC2=1.50;  TC3=2.72;  TC4=2.44; TC5=2.94
pgw_mean_w10m_change_rel=(df_pgw.w10m_max.mean()-df_ref.w10m_max.mean())/df_ref.w10m_max.mean()*100 #TS=2.83;  TC1=1.72;  TC2=3.09;  TC3=4.70;  TC4=4.88; TC5=5.28
tp2_mean_w10m_change_abs=df_tp2.w10m_max.mean()-df_ref.w10m_max.mean()                              #TS=-0.65; TC1=-0.99; TC2=-0.65; TC3=-0.02; TC4=0.54; TC5=-1.89
tp2_mean_w10m_change_rel=(df_tp2.w10m_max.mean()-df_ref.w10m_max.mean())/df_ref.w10m_max.mean()*100 #TS=-1.37; TC1=-2.05; TC2=-1.34; TC3=-0.03; TC4=1.07; TC5=-3.31

#CDF timesteps
a=df_ref.w10m_max.values
b=df_pgw.w10m_max.values
c=df_tp2.w10m_max.values
d=df_ibt.USA_WIND.values
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='ref (n=%s)'%(len(a)))
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='pgw (n=%s)'%(len(b)))
plt.plot(np.sort(c), np.linspace(0, 1, len(c), endpoint=False),label='tp2 (n=%s)'%(len(c)))
plt.plot(np.sort(d), np.linspace(0, 1, len(d), endpoint=False),label='ibtracs (n=%s)'%(len(d)))
plt.title('TC3 cdf maximum windspeed per timestep')
plt.ylabel('probability')
plt.xlabel('windspeed (m/s)')
plt.legend()
plt.savefig('cdf_windspeed_timesteps_TC3.png',dpi=600)
plt.close()

#CDF cyclones
ref_w10ms=[]
ids=df_ref['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ref.loc[df_ref['id']==storm_id]
    ref_w10ms.append(df_storm.w10m_max.max())

tp2_w10ms=[]
ids=df_tp2['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_tp2.loc[df_tp2['id']==storm_id]
    tp2_w10ms.append(df_storm.w10m_max.max())

pgw_w10ms=[]
ids=df_pgw['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_pgw.loc[df_pgw['id']==storm_id]
    pgw_w10ms.append(df_storm.w10m_max.max())

ibt_w10ms=[]
ids=df_ibt['SID'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibt.loc[df_ibt['SID']==storm_id]
    ibt_w10ms.append(df_storm.USA_WIND.max())

a=np.asarray(ref_w10ms)
b=np.asarray(pgw_w10ms)
c=np.asarray(tp2_w10ms)
d=np.asarray(ibt_w10ms)
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='ref (n=%s)'%(len(a)))
plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='pgw (n=%s)'%(len(b)))
plt.plot(np.sort(c), np.linspace(0, 1, len(c), endpoint=False),label='tp2 (n=%s)'%(len(c)))
plt.plot(np.sort(d), np.linspace(0, 1, len(d), endpoint=False),label='ibtracs (n=%s)'%(len(d)))
plt.title('TC3 cdf maximum windspeed per track')
plt.ylabel('probability')
plt.xlabel('windspeed (m/s)')
plt.legend()
plt.savefig('cdf_windspeed_track_TC3.png',dpi=600)
plt.close()

plt.close()
ids=df_ref['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ref.loc[df_ref['id']==storm_id]
    plt.plot(range(len(df_storm)),df_storm.w10m_max.values,linewidth=0.5)
    plt.title('REF TC5 windspeed over time')
    plt.ylabel('windspeed (m/s)')
    plt.xlabel('time (hr)')

plt.savefig('ref_windspeed_over_time_TC5.png',dpi=600)

plt.close()
ids=df_pgw['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_pgw.loc[df_pgw['id']==storm_id]
    plt.plot(range(len(df_storm)),df_storm.w10m_max.values,linewidth=0.5)
    plt.title('PGW TC5 windspeed over time')
    plt.ylabel('windspeed (m/s)')
    plt.xlabel('time (hr)')

plt.savefig('pgw_windspeed_over_time_TC5.png',dpi=600)

plt.close()
ids=df_tp2['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_tp2.loc[df_tp2['id']==storm_id]
    plt.plot(range(len(df_storm)),df_storm.w10m_max.values,linewidth=0.5)
    plt.title('TP2 TC5 windspeed over time')
    plt.ylabel('windspeed (m/s)')
    plt.xlabel('time (hr)')

plt.savefig('tp2_windspeed_over_time_TC5.png',dpi=600)


plt.close()
ids=df_ref['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ref.loc[df_ref['id']==storm_id]
    plt.plot(df_storm.w10m_max.values,linewidth=0.5)
    plt.title('REF TC5 windspeed over time (tc in ibtracs)')
    plt.ylabel('windspeed (m/s)')
    plt.xlabel('time (hr)')
    plt.xlim([-15,315])
    plt.ylim([0,85])

plt.savefig('ref_windspeed_over_time_TC5_ibtracs.png',dpi=600)

plt.close()
ids=df_pgw['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_pgw.loc[df_pgw['id']==storm_id]
    plt.plot(df_storm.w10m_max.values,linewidth=0.5)
    plt.title('PGW TC5 windspeed over time (tc in ibtracs)')
    plt.ylabel('windspeed (m/s)')
    plt.xlabel('time (hr)')
    plt.xlim([-15,315])
    plt.ylim([0,85])

plt.savefig('pgw_windspeed_over_time_TC5_ibtracs.png',dpi=600)

plt.close()
ids=df_tp2['id'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_tp2.loc[df_tp2['id']==storm_id]
    plt.plot(df_storm.w10m_max.values,linewidth=0.5)
    plt.title('TP2 TC5 windspeed over time (tc in ibtracs)')
    plt.ylabel('windspeed (m/s)')
    plt.xlabel('time (hr)')
    plt.xlim([-15,315])
    plt.ylim([0,85])

plt.savefig('tp2_windspeed_over_time_TC5_ibtracs.png',dpi=600)

plt.close()
ids=df_ibt['SID'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibt.loc[df_ibt['SID']==storm_id]
    plt.plot(df_storm.USA_WIND.values,linewidth=0.5)
    plt.title('IBTrACS TC5 windspeed over time')
    plt.ylabel('windspeed (m/s)')
    plt.xlabel('time (hr)')
    plt.xlim([-15,315])
    plt.ylim([0,85])

plt.savefig('ibt_windspeed_over_time_TC5.png',dpi=600)

#spatial plot tracks
from pandas.plotting import register_matplotlib_converters
import warnings
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.cm as cmplt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
warnings.filterwarnings('ignore')

longitude=df_ref.lon.values-360
latitude=df_ref.lat.values
windspeed=df_ref.w10m_max.values

longitude=df_pgw.lon.values-360
latitude=df_pgw.lat.values
windspeed=df_pgw.w10m_max.values

longitude=df_tp2.lon.values-360
latitude=df_tp2.lat.values
windspeed=df_tp2.w10m_max.values

longitude=df_ibt.USA_LON.values
latitude=df_ibt.USA_LAT.values
windspeed=df_ibt.USA_WIND.values

crg = ccrs.PlateCarree()
crgp = ccrs.Robinson()
        
plt.close('all')
fig = plt.figure()
#ax = plt.add_subplot(111)
ax = plt.axes(projection=crgp,zorder=6)
ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='gainsboro',zorder=1,edgecolor='black',linewidth=0.001)
ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), zorder=4,linewidth=0.2)
ax.set_global()
ax.set_extent([-85, -40, 5, 30], crg)
cmap = cmplt.get_cmap('Spectral_r')
#cmaplist = [cmap(i) for i in range(cmap.N)]
#cmaplist[0] = (.5, .5, .5, 1.0)
#cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
#bounds = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000,100000]
bounds = list(range(10,81))#100,250,500,1000,10000]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#cmap.set_under(cmap(0)); cmap.set_over(cmap(cmap.N-1))
#bs=ax.scatter(longitude,latitude,c=storm_tide,cmap=cmap,vmin=0,vmax=100,transform=crg,s=2,norm=norm,zorder=2)
bs=ax.scatter(longitude,latitude,c=windspeed,cmap=cmap,transform=crg,s=3,norm=norm,zorder=2)
cbaxes=fig.add_axes([0.25,0.085,0.5,0.020])
cbar= plt.colorbar(bs,ax=ax,cax=cbaxes,cmap=cmap,spacing='uniform', orientation='horizontal',shrink=0.35,boundaries=bounds,norm=norm,ticks=[10,20,30,40,50,60,70,80])
cbar.set_label('windspeed (m/s)',fontsize=9)
cbar.ax.tick_params(labelsize=9)
#ids=df_ibt['SID'].unique()
#for storm_id in ids:
#    print(storm_id)
#    df_storm = df_ibt.loc[df_ibt['SID']==storm_id]
#    plt.plot(df_storm.USA_LON.values,df_storm.USA_LAT.values,transform=crg,zorder=2,linewidth=0.5)


##plt.grid()
#plt.title('REF TC5 tracks windspeed (tc in ibtracs)',fontsize=9)
##cbaxes=fig.add_axes([0.3,0.125,0.4,0.020])
##plt.colorbar(bs,ax=ax,cax=cbaxes,cmap=cmap,spacing='uniform', orientation='horizontal',shrink=0.35,boundaries=bounds,norm=norm,ticks=[0,10,20,30,40,50,60,70,80])

gl = ax.gridlines(crs=crg, draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',zorder=5)
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlines = True
gl.ylocator = mticker.FixedLocator(np.arange(-5.,26.,5))
gl.xlocator = mticker.FixedLocator(np.arange(-85.,-41.,5))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.tight_layout()
plt.savefig('ibt_tracks_scatter.png',format='png',dpi=600)



#irma plots
means_irma=[]
df_irma_sorted=df_irma.sort_values(by=['USA_WIND']).reset_index(drop=True)
for i in range(20):
    mean_irma=df_irma_sorted[int(np.around(len(df_irma_sorted)/20*(i))):int(np.around(len(df_irma_sorted)/20*(i+1)))]['USA_WIND'].mean()
    means_irma.append(mean_irma)

data={'irma':means_irma}
df_irma_boxes_5=pd.DataFrame(data=data)

df_irma_boxes_5.index=df_irma_boxes_5.index*5+2.5

plt.rcParams['figure.figsize']=(12,9)
plt.bar(df_irma_boxes_5.index.values,df_irma_boxes_5.irma.values,label='irma ibtracs')
plt.ylabel('windspeed (m/s)')
plt.xlabel('windspeed quantile')
plt.xticks(np.arange(2.5,100,step=5),['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'])
plt.legend()
plt.savefig('windspeed_irma.png',dpi=600)
plt.close()

a=np.asarray(df_irma.USA_WIND.values)
plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False),label='irma (n=%s)'%(len(a)))
#plt.plot(np.sort(b), np.linspace(0, 1, len(b), endpoint=False),label='pgw (n=%s)'%(len(b)))
#plt.plot(np.sort(c), np.linspace(0, 1, len(c), endpoint=False),label='tp2 (n=%s)'%(len(c)))
#plt.plot(np.sort(d), np.linspace(0, 1, len(d), endpoint=False),label='ibtracs (n=%s)'%(len(d)))
plt.title('Irma cdf maximum windspeed')
plt.ylabel('probability')
plt.xlabel('windspeed (m/s)')
plt.legend()
plt.savefig('cdf_irma_windspeed.png',dpi=600)
plt.close()

#frequency plots
plt.close()
df = pd.DataFrame({'simulation':['ref','tp2','pgw','ibt'],'frequency':[len(df_ref),len(df_tp2),len(df_pgw),len(df_ibt)]})
df.plot.bar(x='simulation', y='frequency', rot=0)
plt.title('frequency (active hours)')
plt.savefig('frequency_timesteps_TC1+.png',dpi=600)
plt.close()

ref_w10ms=[]
ref_mslps=[]
for storm_id in df_ref.id.unique():
    storm_df=df_ref.loc[df_ref['id'] == storm_id]
    ref_w10ms.append(storm_df['w10m_max'].max())
    ref_mslps.append(storm_df['mslp_min'].min())

tp2_w10ms=[]
tp2_mslps=[]
for storm_id in df_tp2.id.unique():
    storm_df=df_tp2.loc[df_tp2['id'] == storm_id]
    tp2_w10ms.append(storm_df['w10m_max'].max())
    tp2_mslps.append(storm_df['mslp_min'].min())

pgw_w10ms=[]
pgw_mslps=[]
for storm_id in df_pgw.id.unique():
    storm_df=df_pgw.loc[df_pgw['id'] == storm_id]
    pgw_w10ms.append(storm_df['w10m_max'].max())
    pgw_mslps.append(storm_df['mslp_min'].min())

ibt_w10ms=[]
ibt_mslps=[]
for storm_id in df_ibt.SID.unique():
    storm_df=df_ibt.loc[df_ibt['SID'] == storm_id]
    ibt_w10ms.append(storm_df['USA_WIND'].max())
    ibt_mslps.append(storm_df['USA_PRES'].min())

#frequency tracks
plt.close()
df = pd.DataFrame({'simulation':['ref','tp2','pgw','ibt'],'frequency':[len(ref_w10ms),len(tp2_w10ms),len(pgw_w10ms),len(ibt_w10ms)]})
df.plot.bar(x='simulation', y='frequency', rot=0)
plt.title('frequency (number of tracks)')
plt.savefig('frequency_tracks_TS+.png',dpi=600)
plt.close()

#rapid intensification
ref_rapid_ids=[]
for storm_id in df_ref.id.unique():
    storm_df=df_ref.loc[df_ref['id'] == storm_id]
    storm_wind=storm_df['w10m_max'].values
    windspeed_difs=[]
    for i in range(len(storm_wind)-24):
        windspeed_difs.append(storm_wind[i+24]-storm_wind[i])
    if max(windspeed_difs)>15.433:
        ref_rapid_ids.append(storm_id)

df_ref = df_ref.loc[df_ref['id'].isin(ref_rapid_ids)].reset_index(drop=True)


pgw_rapid_ids=[]
for storm_id in df_pgw.id.unique():
    storm_df=df_pgw.loc[df_pgw['id'] == storm_id]
    storm_wind=storm_df['w10m_max'].values
    windspeed_difs=[]
    for i in range(len(storm_wind)-24):
        windspeed_difs.append(storm_wind[i+24]-storm_wind[i])
    if max(windspeed_difs)>15.433:
        pgw_rapid_ids.append(storm_id)

df_pgw = df_pgw.loc[df_pgw['id'].isin(pgw_rapid_ids)].reset_index(drop=True)


tp2_rapid_ids=[]
for storm_id in df_tp2.id.unique():
    storm_df=df_tp2.loc[df_tp2['id'] == storm_id]
    storm_wind=storm_df['w10m_max'].values
    windspeed_difs=[]
    for i in range(len(storm_wind)-24):
        windspeed_difs.append(storm_wind[i+24]-storm_wind[i])
    if max(windspeed_difs)>15.433:
        tp2_rapid_ids.append(storm_id)

df_tp2 = df_tp2.loc[df_tp2['id'].isin(tp2_rapid_ids)].reset_index(drop=True)

ibt_rapid_ids=[]
for storm_id in df_ibt.SID.unique():
    storm_df=df_ibt.loc[df_ibt['SID'] == storm_id]
    storm_wind=storm_df['USA_WIND'].values
    windspeed_difs=[]
    for i in range(len(storm_wind)-24):
        windspeed_difs.append(storm_wind[i+24]-storm_wind[i])
    if max(windspeed_difs)>15.433:
        ibt_rapid_ids.append(storm_id)

df_ibt = df_ibt.loc[df_ibt['SID'].isin(ibt_rapid_ids)].reset_index(drop=True)



# spatial box plot
ref_lons = df_ref.lon.values
ref_lats = df_ref.lat.values
pgw_lons = df_pgw.lon.values
pgw_lats = df_pgw.lat.values
tp2_lons = df_tp2.lon.values
tp2_lats = df_tp2.lat.values
ibt_lons = df_ibt.USA_LON.values
ibt_lats = df_ibt.USA_LAT.values

#lon_bins = np.linspace(-85,-40,46)
#lat_bins = np.linspace(8,28,21)

x=ref_lons-360
y=ref_lats
plt.hist2d(x, y,bins=(45,20), range=[[-85,-40],[8,28]],cmap=plt.cm.jet)#
plt.show()




















