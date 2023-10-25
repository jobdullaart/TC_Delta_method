import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
import os
import glob
from cdo import Cdo
cdo=Cdo()
from scipy.ndimage import minimum_filter
from itertools import *
from operator import itemgetter
import pandas as pd              
from pathlib import Path

#define folder paths
ref_folder='/gpfs/work1/0/ESLRP/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-REF-fERA5/1Hourly_data/'
tp2_folder='/gpfs/work1/0/ESLRP/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-TP2-fERA5/1Hourly_data/'
pgw_folder='/gpfs/work1/0/ESLRP/eucp_knmi/data/RACMO/LCARIB12/wCY33-v563-JJASON-PGW-CMIP5-19models-fERA5/1Hourly_data/'

ref_folder_out='/gpfs/work1/0/ESLRP/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-REF-fERA5/1Hourly_data_regridded/'
tp2_folder_out='/gpfs/work1/0/ESLRP/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-TP2-fERA5/1Hourly_data_regridded/'
pgw_folder_out='/gpfs/work1/0/ESLRP/eucp_knmi/data/RACMO/LCARIB12/wCY33-v563-JJASON-PGW-CMIP5-19models-fERA5/1Hourly_data_regridded/'

#######################
#0.1 read IBTrACS data#
#######################

ibtracs = xr.open_dataset('/gpfs/work1/0/ESLRP/eucp_knmi/data/IBTrACS/IBTrACS.NA.v04r00.nc')
ibtracs = pd.read_csv('/gpfs/work1/0/ESLRP/eucp_knmi/data/IBTrACS/ibtracs.NA.list.v04r00.csv')
df = ibtracs.loc[ibtracs['SEASON'].isin((range(1979,2021)))]
df_keep = df[['SID','ISO_TIME','USA_LAT','USA_LON','USA_PRES','USA_WIND']]
df_keep['USA_WIND']=pd.to_numeric(df_keep['USA_WIND'],errors='coerce')
df_keep['USA_PRES']=pd.to_numeric(df_keep['USA_PRES'],errors='coerce')
df_keep['USA_WIND']=df_keep['USA_WIND']*0.5144444
df_keep['USA_LON']=pd.to_numeric(df_keep['USA_LON'],errors='coerce')
df_keep['USA_LAT']=pd.to_numeric(df_keep['USA_LAT'],errors='coerce')
df_keep=df_keep.dropna(subset=['USA_WIND']).reset_index(drop=True)
df_keep=df_keep.dropna(subset=['USA_PRES']).reset_index(drop=True)
df_keep['ISO_TIME']=pd.to_datetime(df_keep['ISO_TIME'])

ids=df_keep['SID'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_keep.loc[df_keep['SID']==storm_id]
    df_storm = df_storm.set_index('ISO_TIME')
    df_storm = df_storm.resample('1H').interpolate(method='linear')
    df_storm['SID']=storm_id
    df_storm = df_storm.reset_index()
    df_storm = df_storm.loc[(df_storm['USA_LON'] >= -90) & (df_storm['USA_LON'] <= -35)]
    df_storm = df_storm.loc[(df_storm['USA_LAT'] >= 5) & (df_storm['USA_LAT'] <= 30)]
    df_storm = df_storm.loc[(df_storm['USA_WIND'] >= 15)]
    df_storm = df_storm.loc[(df_storm['USA_PRES'] <= 1010)]
    if df_storm.USA_WIND.max()>18:
        if len(df_storm)>71:
            dfs.append(df_storm)

df_ibtracs = pd.concat(dfs).reset_index(drop=True)

#we do this to make sure that the xy is within 3 degrees from the boundary of RACMO
mslp_file_interpolated=xr.open_dataset(glob.glob(ref_folder_out+'mslp.KNMI-2017*.nc')[0])
lons=mslp_file_interpolated.lon.values
lats=mslp_file_interpolated.lat.values
xx,yy=np.meshgrid(lons,lats)
mslp=mslp_file_interpolated.mslp.values[0,:,:]
bool_arr = np.zeros((len(lats),len(lons)), dtype=bool)
for i in range(30,221):
    for j in range(30,521):
        values=mslp[i-30:i+30,j-30:j+30]
        if np.nansum(values>0)==len(values.flatten()):
            bool_arr[i,j]=True

#plt.scatter(xx.flatten()[bool_arr.flatten()],yy.flatten()[bool_arr.flatten()])
#plt.show()

x_box=xx[bool_arr]
y_box=yy[bool_arr]
list_xy=[]
for i in range(len(x_box)):
    list_xy.append(list((np.around(x_box[i],1),np.around(y_box[i],1))))

bool_df=[]
for i in range(len(df_ibtracs)):
    print(i)
    lon,lat=df_ibtracs['USA_LON'][i],df_ibtracs['USA_LAT'][i]
    to_find = [np.around(lon+360,1),np.around(lat,1)]
    if to_find in list_xy:
        bool_df.append(True)
    else:
        bool_df.append(False)

df_ibtracs=df_ibtracs[bool_df]

ids=df_ibtracs['SID'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibtracs.loc[df_ibtracs['SID']==storm_id]
    df_storm['groups'] = (df_storm.ISO_TIME.diff().dt.seconds > 3600).cumsum()
    for unique_group in df_storm['groups'].unique():
        df_storm_unique = df_storm.loc[df_storm['groups']==unique_group]
        if df_storm_unique.USA_WIND.max()>18:
            if len(df_storm_unique)>71:
                if unique_group>0:
                    df_storm_unique['SID']=df_storm['SID'].iloc[0]+str(unique_group)
                dfs.append(df_storm_unique)

df_ibtracs = pd.concat(dfs).reset_index(drop=True)

ids=df_ibtracs['SID'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibtracs.loc[df_ibtracs['SID']==storm_id]
    #plt.scatter(df_storm.USA_LON.values,df_storm.USA_LAT.values)
    plt.plot(df_storm.USA_LON.values,df_storm.USA_LAT.values,color='green')

plt.show()

#df_ibtracs.to_pickle('df_ibtracs.pkl')

#save in between, to make sure that I dont need to run everything again each time

#df_keep = df_keep.apply(pd.to_numeric, errors='coerce')
#########################
#0.2 read knmi track data#
#########################

#year=2017
#scenario=ref_folder

#tracks_file=xr.open_dataset('tracks.nc').drop(['lev']).squeeze()
#tracks_file = tracks_file.rename({'x':'lon','y':'lat'})
#cdo.remapbic('mygrid.ascii',input=tracks_file,output='tracks_file_interpolated_remapbic.nc')#r144x73
#tracks_file_interpolated=xr.open_dataset('tracks_file_interpolated_remapbic.nc')

###########################################
#1. interpolate from polar to regular grid#
###########################################

def interpolate_pressure(year,scenario):
    mslp_file_name=glob.glob(scenario+'/mslp*'+str(year)+'*')[0]
    mslp_file=xr.open_dataset(mslp_file_name).drop(['dir','block1','block2','time_bnds','dtg','date_bnds','hms_bnds','assigned','rotated_pole','height']).squeeze()
    cdo.remapbic('mygrid.ascii',input=mslp_file,output=ref_folder_out+'mslp.KNMI-%d.LCARIB12.wCY33-v556-JJASON-REF-fERA5.1H.nc'%(year))#/gpfs/work1/0/ESLRP/eucp_knmi/data/RACMO/LCARIB12/wCY33-v556-JJASON-REF-fERA5/1Hourly_data_regridded/mslp_file_interpolated_remapbic_%d.nc'%(year))#r144x73

def interpolate_wind(year,scenario):
    w10m_file_name=glob.glob(scenario+'/w10m*'+str(year)+'*')[0]
    w10m_file=xr.open_dataset(w10m_file_name).drop(['dir','block1','block2','time_bnds','dtg','date_bnds','hms_bnds','assigned','rotated_pole','height']).squeeze()
    cdo.remapbic('mygrid.ascii',input=w10m_file,output=ref_folder_out+'w10m.KNMI-%d.LCARIB12.wCY33-v556-JJASON-REF-fERA5.1H.nc'%(year))#r144x73

#scenario=ref_folder
#for i in range(2019,2021):
#    print(i)
#    interpolate_pressure(i,scenario)
#    interpolate_wind(i,scenario)

###############################################################
#2.1 keep data where mslp at least 17 hpa below annual maximum#
#2.2 keep data where w10m at least 15 m/s                     #
###############################################################
def tracking_pressure(year,scenario):
    mslp_file_interpolated=xr.open_dataset(glob.glob(scenario+'mslp.KNMI-%d*.nc'%(year))[0])
    pressure_diff=mslp_file_interpolated-mslp_file_interpolated.max(dim='time')
    mslp_file_interpolated_diff_17=mslp_file_interpolated.where(pressure_diff.mslp < -1700, drop=False)
    mslp_file_interpolated_diff_17.to_netcdf('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/conditional_mslp_diff_17/mslp_diff_17_year_%d_scenario_%s.nc'%(year,scenario_naming))
    
    return mslp_file_interpolated_diff_17

def tracking_wind_maxima(year,scenario):
    w10m_file_interpolated=xr.open_dataset(glob.glob(scenario+'w10m.KNMI-%d*.nc'%(year))[0])
    w10m_file_interpolated_min_15=w10m_file_interpolated.where(w10m_file_interpolated > 15, drop=False)
    w10m_file_interpolated_min_15.to_netcdf('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/conditional_w10m_min_15/w10m_min_15_year_%d_scenario_%s.nc'%(year,scenario_naming))
    
    return w10m_file_interpolated_min_15

#scenario=ref_folder_out
#scenario_naming='ref'
#for i in range(1979,2021):
#    print(i)
#    tracking_pressure_minima(i,scenario)
#    tracking_wind_maxima(i,scenario)

#############################################
#3.1 find local pressure minima per timestep#
#3.2 search for winds > 15 m/s within 300 km#
#3.3 if found set 'potential_cells' to True #
#############################################

radius= 6371
def tracking_potential_cells(year,scenario):
    mslp_file_interpolated_diff_17 = xr.open_dataset('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/conditional_mslp_diff_17/mslp_diff_17_year_%d_scenario_%s.nc'%(year,scenario))#mslp_file_interpolated_diff_17.copy(deep=False)
    w10m_file_interpolated_min_15=xr.open_dataset('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/conditional_w10m_min_15/w10m_min_15_year_%d_scenario_%s.nc'%(year,scenario))
    potential_cells = mslp_file_interpolated_diff_17.copy(deep=True)
    potential_cells['boolean']=potential_cells.mslp<0
    potential_cells = potential_cells.drop('mslp')
    for i in range(len(mslp_file_interpolated_diff_17.time.values)):            #loop over time slices
        print(i)
        if mslp_file_interpolated_diff_17.isel(time=[i]).max().mslp>0:          #if the pressure drop exceeds 17 hpa relative to yearly max for at least one cell at a time slice
            field = mslp_file_interpolated_diff_17.isel(time=[i]).squeeze()     #select the respective time slice pressure field
            field_diff_17 = field.where(field.mslp>0,drop=True)                 #keep part of field (square) where pressure drop > 17 hpa relative to yearly max
            data = field_diff_17.mslp.values                                    #here we make use of the scipy function called minimum_filter
            minima = (data == minimum_filter(data, 3, mode='constant',cval=0))  #it can detect pressure minima in 2D pressure fields
            xx,yy=np.meshgrid(field_diff_17.lon.values,field_diff_17.lat.values)#we create a meshgrid from the pressure minima lons and lats
            lon_minima=xx[minima]                                               #make a 1d array of longitude pressure minima's
            lat_minima=yy[minima]                                               #make a 1d array of latitude pressure minima's
            for j,k in zip(range(len(lon_minima)),range(len(lon_minima))):      #loop over lon/lat's
                point_diff_17 = field_diff_17.sel(lon=lon_minima[j],lat=lat_minima[k]) #select a coordinate (goal?: compute distance to nearest cell of which wind speed > 15 m/s, if distance < 300 km, keep pressure cell as potential track cell)
                if point_diff_17.mslp.values>0:                                 #if the pressure drop exceeds 17 hpa relative to yearly max
                    lon_pressure = point_diff_17.lon.values                     #lon
                    lat_pressure = point_diff_17.lat.values                     #lat
                    field_w10m = w10m_file_interpolated_min_15.isel(time=[i]).squeeze() #select the respective time slice wind field
                    field_w10m_4_by_4_box = field_w10m.sel(lon=slice((lon_pressure-4),(lon_pressure+4)),lat=slice((lat_pressure-4),(lat_pressure+4))) #select box from wind field of 8 by 8 degrees around pressure coordinate
                    field_w10m_15 = field_w10m_4_by_4_box.where(field_w10m_4_by_4_box.w10m>15,drop=True) #keep part of box where wind speed > 15 m/s
                    xx,yy=np.meshgrid(field_w10m_15.lon.values,field_w10m_15.lat.values) #create lon lat meshgrid 
                    lon_wind = xx[field_w10m_15.w10m.values>15]                 #create list of lon's
                    lat_wind = yy[field_w10m_15.w10m.values>15]                 #create list of lat's
                    dist_single=[]
                    if len(lon_wind)>0:
                        for m in range(len(lon_wind)):                          #loop over coordinates & calculate distance (km)
                            latR1,latR2,lonR1,lonR2 = radians(lat_pressure),radians(lat_wind[m]),radians(lon_pressure),radians(lon_wind[m])
                            dlon = lonR2 - lonR1
                            dlat = latR2 - latR1
                            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                            if a>1:
                                a=1
                            c = 2 * atan2(sqrt(a), sqrt(1-a))
                            dist_single.append(radius * c)
                            if min(dist_single)<300:                            #when a wind speed cell (> 15 m/s) is found within 300 km of the pressure coordinate, the pressure coordinate becomes true, meaning it is a potential tc cell
                                break
                        if min(dist_single)<300:
                            potential_cells.loc[dict(time=potential_cells.time[i].values,lon=lon_pressure,lat=lat_pressure)]=True #condition is true or false, true if air pressure drop > 17hpa and wind speed exceeds 15 m/s within 300 km from pressure cell
    
    potential_cells.to_netcdf('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/potential_cells/potential_cells_year_%d_scenario_%s.nc'%(year,scenario))
    
    return potential_cells

#for year in range(2015,2021):
#    print(year)
#    for scenario in ['ref','tp2','pgw']:
#        print(scenario)
#        tracking_potential_cells(year,scenario)

###########################################################################################
#It can happen that suddenly a TC has two local minima's instead of one.                  #
#In that case we only want to keep the local minima with the lowest pressure of the two.  #
#4.1 check if multiple coordinates have value 'True' at a certain timestep                #
#4.2 if yes, compute distances between these coordinates and order them by lowest pressure#
#4.3 set cell with lowest pressure to -1, and ignore lon/lat that are 'True' within 300 km#
#4.4 keep doing this until no coordinates are left. For this we have a while loop         #
###########################################################################################

radius= 6371
def potential_cells_min_distance_apart(year,scenario):
    potential_cell_ID=-1 #instead of 'True' for all potential_cells we use value -1 to indicate potential_cells_min_distance_apart
    mslp_file_interpolated_diff_17 = xr.open_dataset('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/conditional_mslp_diff_17/mslp_diff_17_year_%d_scenario_%s.nc'%(year,scenario)) #open file with mslp data (mslp at least 17 hpa below annual maximum)
    w10m_file_interpolated_min_15=xr.open_dataset('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/conditional_w10m_min_15/w10m_min_15_year_%d_scenario_%s.nc'%(year,scenario)) #open file with w10m data (w10m at least 15 m/s)
    cell_number = mslp_file_interpolated_diff_17.copy(deep=True)                         #create new variable used for putting -1 values (= potential cell at least 300 km apart)
    cell_number = cell_number.where(cell_number.mslp<0,np.nan)                           #set all values to nan
    potential_cells = xr.open_dataset('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/potential_cells/potential_cells_year_%d_scenario_%s.nc'%(year,scenario)) #open file with potential tc cells
    mslp_potential_cells = mslp_file_interpolated_diff_17.where(potential_cells.boolean) #copy values of mslp file to new variable where potential_cells==True
    for i in range(len(mslp_potential_cells.time.values)):                               #loop over hourly timesteps
        #print(i)
        if mslp_potential_cells.isel(time=[i]).max().mslp>0:                             #check if any lon/lat at respective timestep is a potential_cell
            print(i)
            field = mslp_potential_cells.isel(time=[i]).squeeze()                        #select mslp field at timestep i
            field_select = field.where(field.mslp>0,drop=True)                           #keep field where it has a value (so indicating it is potential cell)
            inds_of_min=field_select.mslp.argmin(dim=['lon','lat'])                      #index of minimum pressure
            lon_min,lat_min = field_select.lon[inds_of_min['lon']].values, field_select.lat[inds_of_min['lat']].values #lon/lat of min pressure
            xx,yy=np.meshgrid(field_select.lon.values,field_select.lat.values)           #meshgrid of all lons/lats
            lons=xx[field_select.mslp.values>0]                                          #1d array of all lons
            lats=yy[field_select.mslp.values>0]                                          #1d array of all lats
            mslps=field_select.mslp.values[field_select.mslp.values>0]                   #1d array of corresponding mslp values
            dist_single=[]                                                               #calculate distance between lon/lat min pressure and other lons/lats
            if len(lons)>0:
                for m in range(len(lons)):                                               #loop over other minima coordinates & calculate distance (km) to field minimum lon/lat
                    latR1,latR2,lonR1,lonR2 = radians(lat_min),radians(lats[m]),radians(lon_min),radians(lons[m])
                    dlon = lonR2 - lonR1
                    dlat = latR2 - latR1
                    a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                    if a>1:
                        a=1
                    
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    dist_single.append(radius * c)
                if max(dist_single)<300:                                                 #if distance to other pressure minima smaller than 300 km, other pressure minima must be same TC, so we just give 1 track_ID number to one cell at this timestep, no other TCs simultaneously active
                    cell_number.loc[dict(time=potential_cells.time[i].values,lon=lon_min,lat=lat_min)]=potential_cell_ID
                else:                                                                    #but what if there are pressure minima outside a 300 km range from the field pressure minima lon/lat?
                    cell_number.loc[dict(time=potential_cells.time[i].values,lon=lon_min,lat=lat_min)]=potential_cell_ID
                    while max(dist_single)>300:
                        print('yes!!!!!!    timestep = ',i)
                        mslps_300_not_sorted=mslps[np.asarray(dist_single)>=300]         #do same again, for lons/lats outside the 300 km range, keep repeating until no lons/lats left at this timestep
                        arr1inds=mslps_300_not_sorted.argsort()                          #arrange leftover mslp values by values, starting with lowest value
                        mslps_300=mslps_300_not_sorted[arr1inds]
                        lons_300=lons[np.asarray(dist_single)>=300][arr1inds]
                        lats_300=lats[np.asarray(dist_single)>=300][arr1inds]
                        dist_outside_300=[]
                        for lon,lat in zip(lons_300,lats_300):
                            lon_1=lons_300[0]
                            lat_1=lats_300[0]
                            latR1,latR2,lonR1,lonR2 = radians(lat_1),radians(lat),radians(lon_1),radians(lon)
                            dlon = lonR2 - lonR1
                            dlat = latR2 - latR1
                            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                            if a>1:
                                a=1
                            c = 2 * atan2(sqrt(a), sqrt(1-a))
                            dist_outside_300.append(radius * c)
                        cell_number.loc[dict(time=potential_cells.time[i].values,lon=lons_300[0],lat=lats_300[0])]=potential_cell_ID #set minimum pressure lon/lat to -1, ignore lons/lats within 300 km assume same TC, if still lons/lats outside 300 km repeat 'while'
                        dist_single=dist_outside_300
                        mslps=mslps_300
                        lons=lons_300
                        lats=lats_300
                        if max(dist_single)<300:
                            break
    
    cell_number.to_netcdf('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/potential_cells_min_distance_apart/potential_cells_min_distance_apart_year_%d_scenario_%s.nc'%(year,scenario))
    
    return cell_number

#for year in range(1979,2021):
#    print(year)
#    for scenario in ['ref','tp2','pgw']:
#        print(scenario)
#        potential_cells_min_distance_apart(year,scenario)

#############################################################################################################
#5 here we combine the cells to make a track from them                                                      #
#5.1 first we make a list of timesteps that are active 72 hours, otherwise TC track is too short.           #
#5.2 loop over timesteps and give same track_ID to cells within 300 km from active cell in previous timestep#
#5.3 otherwise track_ID becomes track_ID+1                                                                  #
#############################################################################################################

def make_tracks_from_cells(year,scenario):
    cell_number=xr.open_dataset('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/potential_cells_min_distance_apart/potential_cells_min_distance_apart_year_%d_scenario_%s.nc'%(year,scenario)) #open file with cells
    cell_number['cells']=cell_number['mslp']
    cell_number=cell_number.drop('mslp')
    track_number=cell_number.copy(deep=True)                                            #make a copy of the 'cell' file
    track_number=track_number.where(track_number.cells<-2,np.nan)
    active_timesteps=[]
    for i in range(len(cell_number.time.values)):
        if cell_number.isel(time=[i]).max().cells<0:
            active_timesteps.append(i)
    
    active_timesteps_splitted=[list(map(itemgetter(1), g)) for k, g in groupby(enumerate(active_timesteps), lambda x: x[0]-x[1])]
    active_timesteps_72h_min=[]
    for j in active_timesteps_splitted:
        if len(j)>71:
            active_timesteps_72h_min.append(np.asarray(j))
    
    if active_timesteps_72h_min:
        active_timesteps_72h_min=np.concatenate(active_timesteps_72h_min,axis=0)
        for i in range(len(active_timesteps_72h_min)):
            if i == 0:
                track_ID=1
                field = cell_number.isel(time=active_timesteps_72h_min[i]).squeeze()
                point = field.where(field.cells.notnull(),drop=True)
                for j in range(np.nansum(field.cells<0)):
                    track_number.loc[dict(time=track_number.time.values[active_timesteps_72h_min[i]],lon=point.lon.values[j],lat=point.lat.values[j])]=track_ID+j
                    lon_tmin1_list=list(point.lon.values)
                    lat_tmin1_list=list(point.lat.values)
            else:
                field = cell_number.isel(time=active_timesteps_72h_min[i]).squeeze()
                point = field.where(field.cells.notnull(),drop=True)
                point_xx,point_yy = np.meshgrid(point.lon.values,point.lat.values)
                point_lon = point_xx[point.cells.values<0]
                point_lat = point_yy[point.cells.values<0]
                for j in range(np.nansum(point.cells<0)):
                    if np.nansum(point.cells<0)>1:
                        print(i, j)
                    lon_tnow=point_lon[j]
                    lat_tnow=point_lat[j]
                    lon_tmin1=lon_tmin1_list
                    lat_tmin1=lat_tmin1_list
                    track_distance=[]
                    radius= 6371
                    for lon,lat in zip(lon_tmin1,lat_tmin1):
                        latR1,latR2,lonR1,lonR2 = radians(lat_tnow),radians(lat),radians(lon_tnow),radians(lon)
                        dlon = lonR2 - lonR1
                        dlat = latR2 - latR1
                        a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                        if a>1:
                            a=1
                        c = 2 * atan2(sqrt(a), sqrt(1-a))
                        track_distance.append(radius * c)
                    if (min(track_distance)<300) & (active_timesteps_72h_min[i]-active_timesteps_72h_min[i-1]==1):
                        track_ID=int(track_number.loc[dict(time=track_number.time.values[active_timesteps_72h_min[i-1]],lon=lon_tmin1[np.argmin(track_distance)],lat=lat_tmin1[np.argmin(track_distance)])].cells.values)
                        track_number.loc[dict(time=track_number.time.values[active_timesteps_72h_min[i]],lon=lon_tnow,lat=lat_tnow)]=track_ID
                    else:
                        track_ID=int(track_number.max().cells.values)+1
                        track_number.loc[dict(time=track_number.time.values[active_timesteps_72h_min[i]],lon=lon_tnow,lat=lat_tnow)]=track_ID
            
                lon_tmin1_list=point_lon
                lat_tmin1_list=point_lat
        
        track_number.to_netcdf('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_number/track_number_year_%d_scenario_%s.nc'%(year,scenario))
    
    return track_number

for year in range(1979,2021):
    print(year)
    for scenario in ['ref','tp2','pgw']:
        abc=make_tracks_from_cells(year,scenario)

####################################################################
#6.  Now it is time to gather all data corresponding to the tracks #
#6.1 Loop over track numbers per year and make df of lon/lats/time #
#6.2 Open original mslp and w10m file to extract corresponding data#
####################################################################

def extract_track_data(year,scenario,scenario_naming):
    my_file = Path('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_number/track_number_year_%d_scenario_%s.nc'%(year,scenario))
    if my_file.is_file():
        track_number=xr.open_dataset('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_number/track_number_year_%d_scenario_%s.nc'%(year,scenario)) #open file with cells
        mslp_file_interpolated=xr.open_dataset(glob.glob(scenario_naming+'mslp.KNMI-%d*.nc'%(year))[0])
        w10m_file_interpolated=xr.open_dataset(glob.glob(scenario_naming+'w10m.KNMI-%d*.nc'%(year))[0])
        lons_tracks=[]
        lats_tracks=[]
        time_tracks=[]
        id_tracks=[]
        mslp_tracks=[]
        w10m_tracks=[]
        new_track_id=1
        year_tracks=[]
        for number in np.unique(track_number.cells.values)[:-1]:
            print(number)
            lons_track=[]
            lats_track=[]
            time_track=[]
            id_track=[]
            mslp_track=[]
            w10m_track=[]
            year_track=[]
            track_field=track_number.where(track_number.cells==number,drop=True)
            if len(track_field.time.values)>71:
                for t in range(len(track_field.time)):
                    track_field_t = track_field.isel(time=[t],drop=True)
                    track_point_t = track_field_t.where(track_field_t.cells==number,drop=True)
                    if len(track_point_t.lon.values)>1 or len(track_point_t.lat.values)>1:
                        #could check here which point is closest to xy of previous track lon/lat point
                        previous_lon=lons_track[-1]
                        previous_lat=lats_track[-1]
                        lons_now=track_point_t.lon.values
                        lats_now=track_point_t.lat.values
                        distance=[]
                        radius= 6371
                        for lon,lat in zip(lons_now,lats_now):
                            latR1,latR2,lonR1,lonR2 = radians(previous_lat),radians(lat),radians(previous_lon),radians(lon)
                            dlon = lonR2 - lonR1
                            dlat = latR2 - latR1
                            a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                            if a>1:
                                a=1
                            
                            c = 2 * atan2(sqrt(a), sqrt(1-a))
                            distance.append(radius * c)
                        track_point_t = track_point_t.loc[dict(lon=lons_now[np.argmin(np.asarray(distance))],lat=lats_now[np.argmin(np.asarray(distance))])]
                        print('mistake! track_ID in timestep twice.')
                        print('track_ID = '+str(number))
                        print('t = '+str(t))
                    lons_track.append(float(track_point_t.lon.values))
                    lats_track.append(float(track_point_t.lat.values))
                    time_track.append(track_point_t.time.values[0])
                    id_track.append(str(year)+'_'+str(new_track_id))
                    mslp_track.append(float(mslp_file_interpolated.loc[dict(time=track_point_t.time.values[0],lon=float(track_point_t.lon.values),lat=float(track_point_t.lat.values))].mslp.values))
                    #w10m_field=w10m_file_interpolated.sel(time=track_point_t.time.values[0],lon=slice((float(track_point_t.lon.values)-5),(float(track_point_t.lon.values)+5)),lat=slice((float(track_point_t.lat.values)-5),(float(track_point_t.lat.values)+5)))
                    #lons=w10m_field.lon.values
                    #lats=w10m_field.lat.values
                    #xx,yy=np.meshgrid(lons,lats)
                    #distance=[]
                    #radius= 6371
                    #for lon,lat in zip(xx.flatten(),yy.flatten()):
                    #    latR1,latR2,lonR1,lonR2 = radians(track_point_t.lat.values),radians(lat),radians(track_point_t.lon.values),radians(lon)
                    #    dlon = lonR2 - lonR1
                    #    dlat = latR2 - latR1
                    #    a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                    #    if a>1:
                    #        a=1
                    #    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    #    distance.append(radius * c)
                    #
                    #w10m_track.append(np.nanmax(w10m_field.w10m.values.flatten()[np.asarray(distance)<500]))
                    w10m_track.append(float(w10m_file_interpolated.sel(time=track_point_t.time.values[0],lon=slice((float(track_point_t.lon.values)-4),(float(track_point_t.lon.values)+4)),lat=slice((float(track_point_t.lat.values)-4),(float(track_point_t.lat.values)+4))).max().w10m.values))
                    year_track.append(int(track_point_t.time.values[0].astype('datetime64[Y]').astype(int) + 1970))
                if len(lons_track)>71:
                    lons_tracks.append(lons_track)
                    lats_tracks.append(lats_track)
                    time_tracks.append(time_track)
                    id_tracks.append(id_track)
                    mslp_tracks.append(mslp_track)
                    w10m_tracks.append(w10m_track)
                    year_tracks.append(year_track)
                    new_track_id=new_track_id+1
        
        lons_tracks=[item for sublist in lons_tracks for item in sublist]
        lats_tracks=[item for sublist in lats_tracks for item in sublist]
        time_tracks=[item for sublist in time_tracks for item in sublist]
        id_tracks=[item for sublist in id_tracks for item in sublist]
        mslp_tracks=[item for sublist in mslp_tracks for item in sublist]
        w10m_tracks=[item for sublist in w10m_tracks for item in sublist]
        year_tracks=[item for sublist in year_tracks for item in sublist]
        
        df = pd.DataFrame(list(zip(id_tracks,year_tracks,time_tracks,lons_tracks,lats_tracks,mslp_tracks,w10m_tracks)),columns=['id','year','time','lon','lat','mslp_min','w10m_max'])
        df.to_pickle('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/track_data_year_%d_scenario_%s.pkl'%(year,scenario))

for year in range(1979,2021):
    print(year)
    scenario='ref'
    scenario_naming=ref_folder_out
    extract_track_data(year,scenario,scenario_naming)
    #for scenario,scenario_naming in zip(['ref','tp2','pgw'],[ref_folder_out,tp2_folder_out,pgw_folder_out]):
    #    print(scenario)
    #    print(scenario_naming)
    #    extract_track_data(year,scenario,scenario_naming)

for year in range(2008,2021):
    print(year)
    scenario='tp2'
    scenario_naming=tp2_folder_out
    extract_track_data(year,scenario,scenario_naming)

for year in range(1985,2021):
    print(year)
    scenario='pgw'
    scenario_naming=pgw_folder_out
    extract_track_data(year,scenario,scenario_naming)

def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_tracks(year):
    my_file_ref = Path('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/track_data_year_%d_scenario_ref.pkl'%(year))
    my_file_tp2 = Path('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/track_data_year_%d_scenario_tp2.pkl'%(year))
    my_file_pgw = Path('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/track_data_year_%d_scenario_pgw.pkl'%(year))
    if my_file_ref.is_file():
        df_ref = pd.read_pickle('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/track_data_year_%d_scenario_ref.pkl'%(year))
        plt.scatter(df_ref.lon.values,df_ref.lat.values,color='red')
    
    if my_file_tp2.is_file():
        df_tp2 = pd.read_pickle('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/track_data_year_%d_scenario_tp2.pkl'%(year))
        plt.scatter(df_tp2.lon.values,df_tp2.lat.values,color='blue')
    
    if my_file_pgw.is_file():
        df_pgw = pd.read_pickle('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/track_data_year_%d_scenario_pgw.pkl'%(year))
        plt.scatter(df_pgw.lon.values,df_pgw.lat.values,color='green')
    
    plt.savefig('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_figures/tracks_year_%d_72hr.png'%(year),dpi=600)
    plt.close()

for year in range(1980,2021):
    print(year)
    plot_tracks(year)

#keep all tracks
df_ref = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*ref.pkl'))).reset_index()
df_tp2 = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*tp2.pkl'))).reset_index()
df_pgw = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*pgw.pkl'))).reset_index()

#keep tracks that exceed wind speed of 18 m/s at least once during lifetime
df_ref_ids=[]
for ref_file in glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*ref.pkl'):
    df = pd.read_pickle(ref_file)
    for unique_id in df.id.unique():
        df_select = df.loc[df['id'] == unique_id]
        if df_select.w10m_max.max()>18:
            df_ref_ids.append(unique_id)

df_tp2_ids=[]
for tp2_file in glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*tp2.pkl'):
    df = pd.read_pickle(tp2_file)
    for unique_id in df.id.unique():
        df_select = df.loc[df['id'] == unique_id]
        if df_select.w10m_max.max()>18:
            df_tp2_ids.append(unique_id)

df_pgw_ids=[]
for pgw_file in glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*pgw.pkl'):
    df = pd.read_pickle(pgw_file)
    for unique_id in df.id.unique():
        df_select = df.loc[df['id'] == unique_id]
        if df_select.w10m_max.max()>18:
            df_pgw_ids.append(unique_id)

df_ref = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*ref.pkl'))).reset_index()
df_ref = df_ref.loc[df_ref['id'].isin(df_ref_ids)].reset_index(drop=True)
df_tp2 = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*tp2.pkl'))).reset_index()
df_tp2 = df_tp2.loc[df_tp2['id'].isin(df_tp2_ids)].reset_index(drop=True)
df_pgw = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*pgw.pkl'))).reset_index()
df_pgw = df_pgw.loc[df_pgw['id'].isin(df_pgw_ids)].reset_index(drop=True)

#do this per simulation, remove xy to close to border of model grid
#1. RACMO-REF
bool_df=[]
for i in range(len(df_ref)):
    print(i)
    lon,lat=df_ref['lon'][i],df_ref['lat'][i]
    to_find = [np.around(lon,1),np.around(lat,1)]
    if to_find in list_xy:
        bool_df.append(True)
    else:
        bool_df.append(False)

df_ref=df_ref[bool_df]

ids=df_ref['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_ref.loc[df_ref['id']==storm_id]
    df_storm['groups'] = (df_storm.time.diff().dt.seconds > 3600).cumsum()
    for unique_group in df_storm['groups'].unique():
        df_storm_unique = df_storm.loc[df_storm['groups']==unique_group]
        if df_storm_unique.w10m_max.max()>18:
            if len(df_storm_unique)>71:
                if unique_group>0:
                    df_storm_unique['id']=df_storm['id'].iloc[0]+str(unique_group)
                dfs.append(df_storm_unique)

df_ref = pd.concat(dfs).reset_index(drop=True)
#2. RACMO-TP2
bool_df=[]
for i in range(len(df_tp2)):
    print(i)
    lon,lat=df_tp2['lon'][i],df_tp2['lat'][i]
    to_find = [np.around(lon,1),np.around(lat,1)]
    if to_find in list_xy:
        bool_df.append(True)
    else:
        bool_df.append(False)

df_tp2=df_tp2[bool_df]

ids=df_tp2['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_tp2.loc[df_tp2['id']==storm_id]
    df_storm['groups'] = (df_storm.time.diff().dt.seconds > 3600).cumsum()
    for unique_group in df_storm['groups'].unique():
        df_storm_unique = df_storm.loc[df_storm['groups']==unique_group]
        if df_storm_unique.w10m_max.max()>18:
            if len(df_storm_unique)>71:
                if unique_group>0:
                    df_storm_unique['id']=df_storm['id'].iloc[0]+str(unique_group)
                dfs.append(df_storm_unique)

df_tp2 = pd.concat(dfs).reset_index(drop=True)
#3. RACMO-PGW
bool_df=[]
for i in range(len(df_pgw)):
    print(i)
    lon,lat=df_pgw['lon'][i],df_pgw['lat'][i]
    to_find = [np.around(lon,1),np.around(lat,1)]
    if to_find in list_xy:
        bool_df.append(True)
    else:
        bool_df.append(False)

df_pgw=df_pgw[bool_df]

ids=df_pgw['id'].unique()
dfs=[]
for storm_id in ids:
    print(storm_id)
    df_storm = df_pgw.loc[df_pgw['id']==storm_id]
    df_storm['groups'] = (df_storm.time.diff().dt.seconds > 3600).cumsum()
    for unique_group in df_storm['groups'].unique():
        df_storm_unique = df_storm.loc[df_storm['groups']==unique_group]
        if df_storm_unique.w10m_max.max()>18:
            if len(df_storm_unique)>71:
                if unique_group>0:
                    df_storm_unique['id']=df_storm['id'].iloc[0]+str(unique_group)
                dfs.append(df_storm_unique)

df_pgw = pd.concat(dfs).reset_index(drop=True)

df_ref.to_pickle('df_ref.pkl')
df_tp2.to_pickle('df_tp2.pkl')
df_pgw.to_pickle('df_pgw.pkl')

df_ref=pd.read_pickle('df_ref.pkl')
df_tp2=pd.read_pickle('df_tp2.pkl')
df_pgw=pd.read_pickle('df_pgw.pkl')
df_ibtracs=pd.read_pickle('df_ibtracs.pkl')

ids=df_ibtracs['SID'].unique()
for storm_id in ids:
    print(storm_id)
    df_storm = df_ibtracs.loc[df_ibtracs['SID']==storm_id]
    plt.plot(df_storm.USA_LON.values,df_storm.USA_LAT.values,color='green')

plt.savefig('df_ibtracs.png',dpi=600)
plt.close()


#
plt.close()
df = pd.DataFrame({'simulation':['ref','tp2','pgw','ibt'],'frequency':[len(df_ref),len(df_tp2),len(df_pgw),len(df_ibtracs)]})
df.plot.bar(x='simulation', y='frequency', rot=0)
plt.title('frequency (active hours)')
plt.savefig('frequency_timesteps_TS+.png',dpi=600)
plt.close()

#some first plotting trials
#keep tracks that exceed wind speed of 58 m/s at least once during lifetime (CAT 4 minimum)
df_ref_ids=[]
for ref_file in glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*ref.pkl'):
    df = pd.read_pickle(ref_file)
    for unique_id in df.id.unique():
        df_select = df.loc[df['id'] == unique_id]
        if df_select.w10m_max.max()>58:
            df_ref_ids.append(unique_id)

df_tp2_ids=[]
for tp2_file in glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*tp2.pkl'):
    df = pd.read_pickle(tp2_file)
    for unique_id in df.id.unique():
        df_select = df.loc[df['id'] == unique_id]
        if df_select.w10m_max.max()>58:
            df_tp2_ids.append(unique_id)

df_pgw_ids=[]
for pgw_file in glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*pgw.pkl'):
    df = pd.read_pickle(pgw_file)
    for unique_id in df.id.unique():
        df_select = df.loc[df['id'] == unique_id]
        if df_select.w10m_max.max()>58:
            df_pgw_ids.append(unique_id)

df_ibt_ids=[]
for storm_id in df_ibtracs.SID.unique():
    df = df_ibtracs.loc[df_ibtracs['SID'] == storm_id]
    if df.USA_WIND.max()>58:
        df_ibt_ids.append(storm_id)

df_ref = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*ref.pkl'))).reset_index()
df_ref = df_ref.loc[df_ref['id'].isin(df_ref_ids)]
df_tp2 = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*tp2.pkl'))).reset_index()
df_tp2 = df_tp2.loc[df_tp2['id'].isin(df_tp2_ids)]
df_pgw = pd.concat(map(pd.read_pickle,glob.glob('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_data/*pgw.pkl'))).reset_index()
df_pgw = df_pgw.loc[df_pgw['id'].isin(df_pgw_ids)]
df_ibt = df_ibtracs.loc[df_ibtracs['SID'].isin(df_ibt_ids)]

plt.close()
df = pd.DataFrame({'simulation':['ref','tp2','pgw','ibt'],'frequency':[len(df_ref),len(df_tp2),len(df_pgw),len(df_ibt)]})
df.plot.bar(x='simulation', y='frequency', rot=0)
plt.title('frequency (active hours CAT4+ tracks)')
plt.savefig('frequency_timesteps_CAT3+_tracks.png',dpi=600)
plt.close()

#relative frequency wind speed per cyclone category saffir simpson scale
bars_ref = [np.nansum(df_ref.w10m_max<=17)/len(df_ref),np.nansum((df_ref.w10m_max>17)&(df_ref.w10m_max<=32))/len(df_ref),np.nansum((df_ref.w10m_max>32)&(df_ref.w10m_max<=43))/len(df_ref),np.nansum((df_ref.w10m_max>43)&(df_ref.w10m_max<=50))/len(df_ref),np.nansum((df_ref.w10m_max>50)&(df_ref.w10m_max<=58))/len(df_ref),np.nansum((df_ref.w10m_max>58)&(df_ref.w10m_max<=70))/len(df_ref),np.nansum(df_ref.w10m_max>70)/len(df_ref)]
bars_tp2 = [np.nansum(df_tp2.w10m_max<=17)/len(df_tp2),np.nansum((df_tp2.w10m_max>17)&(df_tp2.w10m_max<=32))/len(df_tp2),np.nansum((df_tp2.w10m_max>32)&(df_tp2.w10m_max<=43))/len(df_tp2),np.nansum((df_tp2.w10m_max>43)&(df_tp2.w10m_max<=50))/len(df_tp2),np.nansum((df_tp2.w10m_max>50)&(df_tp2.w10m_max<=58))/len(df_tp2),np.nansum((df_tp2.w10m_max>58)&(df_tp2.w10m_max<=70))/len(df_tp2),np.nansum(df_tp2.w10m_max>70)/len(df_tp2)]
bars_pgw = [np.nansum(df_pgw.w10m_max<=17)/len(df_pgw),np.nansum((df_pgw.w10m_max>17)&(df_pgw.w10m_max<=32))/len(df_pgw),np.nansum((df_pgw.w10m_max>32)&(df_pgw.w10m_max<=43))/len(df_pgw),np.nansum((df_pgw.w10m_max>43)&(df_pgw.w10m_max<=50))/len(df_pgw),np.nansum((df_pgw.w10m_max>50)&(df_pgw.w10m_max<=58))/len(df_pgw),np.nansum((df_pgw.w10m_max>58)&(df_pgw.w10m_max<=70))/len(df_pgw),np.nansum(df_pgw.w10m_max>70)/len(df_pgw)]
bars_ibt = [np.nansum(df_ibtracs.USA_WIND<=17)/len(df_ibtracs),np.nansum((df_ibtracs.USA_WIND>17)&(df_ibtracs.USA_WIND<=32))/len(df_ibtracs),np.nansum((df_ibtracs.USA_WIND>32)&(df_ibtracs.USA_WIND<=43))/len(df_ibtracs),np.nansum((df_ibtracs.USA_WIND>43)&(df_ibtracs.USA_WIND<=50))/len(df_ibtracs),np.nansum((df_ibtracs.USA_WIND>50)&(df_ibtracs.USA_WIND<=58))/len(df_ibtracs),np.nansum((df_ibtracs.USA_WIND>58)&(df_ibtracs.USA_WIND<=70))/len(df_ibtracs),np.nansum(df_ibtracs.USA_WIND>70)/len(df_ibtracs)]

barWidth=0.2
r1 = np.arange(len(bars_ref))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, bars_ref, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='ref')
plt.bar(r2, bars_tp2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='tp2')
plt.bar(r3, bars_pgw, width = barWidth, color = 'red', edgecolor = 'black', capsize=7, label='pgw')
plt.bar(r4, bars_ibt, width = barWidth, color = 'pink', edgecolor = 'black', capsize=7, label='ibt')

plt.xticks([r + barWidth for r in range(len(bars_ref))],['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'])
plt.ylabel('relative frequency')
plt.title('relative frequency w10m max')
plt.legend()
plt.savefig('relative_frequency_w10m_max_CAT4+.png',dpi=600)
plt.close()

#relative frequency mslp
bars_ref = [np.nansum(df_ref.mslp_min<90000)/len(df_ref),np.nansum((df_ref.mslp_min>90000)&(df_ref.mslp_min<=92000))/len(df_ref),np.nansum((df_ref.mslp_min>92000)&(df_ref.mslp_min<=94000))/len(df_ref),np.nansum((df_ref.mslp_min>94000)&(df_ref.mslp_min<=96000))/len(df_ref),np.nansum((df_ref.mslp_min>96000)&(df_ref.mslp_min<=98000))/len(df_ref),np.nansum((df_ref.mslp_min>98000)&(df_ref.mslp_min<=100000))/len(df_ref),np.nansum(df_ref.mslp_min>=100000)/len(df_ref)]
bars_tp2 = [np.nansum(df_tp2.mslp_min<90000)/len(df_tp2),np.nansum((df_tp2.mslp_min>90000)&(df_tp2.mslp_min<=92000))/len(df_tp2),np.nansum((df_tp2.mslp_min>92000)&(df_tp2.mslp_min<=94000))/len(df_tp2),np.nansum((df_tp2.mslp_min>94000)&(df_tp2.mslp_min<=96000))/len(df_tp2),np.nansum((df_tp2.mslp_min>96000)&(df_tp2.mslp_min<=98000))/len(df_tp2),np.nansum((df_tp2.mslp_min>98000)&(df_tp2.mslp_min<=100000))/len(df_tp2),np.nansum(df_tp2.mslp_min>=100000)/len(df_tp2)]
bars_pgw = [np.nansum(df_pgw.mslp_min<90000)/len(df_pgw),np.nansum((df_pgw.mslp_min>90000)&(df_pgw.mslp_min<=92000))/len(df_pgw),np.nansum((df_pgw.mslp_min>92000)&(df_pgw.mslp_min<=94000))/len(df_pgw),np.nansum((df_pgw.mslp_min>94000)&(df_pgw.mslp_min<=96000))/len(df_pgw),np.nansum((df_pgw.mslp_min>96000)&(df_pgw.mslp_min<=98000))/len(df_pgw),np.nansum((df_pgw.mslp_min>98000)&(df_pgw.mslp_min<=100000))/len(df_pgw),np.nansum(df_pgw.mslp_min>=100000)/len(df_pgw)]
bars_ibt = [np.nansum(df_ibtracs.USA_PRES<900)/len(df_ibtracs),np.nansum((df_ibtracs.USA_PRES>900)&(df_ibtracs.USA_PRES<=920))/len(df_ibtracs),np.nansum((df_ibtracs.USA_PRES>920)&(df_ibtracs.USA_PRES<=940))/len(df_ibtracs),np.nansum((df_ibtracs.USA_PRES>940)&(df_ibtracs.USA_PRES<=960))/len(df_ibtracs),np.nansum((df_ibtracs.USA_PRES>960)&(df_ibtracs.USA_PRES<=980))/len(df_ibtracs),np.nansum((df_ibtracs.USA_PRES>980)&(df_ibtracs.USA_PRES<=1000))/len(df_ibtracs),np.nansum(df_ibtracs.USA_PRES>=1000)/len(df_ibtracs)]

barWidth=0.2
r1 = np.arange(len(bars_ref))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, bars_ref, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='ref')
plt.bar(r2, bars_tp2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='tp2')
plt.bar(r3, bars_pgw, width = barWidth, color = 'red', edgecolor = 'black', capsize=7, label='pgw')
plt.bar(r4, bars_ibt, width = barWidth, color = 'pink', edgecolor = 'black', capsize=7, label='ibt')

plt.xticks([r + barWidth for r in range(len(bars_ref))],['<900','900-920','920-940','940-960','960-980','980-1000','1000>'])
plt.ylabel('relative frequency')
plt.title('relative frequency mslp min (hPa)')
plt.legend()
plt.savefig('relative_frequency_mslp_min_CAT4+.png',dpi=600)
plt.close()

#relative frequency wind speed and mslp per track instead of per timestep
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
for storm_id in df_ibtracs.SID.unique():
    storm_df=df_ibtracs.loc[df_ibtracs['SID'] == storm_id]
    ibt_w10ms.append(storm_df['USA_WIND'].max())
    ibt_mslps.append(storm_df['USA_PRES'].min())

#frequency tracks
plt.close()
df = pd.DataFrame({'simulation':['ref','tp2','pgw','ibt'],'frequency':[len(ref_w10ms),len(tp2_w10ms),len(pgw_w10ms),len(ibt_w10ms)]})
df.plot.bar(x='simulation', y='frequency', rot=0)
plt.title('frequency (number of tracks)')
plt.savefig('frequency_tracks_CAT4+.png',dpi=600)
plt.close()

#wind speed per track
bars_ref = [np.nansum(np.asarray(ref_w10ms)<=17)/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>17)&(np.asarray(ref_w10ms)<=32))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>32)&(np.asarray(ref_w10ms)<=43))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>43)&(np.asarray(ref_w10ms)<=50))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>50)&(np.asarray(ref_w10ms)<=58))/len(ref_w10ms),np.nansum((np.asarray(ref_w10ms)>58)&(np.asarray(ref_w10ms)<=70))/len(ref_w10ms),np.nansum(np.asarray(ref_w10ms)>70)/len(ref_w10ms)]
bars_tp2 = [np.nansum(np.asarray(tp2_w10ms)<=17)/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>17)&(np.asarray(tp2_w10ms)<=32))/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>32)&(np.asarray(tp2_w10ms)<=43))/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>43)&(np.asarray(tp2_w10ms)<=50))/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>50)&(np.asarray(tp2_w10ms)<=58))/len(tp2_w10ms),np.nansum((np.asarray(tp2_w10ms)>58)&(np.asarray(tp2_w10ms)<=70))/len(tp2_w10ms),np.nansum(np.asarray(tp2_w10ms)>70)/len(tp2_w10ms)]
bars_pgw = [np.nansum(np.asarray(pgw_w10ms)<=17)/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>17)&(np.asarray(pgw_w10ms)<=32))/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>32)&(np.asarray(pgw_w10ms)<=43))/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>43)&(np.asarray(pgw_w10ms)<=50))/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>50)&(np.asarray(pgw_w10ms)<=58))/len(pgw_w10ms),np.nansum((np.asarray(pgw_w10ms)>58)&(np.asarray(pgw_w10ms)<=70))/len(pgw_w10ms),np.nansum(np.asarray(pgw_w10ms)>70)/len(pgw_w10ms)]
bars_ibt = [np.nansum(np.asarray(ibt_w10ms)<=17)/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>17)&(np.asarray(ibt_w10ms)<=32))/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>32)&(np.asarray(ibt_w10ms)<=43))/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>43)&(np.asarray(ibt_w10ms)<=50))/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>50)&(np.asarray(ibt_w10ms)<=58))/len(ibt_w10ms),np.nansum((np.asarray(ibt_w10ms)>58)&(np.asarray(ibt_w10ms)<=70))/len(ibt_w10ms),np.nansum(np.asarray(ibt_w10ms)>70)/len(ibt_w10ms)]

barWidth=0.2
r1 = np.arange(len(bars_ref))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, bars_ref, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='ref')
plt.bar(r2, bars_tp2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='tp2')
plt.bar(r3, bars_pgw, width = barWidth, color = 'red', edgecolor = 'black', capsize=7, label='pgw')
plt.bar(r4, bars_ibt, width = barWidth, color = 'pink', edgecolor = 'black', capsize=7, label='ibt')

plt.xticks([r + barWidth for r in range(len(bars_ref))],['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'])
plt.ylabel('relative frequency')
plt.title('relative frequency w10m max per track CAT4+')
plt.legend()
plt.savefig('relative_frequency_w10m_max_per_track_CAT4+.png',dpi=600)
plt.close()

#pressure per track
bars_ref = [np.nansum(np.asarray(ref_mslps)<90000)/len(ref_mslps),np.nansum((np.asarray(ref_mslps)>90000)&(np.asarray(ref_mslps)<=92000))/len(ref_mslps),np.nansum((np.asarray(ref_mslps)>92000)&(np.asarray(ref_mslps)<=94000))/len(ref_mslps),np.nansum((np.asarray(ref_mslps)>94000)&(np.asarray(ref_mslps)<=96000))/len(ref_mslps),np.nansum((np.asarray(ref_mslps)>96000)&(np.asarray(ref_mslps)<=98000))/len(ref_mslps),np.nansum((np.asarray(ref_mslps)>98000)&(np.asarray(ref_mslps)<=100000))/len(ref_mslps),np.nansum(np.asarray(ref_mslps)>=100000)/len(ref_mslps)]
bars_tp2 = [np.nansum(np.asarray(tp2_mslps)<90000)/len(tp2_mslps),np.nansum((np.asarray(tp2_mslps)>90000)&(np.asarray(tp2_mslps)<=92000))/len(tp2_mslps),np.nansum((np.asarray(tp2_mslps)>92000)&(np.asarray(tp2_mslps)<=94000))/len(tp2_mslps),np.nansum((np.asarray(tp2_mslps)>94000)&(np.asarray(tp2_mslps)<=96000))/len(tp2_mslps),np.nansum((np.asarray(tp2_mslps)>96000)&(np.asarray(tp2_mslps)<=98000))/len(tp2_mslps),np.nansum((np.asarray(tp2_mslps)>98000)&(np.asarray(tp2_mslps)<=100000))/len(tp2_mslps),np.nansum(np.asarray(tp2_mslps)>=100000)/len(tp2_mslps)]
bars_pgw = [np.nansum(np.asarray(pgw_mslps)<90000)/len(pgw_mslps),np.nansum((np.asarray(pgw_mslps)>90000)&(np.asarray(pgw_mslps)<=92000))/len(pgw_mslps),np.nansum((np.asarray(pgw_mslps)>92000)&(np.asarray(pgw_mslps)<=94000))/len(pgw_mslps),np.nansum((np.asarray(pgw_mslps)>94000)&(np.asarray(pgw_mslps)<=96000))/len(pgw_mslps),np.nansum((np.asarray(pgw_mslps)>96000)&(np.asarray(pgw_mslps)<=98000))/len(pgw_mslps),np.nansum((np.asarray(pgw_mslps)>98000)&(np.asarray(pgw_mslps)<=100000))/len(pgw_mslps),np.nansum(np.asarray(pgw_mslps)>=100000)/len(pgw_mslps)]
bars_ibt = [np.nansum(np.asarray(ibt_mslps)<900)/len(ibt_mslps),np.nansum((np.asarray(ibt_mslps)>900)&(np.asarray(ibt_mslps)<=920))/len(ibt_mslps),np.nansum((np.asarray(ibt_mslps)>920)&(np.asarray(ibt_mslps)<=940))/len(ibt_mslps),np.nansum((np.asarray(ibt_mslps)>940)&(np.asarray(ibt_mslps)<=960))/len(ibt_mslps),np.nansum((np.asarray(ibt_mslps)>960)&(np.asarray(ibt_mslps)<=980))/len(ibt_mslps),np.nansum((np.asarray(ibt_mslps)>980)&(np.asarray(ibt_mslps)<=1000))/len(ibt_mslps),np.nansum(np.asarray(ibt_mslps)>=1000)/len(ibt_mslps)]

barWidth=0.2
r1 = np.arange(len(bars_ref))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, bars_ref, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='ref')
plt.bar(r2, bars_tp2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='tp2')
plt.bar(r3, bars_pgw, width = barWidth, color = 'red', edgecolor = 'black', capsize=7, label='pgw')
plt.bar(r4, bars_ibt, width = barWidth, color = 'pink', edgecolor = 'black', capsize=7, label='ibt')

plt.xticks([r + barWidth for r in range(len(bars_ref))],['<900','900-920','920-940','940-960','960-980','980-1000','1000>'])
plt.ylabel('relative frequency')
plt.title('relative frequency mslp min (hPa) per track CAT4+')
plt.legend()
plt.savefig('relative_frequency_mslp_min_per_track_CAT4+.png',dpi=600)
plt.close()

#################################
df = pd.DataFrame({'simulation':['ref','tp2','pgw'],'

combine_select_mslp=pd.concat([df_ref.mslp_min/100,df_tp2.mslp_min/100,df_pgw.mslp_min/100],axis=1)
combine_select_mslp.columns=['ref','tp2','pgw']
combine_select_mslp.boxplot()
plt.title('mslp boxplot all timesteps 1979-2020')
plt.xlabel('simulation')
plt.ylabel('hPa')
plt.savefig('mslp_boxplot_1979_2020.png',dpi=600)
plt.close()

combine_select_w10m=pd.concat([df_ref.w10m_max,df_tp2.w10m_max,df_pgw.w10m_max],axis=1)
combine_select_w10m.columns=['ref','tp2','pgw']
combine_select_w10m.boxplot()
plt.title('w10m boxplot all timesteps 1979-2020')
plt.xlabel('simulation')
plt.ylabel('wind speed (m/s)')
plt.savefig('w10m_boxplot_1979_2020.png',dpi=600)
plt.close()
    
#cmap = get_cmap(len(lons_tracks))
cmap = get_cmap(len(np.unique(np.hstack(time_tracks))))

for i in range(len(lons_tracks)):
    if len(lons_tracks[i])>71:
        #mslp_track=[]
        #for j in range(len(time_tracks[i])):
        #    mslp_track.append(float(mslp_file_interpolated.loc[dict(time=time_tracks[i][j],lon=lons_tracks[i][j],lat=lats_tracks[i][j])].mslp.values))
        #plt.scatter(lons_tracks[i],lats_tracks[i],c=np.asarray(mslp_track))
        #plt.scatter(lons_tracks[i],lats_tracks[i],color=cmap(i))
        plt.scatter(lons_tracks[i],lats_tracks[i],c=time_tracks[i])
        len(lons_tracks[i])

plt.savefig('/gpfs/work1/0/ESLRP/eucp_knmi/analysis/track_figures/tracks_job_year_%d_scenario_%s_72hrxx.png'%(year,scenario),dpi=600)
plt.close()

###check how many cells fulfill pressure criteria
#len(mslp_file_interpolated_diff_17.mslp.values.flatten())-np.nansum(np.isnan(mslp_file_interpolated_diff_17.mslp.values)) = 276k

###check how many cells fulfill windspeed criteria
#len(w10m_file_interpolated_min_15.w10m.values.flatten())-np.nansum(np.isnan(w10m_file_interpolated_min_15.w10m.values)) = 256k

###check how many cells not nan after interpolating
#4392*251*551-np.nansum(np.isnan(w10m_file_interpolated.w10m.values)) = 454M

###plot difference in pressure to check
#lon,lat=np.meshgrid(pressure_diff.lon.values,pressure_diff.lat.values)
#lon=lon-360
#plt.scatter(lon,lat,c=pressure_diff.mslp.values[1420,:,:]/100)
#plt.xlim(-90,-35)
#plt.ylim(5,30)
#plt.colorbar()
#plt.savefig('mslp_diff_max_vs_t1420.png')
#plt.close()



#old condition script, too slow
'''
radius= 6371
def tracking_cells(year,scenario):
    mslp_file_interpolated_diff_17 = tracking_pressure_minima(year,scenario)    #interpolate pressure fields
    w10m_file_interpolated_min_15  = tracking_wind_maxima(year,scenario)        #interpolate wind fields
    
    condition = mslp_file_interpolated_diff_17.copy(deep=False)
    condition['boolean']=condition.mslp<0
    condition = condition.drop('mslp')
    for i in range(len(mslp_file_interpolated_diff_17.time.values)):            #loop over time slices
        print(i)
        if mslp_file_interpolated_diff_17.isel(time=[i]).max().mslp>0:          #if the pressure drop exceeds 17 hpa relative to yearly max for at least one cell at a time slice
            field = mslp_file_interpolated_diff_17.isel(time=[i]).squeeze()     #select the respective time slice pressure field
            field_diff_17 = field.where(field.mslp>0,drop=True)                 #keep part of field (square) where pressure drop > 17 hpa relative to yearly max
            for j in range(len(field_diff_17.lon.values)):                      #loop over lon's
                for k in range(len(field_diff_17.lat.values)):                  #loop over lat's
                    point_diff_17 = field_diff_17.isel(lon=[j],lat=[k])         #select a coordinate (goal?: compute distance to nearest cell of which wind speed > 15 m/s, if distance < 300 km, keep pressure cell as potential track cell)
                    if point_diff_17.mslp.values>0:                             #if the pressure drop exceeds 17 hpa relative to yearly max
                        lon_pressure = point_diff_17.lon.values                 #lon
                        lat_pressure = point_diff_17.lat.values                 #lat
                        field_w10m = w10m_file_interpolated_min_15.isel(time=[i]).squeeze() #select the respective time slice wind field
                        field_w10m_4_by_4_box = field_w10m.sel(lon=slice((lon_pressure-4)[0],(lon_pressure+4)[0]),lat=slice((lat_pressure-4)[0],(lat_pressure+4)[0])) #select box from wind field of 8 by 8 degrees around pressure coordinate
                        field_w10m_15 = field_w10m_4_by_4_box.where(field_w10m_4_by_4_box.w10m>15,drop=True) #keep part of box where wind speed > 15 m/s
                        xx,yy=np.meshgrid(field_w10m_15.lon.values,field_w10m_15.lat.values) #create lon lat meshgrid 
                        lon_wind = xx[field_w10m_15.w10m.values>15]             #create list of lon's
                        lat_wind = yy[field_w10m_15.w10m.values>15]             #create list of lat's
                        dist_single=[]
                        if len(lon_wind)>0:
                            for m in range(len(lon_wind)):                      #loop over coordinates & calculate distance (km)
                                latR1,latR2,lonR1,lonR2 = radians(lat_pressure),radians(lat_wind[m]),radians(lon_pressure),radians(lon_wind[m])
                                dlon = lonR2 - lonR1
                                dlat = latR2 - latR1
                                a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
                                if a>1:
                                    a=1
                                c = 2 * atan2(sqrt(a), sqrt(1-a))
                                dist_single.append(radius * c)
                                if min(dist_single)<300:                        #when a wind speed cell (> 15 m/s) is found within 300 km of the pressure coordinate, the pressure coordinate becomes true, meaning it is a potential tc cell
                                    break
                            if min(dist_single)<300:
                                condition.loc[dict(time=condition.time[i].values,lon=lon_pressure,lat=lat_pressure)]=True
    
    condition.to_netcdf('condition_year_%s_scenario_ref.nc'%(year))

    return condition
'''





#script for computing km distances based on lon/lat values = very fast
latR1,latR2,lonR1,lonR2 = radians(19.8),radians(21.8),radians(277.2),radians(279.3)
dlon = lonR2 - lonR1
dlat = latR2 - latR1
a = sin(dlat / 2)**2 + cos(latR1) * cos(latR2) * sin(dlon / 2)**2
if a>1:
    a=1

c = 2 * atan2(sqrt(a), sqrt(1-a))
distance.append(radius * c)

