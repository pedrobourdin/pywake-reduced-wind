import py_wake as pw
import numpy as np
import xarray as xr
import dask.distributed
import matplotlib.pyplot as plt
import pandas as pd
import xesmf as xe
from joblib import Parallel, delayed
import glob
import dask
import os
import socket
from sys import argv
#from memory_profiler import profile


from py_wake.site import XRSite
from py_wake.site.shear import PowerShear
from numpy import newaxis as na
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.superposition_models import LinearSum
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site.distance import StraightDistance
from py_wake.utils.numpy_utils import Numpy32

#function to get x and y coordinates of all wind turbines
def windfarm(P_t,P,D,latitude,longitude): 
    latituderef=59.35
    longituderef=20
    r=6362e3
    d=3*D  #distance between turbines -->3D or 5D or 7D
    n=np.ceil(P_t/P)
    dim=int(np.floor(np.sqrt(n))) #dimension of the square 
    diff=n-dim**2 #remaining 
    #turbines placement (square)
    #creating a dim*dim grid
    x0=np.arange(-dim*d/2,dim*d/2,d)
    y0=np.arange(-dim*d/2,dim*d/2,d)
    X0,Y0=np.meshgrid(x0,y0)
    
    #getting coordinates of each point
    wt_x=np.ravel(X0)
    wt_y=np.ravel(Y0)
    
    #condition to add remaining turbines, if dim**2 (number of turbines in the square) is not equal to n 
    if diff!=0:
    #coordinates of remaining turbines, added in a last row 
        if diff<=dim:
            ysup=np.linspace(-dim*d/2,d*(dim/2-1),int(diff))
            xsup=np.ones(len(ysup))*((dim*d/2))
    #coordinates of all of turbines  
            wt_x=np.concatenate([wt_x,xsup])
            wt_y=np.concatenate([wt_y,ysup])
        else:
            xsup0=np.arange(-dim*d/2,dim*d/2,d)
            ysup0=np.ones(len(xsup0))*((dim*d/2))
            
            ysup1=np.linspace(-dim*d/2,ysup0[0],int(diff-dim))
            xsup1=np.ones(len(ysup1))*((dim*d/2))
            
            wt_x=np.concatenate([wt_x,xsup0,xsup1])
            wt_y=np.concatenate([wt_y,ysup0,ysup1])
    
    wt_x=(wt_x+np.radians(longitude-longituderef)*r*np.cos(np.radians(latitude))).astype(np.float32)
    wt_y=(wt_y+np.radians(latitude-latituderef)*r).astype(np.float32)
    
    return wt_x,wt_y

def wake(data,wt_x,wt_y,windTurbines,tt):
    #opening CERRA data from Copernicus
    data=data.sel(longitude=slice(9.7,24.65),latitude=slice(65.68,54.1))  #extremum coordinates of the wind farms
    data=data.rename_dims({'longitude':'i','latitude':'j'})
                        
    #code regrid début
    latituderef=59.35
    longituderef=20
    r=6362e3
    
    y=(np.radians(data.latitude-latituderef)*r).drop('latitude')
    x=(np.radians(data.longitude-longituderef)*r*np.cos(np.radians(data.latitude.isel(j=-1).values))).drop('longitude') 

    longitude=(longituderef+np.degrees(x.squeeze()/(r*np.cos(np.radians(latituderef)+y.squeeze()/r))))
    latitude=(xr.ones_like(longitude)*(latituderef+np.degrees(y.squeeze()/r)))

    ds_out = xr.Dataset(coords={'longitude':longitude,'latitude':latitude})
    regridder1 = xe.Regridder(data, ds_out, "bilinear",weights='regridder1',reuse_weights=True)
#    regridder1.to_netcdf('/scratch/project_2002251/pedro/NEMO-reduced-wind/PYWAKE_submission/regridder1')
    grid= pw.HorizontalGrid(x=x[:-2],y=y[:-2],h=h)
        

    ws=regridder1(data.ws).assign_coords({'x':x,'y':y}) \
        .sel(heightAboveGround=h,method='nearest').swap_dims({'i':'x','j':'y'}).drop(['longitude','latitude']).compute()
    wd=regridder1(data.wdir).assign_coords({'x':x,'y':y}) \
    .sel(heightAboveGround=h,method='nearest').swap_dims({'i':'x','j':'y'}).drop(['longitude','latitude']).compute()
    
    ds=xr.Dataset(data_vars={'WS':ws.astype(np.float32),'WD':wd.astype(np.float32),'P':xr.ones_like(ws).astype(np.float32),'TI':xr.ones_like(ws).astype(np.float32)*0.05},coords={'y':ws.y,'x':ws.x}).load()
#       ds=ds.drop('longitude').drop('latitude')

#If shear is needed, add shear=PowerShear(h_ref=150, alpha=.2)
    site=XRSite(ds,distance=StraightDistance(wind_direction='WD_i'))
    
    wf_model = PropagateDownwind(site, windTurbines,
                                     wake_deficitModel=BastankhahGaussianDeficit(use_effective_ws=True))
    print('wind farm model starting')
    #with Numpy32():
    wf=wf_model(wt_x, wt_y,ws=10,wd=10,type=types).flow_map(grid).drop(['wd','ws'])
    print('actual calculatitudeion is done')
    
    #final regrid       

    longitude=longituderef+np.degrees(wf.x/(r*np.cos(np.radians(latituderef)+wf.y/r)))
    latitude=xr.ones_like(longitude)*(latituderef+wf.y/r*180/np.pi)
    ds_in=xr.Dataset().assign_coords({'latitude':latitude,'longitude':longitude})
    ds_out = xr.Dataset(coords={'longitude':data.longitude,'latitude':data.latitude})
    regridder2 = xe.Regridder(ds_in, ds_out, "bilinear",weights='regridder2',reuse_weights=True)
#    regridder2.to_netcdf('/scratch/project_2002251/pedro/NEMO-reduced-wind/PYWAKE_submission/regridder2')

    print(tt)
    wind=xr.Dataset(data_vars={'ws':regridder2(wf.WS_eff.squeeze()),'wd':regridder2(wf.WD.squeeze())})
    wind=wind.swap_dims({'j':'latitude','i':'longitude'})
    #wind.to_netcdf(f'/scratch/project_2002251/pedro/NEMO-reduced-wind/PYWAKE_submission/reduced_wind/reduced_{str(tt).zfill(3)}_'+fname.split('/')[-1])
    
    wind.to_netcdf(f'/scratch/project_2010748/pywake_forcing/reduced_wind/squared_{str(tt).zfill(3)}_'+fname.split('/')[-1])
    print('regridding and to netcdf')
    data.close()
    del wf, wf_model
    
if __name__ == '__main__':
    fname=argv[1]
    tt=int(argv[2])
    data=xr.open_dataset(fname,chunks={'time':1})
    print(data.time.size,tt)
    # if os.path.exists(f'/scratch/project_2002251/pedro/NEMO-reduced-wind/PYWAKE_submission/reduced_wind/reduced_{str(tt).zfill(3)}_'+fname.split('/')[-1]):
    #     pass
    # else:
    if data.time.size>tt:
        local_dir = '/scratch/project_2002251/pedro/' #change this to some folder you have access to
        if not os.path.isdir(local_dir):
            os.system('mkdir -p '+local_dir)
            print('created folder '+local_dir)
            #
        n_workers = 1 #this number should be the same (or smaller) than the CPUs you have access to
        n_threads = 2
        processes = True
        cluster = dask.distributed.LocalCluster(n_workers=n_workers,
                                                threads_per_worker=n_threads,
                                                processes=processes,
                                                local_directory=local_dir,
                                                lifetime='4 hour',
                                                lifetime_stagger='10 minutes',
                                                lifetime_restart=True,
                                                dashboard_address=None,
                                                worker_dashboard_address=None)
        client  = dask.distributed.Client(cluster)
        # rest of your code comes here
        df = pd.read_csv('windfarms_baltic.csv',delimiter=';')
        power=df['power_mw'].values
        latitude=df['lat'].values
        longitude=df['lon'].values

        #NREL 15MW wind turbine
        nrelturbine_carac = pd.read_csv('NREL_reference_15MW.csv',delimiter=',')
        power_nrel=nrelturbine_carac['Power[kW]'].values
        ct=nrelturbine_carac['Cp[-]'].values
        u=nrelturbine_carac['Wind Speed[m/s]'].values
        h=150
        D=240
        P=15e6
        dtu10mw = DTU10MW()
        NREL15MW = WindTurbine(name='NREL15MW',
                            diameter=240,
                            hub_height=h,
                            powerCtFunction=PowerCtTabular(u,power_nrel,'kW',ct))

        windTurbines = WindTurbines.from_WindTurbine_lst([NREL15MW])

        for i in range(len(df)):
            if i==0:
                wt_x,wt_y=windfarm(power[i]*1e6,P,D,latitude[i],longitude[i])
            else:
                wt_x0,wt_y0=windfarm(power[i]*1e6,P,D,latitude[i],longitude[i])
                wt_x,wt_y=np.r_[wt_x,wt_x0],np.r_[wt_y,wt_y0]
        types=[0]*len(wt_x)

        n_jobs=1 #same as your CPUs for the job
        wake(data.isel(time=tt),wt_x,wt_y,windTurbines,tt)
