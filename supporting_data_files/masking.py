import numpy as np
import xarray as xr
import cartopy.crs as ccrs

def access_land_map():
    '''
    Maps land in the ACCESS-OM2-01 bathymetry.
    '''
    ht = xr.open_dataset('/work/Ruth.Moorman/Bathymetry/ACCESSOM201_bathymetry.nc')
    ht = ht.ht
    land_map = (ht*0).fillna(1)
    return land_map
