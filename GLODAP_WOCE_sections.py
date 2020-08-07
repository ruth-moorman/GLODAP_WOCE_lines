import xarray as xr
import numpy as np

# whole bunch of functions for extracting GLODAP bottle data along some significant Southern Ocean WOCE lines.
# for examples of usabe see GLODAP_WOCE_sections_examples.ipynb


def IO6S_GLODAP_profile(variable, unique_cruise, year , so_glodap):
    """    
    Extracts GLODAP bottle data collected along the IO6S WOCE line.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns IO6S_var_linear, IO6S_var_linear_horizontal, var_IO6S, bathymetry_mask,IO6S_latitude, IO6S_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '2008', '1996', or '1993'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    IO6S_var_linear = vertcially but not horizontally interpolated
    IO6S_var_linear_horizontal = vertically and horizontally interpolated
    var_IO6S = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    IO6S_latitude, IO6S_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-70,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    var_IO6S1 = so_var.where(so_var.cruise == unique_cruise[87], drop = True)
    var_IO6S1 = var_IO6S1.where(var_IO6S1.longitude<31, drop = True)
    var_IO6S2 = so_var.where(so_var.cruise == unique_cruise[92], drop = True)
    var_IO6S3 = so_var.where(so_var.cruise == unique_cruise[93], drop = True)   
    var_IO6S3 = var_IO6S3[:-48]

    if year == '2008':
        var_IO6S = var_IO6S1
    elif year == '1993':
        var_IO6S = var_IO6S2
    elif year == '1996':
        var_IO6S = var_IO6S3    
    else:
        return print('year = 2008, 1993, 1996')
    
    # extract bathymetry information - use all available crossings
    IO6S_all =  xr.concat([var_IO6S1, var_IO6S2, var_IO6S3], dim = 'SAMPLE')
    IO6S_all_bathymetry = xr.DataArray(IO6S_all.bottomdepth, coords = [IO6S_all.latitude], dims = 'latitude')    
    IO6S_all_bathymetry = IO6S_all_bathymetry.sortby('latitude')
    IO6S_all_bathymetry = IO6S_all_bathymetry.groupby('latitude').max()
    IO6S_all_bathymetry = IO6S_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')
    
    
    IO6S_latitude = var_IO6S.latitude.values
    IO6S_longitude = var_IO6S.longitude.values
    IO6S_lat_index = np.sort(np.unique(IO6S_latitude, return_index = True)[1])
    IO6S_lat_index = np.append(IO6S_lat_index,len(IO6S_latitude))
    IO6S_lat_dimension = IO6S_latitude[IO6S_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(IO6S_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    IO6S_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [IO6S_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    IO6S_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_IO6S.where(var_IO6S.latitude == IO6S_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        IO6S_var_linear[i,:] = prof_var_linear
    IO6S_var_linear = IO6S_var_linear.sortby('latitude')
    
    # horizontal interpolation
    IO6S_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    IO6S_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = IO6S_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 3:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            IO6S_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = IO6S_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<IO6S_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a
        
    return IO6S_var_linear, IO6S_var_linear_horizontal, var_IO6S, bathymetry_mask, IO6S_latitude, IO6S_longitude


def P16A_GLODAP_profile(variable, unique_cruise, year, so_glodap,extend = True):
    """    
    Extracts GLODAP bottle data collected along the P16S WOCE line.
    If extend = True is specified, these lines are combined with a 2011 extension that continues south onto the continental shelf.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns P16A_var_linear, P16A_var_linear_horizontal, var_P16A, bathymetry_mask,P16A_latitude, P16A_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '2005', '2014'
    extend = option to extend WOCE profile onto Antarctic continental shelf (but with data from 2011), default is True
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    P16A_var_linear = vertcially but not horizontally interpolated
    P16A_var_linear_horizontal = vertically and horizontally interpolated
    var_P16A = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    P16A_latitude, P16A_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-77,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    var_P16A1 = so_var.where(so_var.cruise == unique_cruise[85], drop = True)
    var_P16A1 = var_P16A1.where(var_P16A1.longitude <-149, drop = True).where(var_P16A1.longitude >-151, drop = True)
    var_P16A2 = so_var.where(so_var.cruise == unique_cruise[147], drop = True)
    var_P16A2 = var_P16A2.where(var_P16A2.longitude <-149, drop = True).where(var_P16A2.longitude >-151, drop = True)
    var_P16A_extend = so_var.where(so_var.cruise == unique_cruise[71], drop = True)
    var_P16A_extend = var_P16A_extend.where(var_P16A_extend.longitude <-149, drop = True).where(var_P16A_extend.longitude >-151, drop = True)

    if year == '2005':
        var_P16A = var_P16A1
    elif year == '2014':
        var_P16A = var_P16A2
    else:
        return print('year = 2005, 2014')
    if extend == True:
        var_P16A = xr.concat([var_P16A, var_P16A_extend], dim = 'SAMPLE')
    
    
    # extract bathymetry information - use all available crossings
    P16A_all =  xr.concat([var_P16A1, var_P16A2, var_P16A_extend], dim = 'SAMPLE')
    P16A_all_bathymetry = xr.DataArray(P16A_all.bottomdepth, coords = [P16A_all.latitude], dims = 'latitude')    
    P16A_all_bathymetry = P16A_all_bathymetry.sortby('latitude')
    P16A_all_bathymetry = P16A_all_bathymetry.groupby('latitude').max()
    P16A_all_bathymetry = P16A_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')

    
    P16A_latitude = var_P16A.latitude.values
    P16A_longitude = var_P16A.longitude.values
    P16A_lat_index = np.sort(np.unique(P16A_latitude, return_index = True)[1])
    P16A_lat_index = np.append(P16A_lat_index,len(P16A_latitude))
    P16A_lat_dimension = P16A_latitude[P16A_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(P16A_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    P16A_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [P16A_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    P16A_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_P16A.where(var_P16A.latitude == P16A_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        P16A_var_linear[i,:] = prof_var_linear
    
    P16A_var_linear = P16A_var_linear.sortby('latitude')
    
    # horizontal interpolation
    P16A_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    P16A_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = P16A_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 3:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            P16A_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = P16A_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<P16A_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a

    return P16A_var_linear, P16A_var_linear_horizontal, var_P16A, bathymetry_mask, P16A_latitude, P16A_longitude

def SO4A_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    
    """    
    Extracts GLODAP bottle data collected along the SO4A WOCE line.
    Bottle data along the cruise line are compressed into longitude/depth space (ignoring small variations in latitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine longitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns SO4A_var_linear, SO4A_var_linear_horizontal, var_SO4A, bathymetry_mask,SO4A_longitude, SO4A_latitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year  '1989', '1990', '1992', '1996', '1998', '2005', '2008', '2010'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    SO4A_var_linear = vertcially but not horizontally interpolated
    SO4A_var_linear_horizontal = vertically and horizontally interpolated
    var_SO4A = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    SO4A_latitude, SO4A_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_longitude = np.arange(-57,-10,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))

    so_var = so_glodap.sel(VARIABLE = variable)
    var_SO4A1 = so_var.where(so_var.cruise == unique_cruise[1], drop = True)
    var_SO4A1 = var_SO4A1.where(var_SO4A1.longitude <-10, drop = True).where(var_SO4A1.longitude >-58, drop = True)
    var_SO4A2 = so_var.where(so_var.cruise == unique_cruise[2], drop = True)
    var_SO4A2 = var_SO4A2.where(var_SO4A2.longitude <-10, drop = True)
    var_SO4A2 = xr.concat([var_SO4A2[:645], var_SO4A2[950:]], dim = 'SAMPLE')
    var_SO4A3 = so_var.where(so_var.cruise == unique_cruise[5], drop = True)
    var_SO4A3 = var_SO4A3[:1437]
    var_SO4A3 = var_SO4A3.where(var_SO4A3.longitude <-10, drop = True)
    var_SO4A4 = so_var.where(so_var.cruise == unique_cruise[6], drop = True)
    var_SO4A4 = var_SO4A4.where(var_SO4A4.longitude <-10, drop = True)
    var_SO4A4 = var_SO4A4[:642]
    var_SO4A5 = so_var.where(so_var.cruise == unique_cruise[7], drop = True)
    var_SO4A5 = var_SO4A5.where(var_SO4A5.longitude <-10, drop = True).where(var_SO4A5.longitude >-58, drop = True).where(var_SO4A5.latitude <-63, drop = True)
    var_SO4A6 = so_var.where(so_var.cruise == unique_cruise[9], drop = True)
    var_SO4A6 = var_SO4A6.where(var_SO4A6.longitude <-10, drop = True).where(var_SO4A6.longitude >-58, drop = True).where(var_SO4A6.latitude <-63, drop = True)
    var_SO4A7 = so_var.where(so_var.cruise == unique_cruise[12], drop = True)
    var_SO4A7 = var_SO4A7.where(var_SO4A7.longitude <-10, drop = True).where(var_SO4A7.longitude >-58, drop = True).where(var_SO4A7.latitude <-63, drop = True)
    var_SO4A8 = so_var.where(so_var.cruise == unique_cruise[13], drop = True)
    var_SO4A8 = var_SO4A8.where(var_SO4A8.longitude <-10, drop = True)
        
    if year == '1989':
        var_SO4A = var_SO4A1
    elif year == '1990':
        var_SO4A = var_SO4A2
    elif year == '1992':
        var_SO4A = var_SO4A3 
    elif year == '1996':
        var_SO4A = var_SO4A4
    elif year == '1998':
        var_SO4A = var_SO4A5
    elif year == '2005':
        var_SO4A = var_SO4A6 
    elif year == '2008':
        var_SO4A = var_SO4A7
    elif year == '2010':
        var_SO4A = var_SO4A8
    else:
        return print('year = 1989,1990,1992,1996,1998,2005,2008,2010')
    
    # extract bathymetry information - use all available crossings
    SO4A_all =  xr.concat([var_SO4A2, var_SO4A3, var_SO4A4, var_SO4A5, var_SO4A7, var_SO4A8], dim = 'SAMPLE')
    SO4A_all_bathymetry = xr.DataArray(SO4A_all.bottomdepth, coords = [SO4A_all.longitude], dims = 'longitude')    
    SO4A_all_bathymetry = SO4A_all_bathymetry.sortby('longitude')
    SO4A_all_bathymetry = SO4A_all_bathymetry.groupby('longitude').max()
    SO4A_all_bathymetry = SO4A_all_bathymetry.interp(longitude = interp_longitude, method = 'linear')
    
    SO4A_longitude = var_SO4A.longitude.values
    SO4A_latitude = var_SO4A.latitude.values
    SO4A_lon_index = np.sort(np.unique(SO4A_longitude, return_index = True)[1])
    SO4A_lon_index = np.append(SO4A_lon_index,len(SO4A_longitude))
    SO4A_lon_dimension = SO4A_longitude[SO4A_lon_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(SO4A_lon_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    SO4A_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [SO4A_lon_dimension, interp_depth], dims = ['longitude', 'depth'])
    SO4A_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lon_var = var_SO4A.where(var_SO4A.longitude == SO4A_lon_dimension[i], drop = True)
        lon_var = lon_var.sortby(lon_var.depth)
        prof_xarray = xr.DataArray(lon_var.values, coords = [lon_var.depth.values], dims = 'depth')
        if len(np.unique(lon_var.depth.values)) != len(lon_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        SO4A_var_linear[i,:] = prof_var_linear
    SO4A_var_linear = SO4A_var_linear.sortby('longitude')
    
    # horizontal interpolation
    SO4A_var_linear_horizontal = xr.DataArray(np.empty((len(interp_longitude),len(interp_depth))), coords = [interp_longitude, interp_depth], dims = ['longitude', 'depth'])
    SO4A_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = SO4A_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 3:
            depth_var = depth_var.interp(longitude = interp_longitude, method = 'linear')
            SO4A_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = SO4A_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.longitude)):
        a = bathymetry_mask.isel(longitude = i)
        a = a.where(a.depth<SO4A_all_bathymetry.isel(longitude = i))
        bathymetry_mask[i,:] = a

    return SO4A_var_linear, SO4A_var_linear_horizontal, var_SO4A, bathymetry_mask, SO4A_longitude, SO4A_latitude


def P12_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    """    
    Extracts GLODAP bottle data collected along the P16S WOCE line.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns P16A_var_linear, P16A_var_linear_horizontal, var_P16A, bathymetry_mask,P16A_latitude, P16A_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year/cruise '1995-summer', '1994', '2011', '2008', '2001', '1996', '1995-winter'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    P12_var_linear = vertcially but not horizontally interpolated
    P12_var_linear_horizontal = vertically and horizontally interpolated
    var_P12 = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    P12_latitude, P12_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    
    interp_latitude = np.arange(-70,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    var_P121 = so_var.where(so_var.cruise == unique_cruise[144], drop = True)
    var_P122 = so_var.where(so_var.cruise == unique_cruise[143], drop = True)
    var_P122 = var_P122.where(var_P122.longitude > 120, drop = True)
    var_P123 = so_var.where(so_var.cruise == unique_cruise[27], drop = True)
    var_P123 = var_P123[:1288]
    var_P124 = so_var.where(so_var.cruise == unique_cruise[26], drop = True)       
    var_P125 = so_var.where(so_var.cruise == unique_cruise[21], drop = True)
    var_P125 = var_P125[:2063]
    var_P126 = so_var.where(so_var.cruise == unique_cruise[19], drop = True)
    var_P126 = var_P126.where(var_P126.longitude <150, drop = True)
    var_P127 = so_var.where(so_var.cruise == unique_cruise[18], drop = True)
    var_P127 = var_P127[1320:]
    
    # extract data from the relevant cruise passing through the P12 WOCE line
    if year == '1995-summer':
        var_P12 = var_P121
    elif year == '1994':
        var_P12 = var_P122
    elif year == '2011':
        var_P12 = var_P123
    elif year == '2008':
        var_P12 = var_P124       
    elif year == '2001':
        var_P12 = var_P125
    elif year == '1996':
        var_P12 = var_P126
    elif year == '1995-winter':
        var_P12 = var_P127
    else:
        return print('year = 1995-summer, 1995-winter,1994,2011,2008,2001,1996')
    
    # extract bathymetry information - use all available crossings
    P12_all =  xr.concat([var_P121, var_P122, var_P123, var_P125, var_P126, var_P127], dim = 'SAMPLE')
    P12_all_bathymetry = xr.DataArray(P12_all.bottomdepth, coords = [P12_all.latitude], dims = 'latitude')    
    P12_all_bathymetry = P12_all_bathymetry.sortby('latitude')
    P12_all_bathymetry = P12_all_bathymetry.groupby('latitude').max()
    P12_all_bathymetry = P12_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')

    
    # vertical interpolation
    P12_latitude = var_P12.latitude.values
    P12_longitude = var_P12.longitude.values
    P12_lat_index = np.sort(np.unique(P12_latitude, return_index = True)[1])
    P12_lat_index = np.append(P12_lat_index,len(P12_latitude))
    P12_lat_dimension = P12_latitude[P12_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    n_profiles = len(P12_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    P12_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [P12_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    P12_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_P12.where(var_P12.latitude == P12_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        P12_var_linear[i,:] = prof_var_linear
#     P12_var_linear = P12_var_linear.where(np.abs(P12_var_linear)>0,drop = True)
    P12_var_linear = P12_var_linear.sortby('latitude')
    
    # horizontal interpolation
    P12_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    P12_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = P12_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 3:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            P12_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = P12_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<P12_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a
        
    return P12_var_linear, P12_var_linear_horizontal, var_P12, bathymetry_mask,P12_latitude, P12_longitude

def A12a_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    """    
    Extracts GLODAP bottle data collected along the A12 WOCE line.
    Two slightly different paths are taken by skips in this region, they diverge around 55S.
    A12a is one of these paths.
    Note the 1996, 2005, and 2010 occupations are partial and only traverse the region shared by A12b.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns A12_var_linear, A12_var_linear_horizontal, var_A12, bathymetry_mask,A12_latitude, A12_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '1992', '1996', '1998', '2005','2010','1983'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    A12_var_linear = vertcially but not horizontally interpolated
    A12_var_linear_horizontal = vertically and horizontally interpolated
    var_A12 = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    A12_latitude, A12_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-70,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    A12_1 = so_var.where(so_var.cruise == unique_cruise[3], drop = True)
    A12_2 = so_var.where(so_var.cruise == unique_cruise[6], drop = True)
    A12_3 = so_var.where(so_var.cruise == unique_cruise[7], drop = True)
    A12_4 = so_var.where(so_var.cruise == unique_cruise[9], drop = True)
    A12_5 = so_var.where(so_var.cruise == unique_cruise[13], drop = True)
    A12_6 = so_var.where(so_var.cruise == unique_cruise[42], drop = True)
    
    A12_1 = A12_1.where(A12_1.longitude >-5, drop = True)
    A12_2 = A12_2.where(A12_2.longitude >-5, drop = True).where(A12_2.longitude < 1.5, drop = True)
    A12_3 = A12_3.where(A12_3.longitude >-3, drop = True)
    A12_3 =  xr.concat([A12_3[:241], A12_3[367:]], dim = 'SAMPLE')
    A12_4 = A12_4.where(A12_4.longitude >-5, drop = True).where(A12_4.longitude <2, drop = True)
    A12_5 = A12_5.where(A12_5.longitude >-2, drop = True)
    A12_6 = A12_6.where(A12_6.longitude >-2, drop = True)
    A12_6 = A12_6[468:]
    A12_6 =  xr.concat([A12_6[:457], A12_6[500:]], dim = 'SAMPLE')
    A12_6 =  xr.concat([A12_6[498:], A12_6[38:498]], dim = 'SAMPLE')
    
    
    if year == '1992':
        var_A12 = A12_1
    elif year == '1996':
        var_A12 = A12_2
    elif year == '1998':
        var_A12 = A12_3    
    elif year == '2005':
        var_A12 = A12_4      
    elif year == '2010':
        var_A12 = A12_5    
    elif year == '1983':
        var_A12 = A12_6    
    else:
        return print('year = 1992, 1996 (partial crossing), 1998, 2005 (partial crossing), 2010 (partial crossing), 1983')
    
    # extract bathymetry information - use all available crossings
    A12_all =  xr.concat([A12_1, A12_2, A12_3, A12_4, A12_5, A12_6], dim = 'SAMPLE')
    A12_all_bathymetry = xr.DataArray(A12_all.bottomdepth, coords = [A12_all.latitude], dims = 'latitude')    
    A12_all_bathymetry = A12_all_bathymetry.sortby('latitude')
    A12_all_bathymetry = A12_all_bathymetry.groupby('latitude').max()
    A12_all_bathymetry = A12_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')
    
    
    A12_latitude = var_A12.latitude.values
    A12_longitude = var_A12.longitude.values
    A12_lat_index = np.sort(np.unique(A12_latitude, return_index = True)[1])
    A12_lat_index = np.append(A12_lat_index,len(A12_latitude))
    A12_lat_dimension = A12_latitude[A12_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(A12_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    A12_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [A12_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    A12_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_A12.where(var_A12.latitude == A12_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        A12_var_linear[i,:] = prof_var_linear
    A12_var_linear = A12_var_linear.sortby('latitude')
    
    # horizontal interpolation
    A12_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    A12_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = A12_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 3:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            A12_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = A12_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<A12_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a
        
    return A12_var_linear, A12_var_linear_horizontal, var_A12, bathymetry_mask,A12_latitude, A12_longitude

def A12b_GLODAP_profile(variable, unique_cruise, year, so_glodap ):
    """    
    Extracts GLODAP bottle data collected along the A12 WOCE line.
    Two slightly different paths are taken by skips in this region, they diverge around 55S.
    A12b is one of these paths (more recent)
    Note the 1996, 2005, and 2010 occupations are partial and only traverse the region shared by A12a.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns A12_var_linear, A12_var_linear_horizontal, var_A12, bathymetry_mask,A12_latitude, A12_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '1996' (partial), '2005' (partial), '2010'(partial), '2002', '2008','2004', '2014'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    A12_var_linear = vertcially but not horizontally interpolated
    A12_var_linear_horizontal = vertically and horizontally interpolated
    var_A12 = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    A12_latitude, A12_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-70,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    A12_1 = so_var.where(so_var.cruise == unique_cruise[6], drop = True)
    A12_2 = so_var.where(so_var.cruise == unique_cruise[9], drop = True)
    A12_3 = so_var.where(so_var.cruise == unique_cruise[13], drop = True)
    A12_4 = so_var.where(so_var.cruise == unique_cruise[8], drop = True)
    A12_5 = so_var.where(so_var.cruise == unique_cruise[12], drop = True)
    A12_6 = so_var.where(so_var.cruise == unique_cruise[104], drop = True)
    A12_7 = so_var.where(so_var.cruise == unique_cruise[133], drop = True)
    A12_8 = so_var.where(so_var.cruise == unique_cruise[137], drop = True)

    A12_1 = A12_1.where(A12_1.longitude >-5, drop = True).where(A12_1.longitude < 1.5, drop = True)
    A12_2 = A12_2.where(A12_2.longitude >-5, drop = True).where(A12_2.longitude <2, drop = True)
    A12_3 = A12_3.where(A12_3.longitude >-2, drop = True)
    A12_4 = A12_4[:987]
    A12_5 = A12_5.where(A12_5.longitude >-1, drop = True)#.where(A12_8.longitude <0.5, drop = True)

    
    if year == '1996':
        var_A12 = A12_1    
    elif year == '2005':
        var_A12 = A12_2      
    elif year == '2010':
        var_A12 = A12_3  
    elif year == '2002':
        var_A12 = A12_4
    elif year == '2008':
        var_A12 = xr.concat([A12_5, A12_6], dim = 'SAMPLE')
    elif year == '2004':
        var_A12 = A12_7
    elif year == '2014':
        var_A12 = A12_8  
    else:
        return print('year = 1996 (partial), 2005 (partial), 2010(partial), 2002, 2008,2004, 2014')
    
    # extract bathymetry information - use all available crossings
    A12_all =  xr.concat([A12_1, A12_2, A12_3, A12_4, A12_5, A12_6, A12_7, A12_8], dim = 'SAMPLE')
    A12_all_bathymetry = xr.DataArray(A12_all.bottomdepth, coords = [A12_all.latitude], dims = 'latitude')    
    A12_all_bathymetry = A12_all_bathymetry.sortby('latitude')
    A12_all_bathymetry = A12_all_bathymetry.groupby('latitude').max()
    A12_all_bathymetry = A12_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')
    
    
    A12_latitude = var_A12.latitude.values
    A12_longitude = var_A12.longitude.values
    A12_lat_index = np.sort(np.unique(A12_latitude, return_index = True)[1])
    A12_lat_index = np.append(A12_lat_index,len(A12_latitude))
    A12_lat_dimension = A12_latitude[A12_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(A12_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    A12_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [A12_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    A12_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_A12.where(var_A12.latitude == A12_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        A12_var_linear[i,:] = prof_var_linear
    A12_var_linear = A12_var_linear.sortby('latitude')
    
    # horizontal interpolation
    A12_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    A12_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = A12_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 3:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            A12_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = A12_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<A12_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a
        
    return A12_var_linear, A12_var_linear_horizontal, var_A12, bathymetry_mask,A12_latitude, A12_longitude

def P15_GLODAP_profile(variable, unique_cruise, year, so_glodap, extend = True):
    """    
    Extracts GLODAP bottle data collected along the P15 WOCE line.
    If extend = True is specified, these lines are combined with a 2011 extension that continues south onto the continental shelf (not an option for 2001, 2009 transects which do not extend very far south).
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns P15_var_linear, P15_var_linear_horizontal, var_P15, bathymetry_mask,P15_latitude, P15_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '2001', '2009', '1996', '2016'
    extend = option to extend WOCE profile onto Antarctic continental shelf (but with data from 2011), default is True
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    P15_var_linear = vertcially but not horizontally interpolated
    P15_var_linear_horizontal = vertically and horizontally interpolated
    var_P15 = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    P15_latitude, P15_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-77,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    P15_1 = so_var.where(so_var.cruise == unique_cruise[33], drop = True)
    P15_2 = so_var.where(so_var.cruise == unique_cruise[34], drop = True)
    P15_3 = so_var.where(so_var.cruise == unique_cruise[61], drop = True)
    P15_4 = so_var.where(so_var.cruise == unique_cruise[71], drop = True)
    P15_5 = so_var.where(so_var.cruise == unique_cruise[142], drop = True)
    P15_1 = P15_1.where(P15_1.longitude<-160, drop = True)
    P15_2 = P15_2.where(P15_2.longitude<-160, drop = True)
    P15_3 = P15_3.where(P15_3.longitude<-160, drop = True)
    P15_4 = P15_4.where(P15_4.longitude<-160, drop = True)
    P15_4 = P15_4[625:]
    P15_5 = P15_5.where(P15_5.longitude<-160, drop = True)
    
    if year == '2001':
        var_P15 = P15_1
    elif year == '2009':
        var_P15 = P15_2
    elif year == '1996':
        var_P15 = P15_3
        if extend == True:
            var_P15 = xr.concat([P15_3, P15_4], dim = 'SAMPLE')
    elif year == '2016':
        var_P15 = P15_5
        if extend == True:
            var_P15 = xr.concat([P15_5, P15_4], dim = 'SAMPLE')
    else:
        return print('year = 2001 (partial), 2009 (partial), 2011 (full, extendable), 2016 (full, extendable)')

    
    
    # extract bathymetry information - use all available crossings
    P15_all =  xr.concat([P15_1,P15_2,P15_3,P15_4,P15_5], dim = 'SAMPLE')
    P15_all_bathymetry = xr.DataArray(P15_all.bottomdepth, coords = [P15_all.latitude], dims = 'latitude')    
    P15_all_bathymetry = P15_all_bathymetry.sortby('latitude')
    P15_all_bathymetry = P15_all_bathymetry.groupby('latitude').max()
    P15_all_bathymetry = P15_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')

    
    P15_latitude = var_P15.latitude.values
    P15_longitude = var_P15.longitude.values
    P15_lat_index = np.sort(np.unique(P15_latitude, return_index = True)[1])
    P15_lat_index = np.append(P15_lat_index,len(P15_latitude))
    P15_lat_dimension = P15_latitude[P15_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(P15_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    P15_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [P15_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    P15_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_P15.where(var_P15.latitude == P15_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        P15_var_linear[i,:] = prof_var_linear
    
    P15_var_linear = P15_var_linear.sortby('latitude')
    
    # horizontal interpolation
    P15_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    P15_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = P15_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 3:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            P15_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = P15_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<P15_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a

    return P15_var_linear, P15_var_linear_horizontal, var_P15, bathymetry_mask,P15_latitude, P15_longitude

def IO9S_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    """    
    Extracts GLODAP bottle data collected along the IO9S WOCE line.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns IO9S_var_linear, IO9S_var_linear_horizontal, var_IO9S, bathymetry_mask,IO9S_latitude, IO9S_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '1995', '2012', '2004'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    IO9S_var_linear = vertcially but not horizontally interpolated
    IO9S_var_linear_horizontal = vertically and horizontally interpolated
    var_IO9S = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    IO9S_latitude, IO9S_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-77,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    IO9S_1 = so_var.where(so_var.cruise == unique_cruise[51], drop = True)
    IO9S_2 = so_var.where(so_var.cruise == unique_cruise[28], drop = True)
    IO9S_3 = so_var.where(so_var.cruise == unique_cruise[23], drop = True)
    IO9S_3 = IO9S_3.where(IO9S_3.longitude > 100, drop = True).where(IO9S_3.longitude < 120, drop = True)
    IO9S_1 = IO9S_1.where(IO9S_1.longitude > 100, drop = True).where(IO9S_1.longitude < 120, drop = True)
    IO9S_2 = IO9S_2.where(IO9S_2.longitude > 100, drop = True).where(IO9S_2.longitude < 120, drop = True)
    
    if year == '1995':
        var_IO9S = IO9S_1
    elif year == '2012':
        var_IO9S = IO9S_2
    elif year == '2004':
        var_IO9S = IO9S_3
    else:
        return print('year = 1995, 2012, 2004')

#     print(var_IO9S)
#     print(var_IO9S.latitude)
    
    # extract bathymetry information - use all available crossings
    IO9S_all =  xr.concat([IO9S_1,IO9S_2,IO9S_3], dim = 'SAMPLE')
    IO9S_all_bathymetry = xr.DataArray(IO9S_all.bottomdepth, coords = [IO9S_all.latitude], dims = 'latitude')    
    IO9S_all_bathymetry = IO9S_all_bathymetry.sortby('latitude')
    IO9S_all_bathymetry = IO9S_all_bathymetry.groupby('latitude').max()
    IO9S_all_bathymetry = IO9S_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')

    
    IO9S_latitude = var_IO9S.latitude.values
    IO9S_longitude = var_IO9S.longitude.values
    IO9S_lat_index = np.sort(np.unique(IO9S_latitude, return_index = True)[1])
    IO9S_lat_index = np.append(IO9S_lat_index,len(IO9S_latitude))
    IO9S_lat_dimension = IO9S_latitude[IO9S_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(IO9S_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    IO9S_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [IO9S_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    IO9S_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_IO9S.where(var_IO9S.latitude == IO9S_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 5:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        IO9S_var_linear[i,:] = prof_var_linear
    
    IO9S_var_linear = IO9S_var_linear.sortby('latitude')
    
    # horizontal interpolation
    IO9S_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    IO9S_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = IO9S_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 5:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            IO9S_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = IO9S_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<IO9S_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a

    return IO9S_var_linear, IO9S_var_linear_horizontal, var_IO9S, bathymetry_mask,IO9S_latitude, IO9S_longitude

def IO8S_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    
    """    
    Extracts GLODAP bottle data collected along the IO8S WOCE line.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns IO8S_var_linear, IO8S_var_linear_horizontal, var_IO8S, bathymetry_mask,IO8S_latitude, IO8S_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '1995', '2007','2016'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    IO8S_var_linear = vertcially but not horizontally interpolated
    IO8S_var_linear_horizontal = vertically and horizontally interpolated
    var_IO8S = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    IO8S_latitude, IO8S_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-77,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    IO8S_1 = so_var.where(so_var.cruise == unique_cruise[51], drop = True)
    IO8S_2 = so_var.where(so_var.cruise == unique_cruise[86], drop = True)
    IO8S_3 = so_var.where(so_var.cruise == unique_cruise[151], drop = True)
    IO8S_1 = IO8S_1.where(IO8S_1.longitude < 100, drop = True)
    
    if year == '1995':
        var_IO8S = IO8S_1
    elif year == '2007':
        var_IO8S = IO8S_2
    elif year == '2016':
        var_IO8S = IO8S_3
    else:
        return print('year = 1995, 2007, 2016')
    
    # extract bathymetry information - use all available crossings
    IO8S_all =  xr.concat([IO8S_1,IO8S_2,IO8S_3], dim = 'SAMPLE')
    IO8S_all_bathymetry = xr.DataArray(IO8S_all.bottomdepth, coords = [IO8S_all.latitude], dims = 'latitude')    
    IO8S_all_bathymetry = IO8S_all_bathymetry.sortby('latitude')
    IO8S_all_bathymetry = IO8S_all_bathymetry.groupby('latitude').max()
    IO8S_all_bathymetry = IO8S_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')

    
    IO8S_latitude = var_IO8S.latitude.values
    IO8S_longitude = var_IO8S.longitude.values
    IO8S_lat_index = np.sort(np.unique(IO8S_latitude, return_index = True)[1])
    IO8S_lat_index = np.append(IO8S_lat_index,len(IO8S_latitude))
    IO8S_lat_dimension = IO8S_latitude[IO8S_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(IO8S_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    IO8S_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [IO8S_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    IO8S_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_IO8S.where(var_IO8S.latitude == IO8S_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 5:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        IO8S_var_linear[i,:] = prof_var_linear
    
    IO8S_var_linear = IO8S_var_linear.sortby('latitude')
    
    # horizontal interpolation
    IO8S_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    IO8S_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = IO8S_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 5:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            IO8S_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = IO8S_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<IO8S_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a

    return IO8S_var_linear, IO8S_var_linear_horizontal, var_IO8S, bathymetry_mask,IO8S_latitude, IO8S_longitude

def P19_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    """    
    Extracts GLODAP bottle data collected along the P19 WOCE line.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns P19_var_linear, P19_var_linear_horizontal, var_P19, bathymetry_mask,P19_latitude, P19_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '2017', '2008', '1994'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    P19_var_linear = vertcially but not horizontally interpolated
    P19_var_linear_horizontal = vertically and horizontally interpolated
    var_P19 = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    P19_latitude, P19_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-77,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    P19_1 = so_var.where(so_var.cruise == unique_cruise[150], drop = True)
    P19_2 = so_var.where(so_var.cruise == unique_cruise[80], drop = True)
    P19_3 = so_var.where(so_var.cruise == unique_cruise[60], drop = True)
    P19_1 = P19_1.where(P19_1.longitude >-105, drop = True).where(P19_1.longitude <-100, drop = True)
    P19_2 = P19_2.where(P19_2.longitude >-105, drop = True).where(P19_2.longitude <-100, drop = True)
    P19_3 = P19_3.where(P19_3.longitude >-105, drop = True).where(P19_3.longitude <-100, drop = True)
    
    if year == '2017':
        var_P19 = P19_1
    elif year == '2008':
        var_P19 = P19_2
    elif year == '1994':
        var_P19 = P19_3
    else:
        return print('year = 1994, 2008, 2017')

    # extract bathymetry information - use all available crossings
    P19_all =  xr.concat([P19_1,P19_2,P19_3], dim = 'SAMPLE')
    P19_all_bathymetry = xr.DataArray(P19_all.bottomdepth, coords = [P19_all.latitude], dims = 'latitude')    
    P19_all_bathymetry = P19_all_bathymetry.sortby('latitude')
    P19_all_bathymetry = P19_all_bathymetry.groupby('latitude').max()
    P19_all_bathymetry = P19_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')

    
    P19_latitude = var_P19.latitude.values
    P19_longitude = var_P19.longitude.values
    P19_lat_index = np.sort(np.unique(P19_latitude, return_index = True)[1])
    P19_lat_index = np.append(P19_lat_index,len(P19_latitude))
    P19_lat_dimension = P19_latitude[P19_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(P19_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    P19_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [P19_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    P19_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_P19.where(var_P19.latitude == P19_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 5:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        P19_var_linear[i,:] = prof_var_linear
    
    P19_var_linear = P19_var_linear.sortby('latitude')
    
    # horizontal interpolation
    P19_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    P19_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = P19_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 5:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            P19_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = P19_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<P19_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a

    return P19_var_linear, P19_var_linear_horizontal, var_P19, bathymetry_mask,P19_latitude, P19_longitude

def SO4P_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    """    
    Extracts GLODAP bottle data collected along the SO4P WOCE line.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in latitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine longitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns SO4P_var_linear, SO4P_var_linear_horizontal, var_SO4P, bathymetry_mask,SO4P_longitude, SO4P_latitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = string of chosen year '2011', '1994'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    SO4P_var_linear = vertcially but not horizontally interpolated
    SO4P_var_linear_horizontal = vertically and horizontally interpolated
    var_SO4P = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    SO4P_latitude, SO4P_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_longitude = np.arange(-180,-72,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))

    so_var = so_glodap.sel(VARIABLE = variable)
    SO4P_1 = so_var.where(so_var.cruise == unique_cruise[134], drop = True)
    SO4P_2 = so_var.where(so_var.cruise == unique_cruise[71], drop = True)
    SO4P_1 = SO4P_1.where(SO4P_1.latitude >-67.1, drop = True).where(SO4P_1.latitude <-66.9, drop = True)
    SO4P_1 = SO4P_1.where(SO4P_1.longitude >-67.1, drop = True).where(SO4P_1.latitude <-66.9, drop = True)
    SO4P_2 = SO4P_2.where(SO4P_2.latitude >-67.1, drop = True).where(SO4P_2.latitude <-66.9, drop = True)
        
    if year == '1992':
        var_SO4P = SO4P_1
    elif year == '2011':
        var_SO4P = SO4P_2
    else:
        return print('year = 1992, 2011')
    
    # extract bathymetry information - use all available crossings
    SO4P_all =  xr.concat([SO4P_1,SO4P_2], dim = 'SAMPLE')
    SO4P_all_bathymetry = xr.DataArray(SO4P_all.bottomdepth, coords = [SO4P_all.longitude], dims = 'longitude')    
    SO4P_all_bathymetry = SO4P_all_bathymetry.sortby('longitude')
    SO4P_all_bathymetry = SO4P_all_bathymetry.groupby('longitude').max()
    SO4P_all_bathymetry = SO4P_all_bathymetry.interp(longitude = interp_longitude, method = 'linear')
    
    
    SO4P_longitude = var_SO4P.longitude.values
    SO4P_latitude = var_SO4P.latitude.values
    SO4P_lon_index = np.sort(np.unique(SO4P_longitude, return_index = True)[1])
    SO4P_lon_index = np.append(SO4P_lon_index,len(SO4P_longitude))
    SO4P_lon_dimension = SO4P_longitude[SO4P_lon_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(SO4P_lon_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    SO4P_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [SO4P_lon_dimension, interp_depth], dims = ['longitude', 'depth'])
    SO4P_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lon_var = var_SO4P.where(var_SO4P.longitude == SO4P_lon_dimension[i], drop = True)
        lon_var = lon_var.sortby(lon_var.depth)
        prof_xarray = xr.DataArray(lon_var.values, coords = [lon_var.depth.values], dims = 'depth')
        if len(np.unique(lon_var.depth.values)) != len(lon_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 3:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        SO4P_var_linear[i,:] = prof_var_linear
    SO4P_var_linear = SO4P_var_linear.sortby('longitude')
    
    # horizontal interpolation
    SO4P_var_linear_horizontal = xr.DataArray(np.empty((len(interp_longitude),len(interp_depth))), coords = [interp_longitude, interp_depth], dims = ['longitude', 'depth'])
    SO4P_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = SO4P_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 3:
            depth_var = depth_var.interp(longitude = interp_longitude, method = 'linear')
            SO4P_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = SO4P_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.longitude)):
        a = bathymetry_mask.isel(longitude = i)
        a = a.where(a.depth<SO4P_all_bathymetry.isel(longitude = i))
        bathymetry_mask[i,:] = a

    return SO4P_var_linear, SO4P_var_linear_horizontal, var_SO4P, bathymetry_mask, SO4P_longitude, SO4P_latitude

def A23_GLODAP_profile(variable, unique_cruise, so_glodap):
    """    
    Extracts GLODAP bottle data collected along the A23 (ish) WOCE line.
    Bottle data along the cruise line are compressed into latitude/depth space (ignoring small variations in longitude from the main WOCE line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine latitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns A23_var_linear, A23_var_linear_horizontal, var_A23, bathymetry_mask,A23_latitude, A23_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    (no year variable, only one available crosisng in 1995)
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    A23_var_linear = vertcially but not horizontally interpolated
    A23_var_linear_horizontal = vertically and horizontally interpolated
    var_A23 = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    A23_latitude, A23_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_latitude = np.arange(-77,-30,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    var_A23 = so_var.where(so_var.cruise == unique_cruise[131], drop = True)
    var_A23 = var_A23.where(var_A23.longitude>-50, drop = True)
    var_A23 = var_A23[198:]

    # extract bathymetry information - use all available crossings
    A23_all_bathymetry = xr.DataArray(var_A23.bottomdepth, coords = [var_A23.latitude], dims = 'latitude')    
    A23_all_bathymetry = A23_all_bathymetry.sortby('latitude')
    A23_all_bathymetry = A23_all_bathymetry.groupby('latitude').max()
    A23_all_bathymetry = A23_all_bathymetry.interp(latitude = interp_latitude, method = 'linear')

    
    A23_latitude = var_A23.latitude.values
    A23_longitude = var_A23.longitude.values
    A23_lat_index = np.sort(np.unique(A23_latitude, return_index = True)[1])
    A23_lat_index = np.append(A23_lat_index,len(A23_latitude))
    A23_lat_dimension = A23_latitude[A23_lat_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(A23_lat_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    A23_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [A23_lat_dimension, interp_depth], dims = ['latitude', 'depth'])
    A23_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lat_var = var_A23.where(var_A23.latitude == A23_lat_dimension[i], drop = True)
        lat_var = lat_var.sortby(lat_var.depth)
        prof_xarray = xr.DataArray(lat_var.values, coords = [lat_var.depth.values], dims = 'depth')
        if len(np.unique(lat_var.depth.values)) != len(lat_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 5:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        A23_var_linear[i,:] = prof_var_linear
    
    A23_var_linear = A23_var_linear.sortby('latitude')
    
    # horizontal interpolation
    A23_var_linear_horizontal = xr.DataArray(np.empty((len(interp_latitude),len(interp_depth))), coords = [interp_latitude, interp_depth], dims = ['latitude', 'depth'])
    A23_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = A23_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 5:
            depth_var = depth_var.interp(latitude = interp_latitude, method = 'linear')
            A23_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = A23_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.latitude)):
        a = bathymetry_mask.isel(latitude = i)
        a = a.where(a.depth<A23_all_bathymetry.isel(latitude = i))
        bathymetry_mask[i,:] = a

    return A23_var_linear, A23_var_linear_horizontal, var_A23, bathymetry_mask,A23_latitude, A23_longitude


def TP_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    """    
    Extracts GLODAP bottle data collected along a trans Pacific southern ocean transect (around 30S)
    Bottle data along the cruise line are compressed into longitude/depth space (ignoring small variations in latitude from the main line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine longitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns TP_var_linear, TP_var_linear_horizontal, var_TP, bathymetry_mask,TP_latitude, TP_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = '1992', '2009-2010', '2003'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    TP_var_linear = vertcially but not horizontally interpolated
    TP_var_linear_horizontal = vertically and horizontally interpolated
    var_TP = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    TP_latitude, TP_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_longitude = np.arange(153,289,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    TP_1 = so_var.where(so_var.cruise == unique_cruise[45], drop = True)
    TP_2 = so_var.where(so_var.cruise == unique_cruise[59], drop = True)
    TP_3 = so_var.where(so_var.cruise == unique_cruise[113], drop = True)
    ###
    TP_1_poslon = TP_1.longitude.where(TP_1.longitude >0)
    TP_1_neglon = TP_1.longitude.where(TP_1.longitude <0)
    TP_1_neglon = 360+TP_1_neglon
    TP_1_lon = TP_1_poslon.fillna(0) + TP_1_neglon.fillna(0)
    TP_1.coords['longitude'] = TP_1_lon
    ###
    TP_2_poslon = TP_2.longitude.where(TP_2.longitude >0)
    TP_2_neglon = TP_2.longitude.where(TP_2.longitude <0)
    TP_2_neglon = 360+TP_2_neglon
    TP_2_lon = TP_2_poslon.fillna(0) + TP_2_neglon.fillna(0)
    TP_2.coords['longitude'] = TP_2_lon
    ###
    TP_3_poslon = TP_3.longitude.where(TP_3.longitude >0)
    TP_3_neglon = TP_3.longitude.where(TP_3.longitude <0)
    TP_3_neglon = 360+TP_3_neglon
    TP_3_lon = TP_3_poslon.fillna(0) + TP_3_neglon.fillna(0)
    TP_3.coords['longitude'] = TP_3_lon
    
    if year == '1992': # winter
        var_TP = TP_1
    elif year == '2009-2010': # summer
        var_TP = TP_2
    elif year == '2003': # spring
        var_TP = TP_3
    else:
        return print('year = 1992, 2009-2010, 2003')

    # extract bathymetry information - use all available crossings
    TP_all =  xr.concat([TP_1,TP_2,TP_3], dim = 'SAMPLE')
    TP_all_bathymetry = xr.DataArray(TP_all.bottomdepth, coords = [TP_all.longitude], dims = 'longitude')    
    TP_all_bathymetry = TP_all_bathymetry.sortby('longitude')
    TP_all_bathymetry = TP_all_bathymetry.groupby('longitude').max()
    TP_all_bathymetry = TP_all_bathymetry.interp(longitude = interp_longitude, method = 'linear')

    
    TP_latitude = var_TP.latitude.values
    TP_longitude = var_TP.longitude.values
    TP_lon_index = np.sort(np.unique(TP_longitude, return_index = True)[1])
    TP_lon_index = np.append(TP_lon_index,len(TP_longitude))
    TP_lon_dimension = TP_longitude[TP_lon_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(TP_lon_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    TP_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [TP_lon_dimension, interp_depth], dims = ['longitude', 'depth'])
    TP_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lon_var = var_TP.where(var_TP.longitude == TP_lon_dimension[i], drop = True)
        lon_var = lon_var.sortby(lon_var.depth)
        prof_xarray = xr.DataArray(lon_var.values, coords = [lon_var.depth.values], dims = 'depth')
        if len(np.unique(lon_var.depth.values)) != len(lon_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 5:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        TP_var_linear[i,:] = prof_var_linear
    
    TP_var_linear = TP_var_linear.sortby('longitude')
    
    # horizontal interpolation
    TP_var_linear_horizontal = xr.DataArray(np.empty((len(interp_longitude),len(interp_depth))), coords = [interp_longitude, interp_depth], dims = ['longitude', 'depth'])
    TP_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = TP_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 5:
            depth_var = depth_var.interp(longitude = interp_longitude, method = 'linear')
            TP_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = TP_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.longitude)):
        a = bathymetry_mask.isel(longitude = i)
        a = a.where(a.depth<TP_all_bathymetry.isel(longitude = i))
        bathymetry_mask[i,:] = a

    return TP_var_linear, TP_var_linear_horizontal, var_TP, bathymetry_mask, TP_longitude, TP_latitude

def TI_GLODAP_profile(variable, unique_cruise, year, so_glodap):
    """    
    Extracts GLODAP bottle data collected along a trans Indian southern ocean transect (around 32S)
    Bottle data along the cruise line are compressed into longitude/depth space (ignoring small variations in latitude from the main line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine longitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns TI_var_linear, TI_var_linear_horizontal, var_TI, bathymetry_mask,TI_latitude, TI_longitude
    
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    year = '2002' '2009'
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    TI_var_linear = vertcially but not horizontally interpolated
    TI_var_linear_horizontal = vertically and horizontally interpolated
    var_TI = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    TI_latitude, TI_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_longitude = np.arange(30,116,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    TI_1 = so_var.where(so_var.cruise == unique_cruise[88], drop = True)
    TI_2 = so_var.where(so_var.cruise == unique_cruise[124], drop = True)
    
    if year == '2009': # winter
        var_TI = TI_1
    elif year == '2002': # summer
        var_TI = TI_2
    else:
        return print('year = 2002, 2009')

    # extract bathymetry information - use all available crossings
    TI_all =  xr.concat([TI_1,TI_2], dim = 'SAMPLE')
    TI_all_bathymetry = xr.DataArray(TI_all.bottomdepth, coords = [TI_all.longitude], dims = 'longitude')    
    TI_all_bathymetry = TI_all_bathymetry.sortby('longitude')
    TI_all_bathymetry = TI_all_bathymetry.groupby('longitude').max()
    TI_all_bathymetry = TI_all_bathymetry.interp(longitude = interp_longitude, method = 'linear')

    
    TI_latitude = var_TI.latitude.values
    TI_longitude = var_TI.longitude.values
    TI_lon_index = np.sort(np.unique(TI_longitude, return_index = True)[1])
    TI_lon_index = np.append(TI_lon_index,len(TI_longitude))
    TI_lon_dimension = TI_longitude[TI_lon_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(TI_lon_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    TI_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [TI_lon_dimension, interp_depth], dims = ['longitude', 'depth'])
    TI_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lon_var = var_TI.where(var_TI.longitude == TI_lon_dimension[i], drop = True)
        lon_var = lon_var.sortby(lon_var.depth)
        prof_xarray = xr.DataArray(lon_var.values, coords = [lon_var.depth.values], dims = 'depth')
        if len(np.unique(lon_var.depth.values)) != len(lon_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 5:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        TI_var_linear[i,:] = prof_var_linear
    
    TI_var_linear = TI_var_linear.sortby('longitude')
    
    # horizontal interpolation
    TI_var_linear_horizontal = xr.DataArray(np.empty((len(interp_longitude),len(interp_depth))), coords = [interp_longitude, interp_depth], dims = ['longitude', 'depth'])
    TI_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = TI_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 5:
            depth_var = depth_var.interp(longitude = interp_longitude, method = 'linear')
            TI_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = TI_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.longitude)):
        a = bathymetry_mask.isel(longitude = i)
        a = a.where(a.depth<TI_all_bathymetry.isel(longitude = i))
        bathymetry_mask[i,:] = a

    return TI_var_linear, TI_var_linear_horizontal, var_TI, bathymetry_mask, TI_longitude, TI_latitude

def TAa_GLODAP_profile(variable, unique_cruise, so_glodap):
    """    
    Extracts GLODAP bottle data collected along a trans Atlantic southern ocean transect (ranges 30S to 45S)
    Bottle data along the cruise line are compressed into longitude/depth space (ignoring small variations in latitude from the main line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine longitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns TA_var_linear, TIAvar_linear_horizontal, var_TA, bathymetry_mask,TA_latitude, TA_longitude
    
    Only one crossing available (1992-1993 summer).
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    TA_var_linear = vertcially but not horizontally interpolated
    TA_var_linear_horizontal = vertically and horizontally interpolated
    var_TA = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    TA_latitude, TA_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_longitude = np.arange(-61,16,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    TA = so_var.where(so_var.cruise == unique_cruise[125], drop = True)
    
    var_TA = TA

    # extract bathymetry information - use all available crossings
    TA_all_bathymetry = xr.DataArray(TA.bottomdepth, coords = [TA.longitude], dims = 'longitude')    
    TA_all_bathymetry = TA_all_bathymetry.sortby('longitude')
    TA_all_bathymetry = TA_all_bathymetry.groupby('longitude').max()
    TA_all_bathymetry = TA_all_bathymetry.interp(longitude = interp_longitude, method = 'linear')

    
    TA_latitude = var_TA.latitude.values
    TA_longitude = var_TA.longitude.values
    TA_lon_index = np.sort(np.unique(TA_longitude, return_index = True)[1])
    TA_lon_index = np.append(TA_lon_index,len(TA_longitude))
    TA_lon_dimension = TA_longitude[TA_lon_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(TA_lon_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    TA_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [TA_lon_dimension, interp_depth], dims = ['longitude', 'depth'])
    TA_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lon_var = var_TA.where(var_TA.longitude == TA_lon_dimension[i], drop = True)
        lon_var = lon_var.sortby(lon_var.depth)
        prof_xarray = xr.DataArray(lon_var.values, coords = [lon_var.depth.values], dims = 'depth')
        if len(np.unique(lon_var.depth.values)) != len(lon_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 5:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        TA_var_linear[i,:] = prof_var_linear
    
    TA_var_linear = TA_var_linear.sortby('longitude')
    
    # horizontal interpolation
    TA_var_linear_horizontal = xr.DataArray(np.empty((len(interp_longitude),len(interp_depth))), coords = [interp_longitude, interp_depth], dims = ['longitude', 'depth'])
    TA_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = TA_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 5:
            depth_var = depth_var.interp(longitude = interp_longitude, method = 'linear')
            TA_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = TA_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.longitude)):
        a = bathymetry_mask.isel(longitude = i)
        a = a.where(a.depth<TA_all_bathymetry.isel(longitude = i))
        bathymetry_mask[i,:] = a

    return TA_var_linear, TA_var_linear_horizontal, var_TA, bathymetry_mask, TA_longitude, TA_latitude

def TAb_GLODAP_profile(variable, unique_cruise, so_glodap):
    """    
    Extracts GLODAP bottle data collected along a trans Atlantic southern ocean transect (35S)
    Bottle data along the cruise line are compressed into longitude/depth space (ignoring small variations in latitude from the main line) then the point data are linearly interpolated onto fine vertical and horizontal grids to aid visualisation.
    A bathymetry file is generated from information collected from all crossings, this is also linearly interpolated onto a fine longitude grid and used to mask out where interpolation assigns tracer values to non ocean locations.
    
    returns TA_var_linear, TIAvar_linear_horizontal, var_TA, bathymetry_mask,TA_latitude, TA_longitude
    
    Only one crossing available (2017 summer).
    inputs:
    variable = string of variable to be extracted, as in so_glodap.VARIABLES (e.g. 'temperature', 'salinity')
    unique_cruise = list of indices marking hte start of each unique cruise in the so_glodap dataset (cruise = so_glodap.cruise.values \ unique_cruise = np.unique(cruise))
    so_glodap = xr.open_dataset('/work/Ruth.Moorman/GLODAP/GLODAPv2.2019_Southern_Ocean_30S.nc') + extract DataArray

    outputs:
    TA_var_linear = vertcially but not horizontally interpolated
    TA_var_linear_horizontal = vertically and horizontally interpolated
    var_TA = no interpolation, shows the position of the samples
    bathymetry_mask = bathymetry mask file (nan below seabed, 1 above)
    TA_latitude, TA_longitude = location of profiles, to be used to map the transect location if desired.
    
    Ruth Moorman, April 2020, Princeton University
    """
    interp_longitude = np.arange(-60,25,0.1)
    interp_depth = np.append(np.arange(1,1000,1),np.arange(1000,6010,10))
    
    so_var = so_glodap.sel(VARIABLE = variable)
    TA = so_var.where(so_var.cruise == unique_cruise[138], drop = True)
    
    var_TA = TA

    # extract bathymetry information - use all available crossings
    TA_all_bathymetry = xr.DataArray(TA.bottomdepth, coords = [TA.longitude], dims = 'longitude')    
    TA_all_bathymetry = TA_all_bathymetry.sortby('longitude')
    TA_all_bathymetry = TA_all_bathymetry.groupby('longitude').max()
    TA_all_bathymetry = TA_all_bathymetry.interp(longitude = interp_longitude, method = 'linear')

    
    TA_latitude = var_TA.latitude.values
    TA_longitude = var_TA.longitude.values
    TA_lon_index = np.sort(np.unique(TA_longitude, return_index = True)[1])
    TA_lon_index = np.append(TA_lon_index,len(TA_longitude))
    TA_lon_dimension = TA_longitude[TA_lon_index[0:-1]] # latitude dimension of transect
    # interpolate onto a fine vertical grid (later sort in latitude coordinate)
    n_profiles = len(TA_lon_index)-1
    ## create an empty xarray with dimensions N_PROF, depth
    TA_var_linear = xr.DataArray(np.empty((n_profiles,len(interp_depth))), coords = [TA_lon_dimension, interp_depth], dims = ['longitude', 'depth'])
    TA_var_linear[:,:] = np.nan
    
    for i in range(n_profiles):
        lon_var = var_TA.where(var_TA.longitude == TA_lon_dimension[i], drop = True)
        lon_var = lon_var.sortby(lon_var.depth)
        prof_xarray = xr.DataArray(lon_var.values, coords = [lon_var.depth.values], dims = 'depth')
        if len(np.unique(lon_var.depth.values)) != len(lon_var.depth.values):
            prof_xarray = prof_xarray.groupby('depth').mean() #average data recorded at repeat depths (necessary for interpolation)
        prof_xarray=prof_xarray.where(np.abs(prof_xarray)>0,drop = True )
        if len(prof_xarray.depth) > 5:
            prof_var_linear = prof_xarray.interp(depth = interp_depth, method = 'linear')
        else:
            fill = np.empty((len(interp_depth)))
            fill[:] = np.nan
            prof_var_linear = fill
        TA_var_linear[i,:] = prof_var_linear
    
    TA_var_linear = TA_var_linear.sortby('longitude')
    
    # horizontal interpolation
    TA_var_linear_horizontal = xr.DataArray(np.empty((len(interp_longitude),len(interp_depth))), coords = [interp_longitude, interp_depth], dims = ['longitude', 'depth'])
    TA_var_linear_horizontal[:,:] = np.nan
    
    for i in range(len(interp_depth)):
        depth_var = TA_var_linear.isel(depth = i)
        depth_var = depth_var.where(np.abs(depth_var)>0, drop = True)
        if len(depth_var) > 5:
            depth_var = depth_var.interp(longitude = interp_longitude, method = 'linear')
            TA_var_linear_horizontal[:,i] = depth_var
   
    bathymetry_mask = TA_var_linear_horizontal * 0 + 1
    bathymetry_mask = bathymetry_mask.fillna(1)
    for i in range(len(bathymetry_mask.longitude)):
        a = bathymetry_mask.isel(longitude = i)
        a = a.where(a.depth<TA_all_bathymetry.isel(longitude = i))
        bathymetry_mask[i,:] = a

    return TA_var_linear, TA_var_linear_horizontal, var_TA, bathymetry_mask, TA_longitude, TA_latitude