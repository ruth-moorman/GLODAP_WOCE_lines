# GLODAP_WOCE_lines

This repo contains a set of python functions (in GLODAP_WOCE_sections.py) that extract GLODAP v2 2019 bottle data (physical and biogeochemical) from occupations of notable Southern Ocean WOCE lines for easy plotting and analysis.

To use these tools you will first be required to obtain the GLODAP v2 2019 bottle data in the fully merged .mat format from https://www.nodc.noaa.gov/archive/arc0107/0162565/2.2/data/0-data/data_product/ before post processing that data into an xarray friendly .nc format using the procedure in SO_GLODAP_create_nc.ipynb. This procedure will only save Southern Ocean bottle data, to save storage space.

Then the generated .nc file may be used as the input for functions in GLODAP_WOCE_sections.py. Example transects and details of function use are in GLODAP_WOCE_sections_examples.ipynb, and a map of available crossings and the years of each crossing are in transect_map.png.
