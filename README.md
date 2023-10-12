# PyHD - PyHeatDemand - Processing Heat Demand Data

<a name="overview"></a>
# Overview 
Knowledge about the heat demand (MWh/area/year) of a respective building, district, city, state, country or even on a 
continental scale is crucial for an adequate heat demand planning or planning for providing power plant capacities.

**PyHeatDemand** is a Python library for processing spatial data containing local, regional or even national heat demand 
data. The heat demand data can be provided as Raster, gridded polygon Shapefile, building footprint polygons Shapefile, 
street network linestring Shapefile, point Shapefiles representing the heat demand of a single building or an 
administrative area, and lastly postal addresses provided as CSV files.  

The package is a result of the works of Herbst et al., (2021) within the 
[Interreg NWE project DGE Rollout, NWE 892](http://www.nweurope.eu/DGE-Rollout). The final heat demand map is also 
accessible within the [DGE Rollout Webviewer](https://data.geus.dk/egdi/?mapname=dgerolloutwebtool#baslay=baseMapGEUS&extent=39620,-1581250,8465360,8046630&layers=dge_heat_final).


<a name="installation"></a>
## Installation
**PyHeatDemand** can be installed via PyPi using `pip install pyhd`. It is recommended to use the provided 
environment.yml and use `conda env create -f environment.yml` to ensure that all dependencies are installed corredctly. 
Make sure that you have downloaded the environment file in that case. Alternatively, you can fork and clone the 
repository or clone it directly from the repository page.

<a name="workflow"></a>
## General Workflow

<p align="center"><img src="https://raw.githubusercontent.com/AlexanderJuestel/pyhd/docs/images/fig1.png" width="600">



<a name="ref"></a>
## References

Herbst, E., Khashfe, E., Jüstel, A., Strozyk, F. & Kukla, P., 2021. A Heat Demand Map of North-West Europe – its impact 
on supply areas and identification of potential production areas for deep geothermal energy. GeoKarlsruhe 2021, 
http://dx.doi.org/10.48380/dggv-j2wj-nk88. 
