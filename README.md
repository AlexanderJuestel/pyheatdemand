# PyHeatDemand - Processing Tool for Heat Demand Data

<a name="overview"></a>
# Overview 
Knowledge about the heat demand (MWh/area/year) of a respective building, district, city, state, country or even on a 
continental scale is crucial for an adequate heat demand planning or planning for providing power plant capacities.

**PyHeatDemand** is a Python library for processing spatial data containing local, regional or even national heat demand 
data. The heat demand data can be provided as Raster, gridded polygon Shapefile, building footprint polygons Shapefile, 
street network linestring Shapefile, point Shapefiles representing the heat demand of a single building or an 
administrative area, and lastly postal addresses provided as CSV files.  

The package is a continuation of the results of Herbst et al., (2021) within the 
[Interreg NWE project DGE Rollout, NWE 892](http://www.nweurope.eu/DGE-Rollout). E. Herbst and E. Khashfe compiled the 
original heat demand map as part of their respective master thesis project at RWTH Aachen University. The final heat 
demand map is also accessible within the [DGE Rollout Webviewer](https://data.geus.dk/egdi/?mapname=dgerolloutwebtool#baslay=baseMapGEUS&extent=39620,-1581250,8465360,8046630&layers=dge_heat_final).


<a name="installation"></a>
## Installation
**PyHeatDemand** can be installed via PyPi using `pip install pyheatdemand` or via Anaconda using `conda install -c conda-forge pyheatdemand`. 
It is recommended to use the provided 
environment.yml and use `conda env create -f environment.yml` to ensure that all dependencies are installed correctly. 
Make sure that you have downloaded the environment file in that case. Alternatively, you can fork and clone the 
repository or clone it directly from the repository page. An additional `requirements.txt` provides a list of all necessary dependencies.

<a name="workflow"></a>
## General Workflow

The general workflow involves creating a global mask of quadratic polygons (e.g. 10 km x 10 km) covering the entire 
studied area. This is especially used for larger areas such as states, countries or the Interreg NWE region to subdivide 
the area into smaller areas. Depending on the size of the input heat demand data, the corresponding underlying global 
mask polygons are selected and the final (e.g. 100 m x 100 m) resolution polygon grid is created. This grid including 
the input heat demand data is need to calculate the final heat demand map. 
![Fig1](docs/images/fig1.png)

The actual heat demand data is divided into four categories or data categories:
* Data Category 1: Heat demand raster data or gridded polygon (vector) data, different scales possible
* Data Category 2: Heat demand data as vector data; building footprints as polygons, street network as linestrings, 
single houses as points
* Data Category 3: Heat demand as points representative for an administrative area
* Data Category 4: Other forms of Heat Demand data such as addresses with assciated heat demand or heat demand provided
as usage of other fuels, e.g. gas demand, biomass demand etc.

Processing steps for Data Types 1 + 2
![Fig1](docs/images/fig2.png)

Processing steps for Data Types 3
![Fig1](docs/images/fig3.png)

## Continuous Integration
A CI is present to test the current code. It can be initiated using `pytest --cov` within the `test` folder. After 
running the tests, `coverage report -m` can be executed to get an report on the coverage and which lines are not covered
by the tests.

<a name="ref"></a>
## References

Herbst, E., Khashfe, E., Jüstel, A., Strozyk, F. & Kukla, P., 2021. A Heat Demand Map of North-West Europe – its impact 
on supply areas and identification of potential production areas for deep geothermal energy. GeoKarlsruhe 2021, 
http://dx.doi.org/10.48380/dggv-j2wj-nk88. 
