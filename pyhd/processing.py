"""
Contributors: Alexander Jüstel, Elias Khashfe, Eileen Herbst

"""

import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import Polygon, shape, box, Point
from itertools import product
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
from rasterstats import zonal_stats
import os
import pyproj
from typing import Union
from tqdm import tqdm
import geopy
import osmnx


def create_polygon_mask(gdf: gpd.GeoDataFrame,
                        step_size: int,
                        crop_gdf: bool = False) -> gpd.GeoDataFrame:
    """Create a mask GeoDataFrame consisting of squares with a defined step_size.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame over which a mask is created.
        step_size : int
            Size of the rasterized squares in meters, e.g. ``step_size=100``.
        crop_gdf : bool, default: ``False``
            Boolean to either crop the GeoDataFrame to the outline or return it as is, e.g. ``crop_gdf=False``.

    Returns
    --------
        gdf_mask : gpd.GeoDataFrame
            GeoDataFrame containing the masked polygons.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> mask = create_polygon_mask(gdf=gdf, step_size=100, crop_gdf=True)
        >>> mask
            geometry
        0   POLYGON ((2651470.877 2135999.353, 2661470.877...
        1   POLYGON ((2651470.877 2145999.353, 2661470.877...
        2   POLYGON ((2651470.877 2155999.353, 2661470.877...


    """

    # Checking that the gdf is of type GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError('gdf must be provided as GeoDataFrame')

    # Checking that the step_size is of type int
    if not isinstance(step_size, int):
        raise TypeError('The step_size must be provided as int')

    # Checking that crop_gdf is of type bool
    if not isinstance(crop_gdf, bool):
        raise TypeError('crop_gdf must be provided as bool')

    # Creating arrays
    x = np.arange(gdf.total_bounds[0],
                  gdf.total_bounds[2], step_size)

    y = np.arange(gdf.total_bounds[1],
                  gdf.total_bounds[3], step_size)

    # Creating polygons
    polygons = [Polygon([(a, b),
                         (a + step_size, b),
                         (a + step_size, b + step_size),
                         (a, b + step_size)]) for a,
                                                  b in
                product(x, y)]

    # Converting polygons to GeoDataFrame
    gdf_mask = gpd.GeoDataFrame(geometry=polygons,
                                crs=gdf.crs)

    # Dropping duplicate cells
    gdf_mask = gdf_mask.drop_duplicates(ignore_index=True)

    # Cropping the gdf if crop_gdf is True
    if crop_gdf:
        gdf_mask = gdf_mask.sjoin(gdf).reset_index()[['geometry']]

    return gdf_mask


def vectorize_raster(path: str) -> gpd.GeoDataFrame:
    """Vectorize Raster.

    Parameters
    ___________
        path : str
            Path to raster file, e.g. ``path='raster.tif'``

    Returns
    ________
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the Polygons of the vectorized raster.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf = vectorize_raster(path='raster.tif')
        >>> gdf
            geometry                                            class
        0   POLYGON ((4038305.864 3086142.360, 4038305.864...   0.292106
        1   POLYGON ((4038405.844 3086142.360, 4038405.844...   41.289803
        2   POLYGON ((4038505.823 3086142.360, 4038505.823...   61.701653
    """

    # Checking that the path is provided as str
    if not isinstance(path, str):
        raise TypeError('The path must be provided as string')

    with rasterio.open(path) as src:
        data = src.read(1,
                        masked=True)

        # Use a generator instead of a list
        shape_gen = ((shape(s),
                      v) for s,
                             v in shapes(data,
                                         transform=src.transform))

        # or build a dict from unpacked shapes
        gdf = gpd.GeoDataFrame(dict(zip(["geometry",
                                         "class"],
                                        zip(*shape_gen))),
                               crs=src.crs)

    return gdf


def create_outline(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create outline from GeoDataFrame.

    Parameters
    ___________
        gdf : gpd.GeoDataFrame
            GeoDataFrame holding the Heat Demand Data.

    Returns
    ________
        outline : gpd.GeoDataFrame
            Outline GeoDataFrame.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf = processing.create_outline(gdf=gdf)
        >>> gdf
            geometry
        0   POLYGON ((3744005.190 2671457.082, 3744005.190...

    """

    # Checking that the gdf is of type GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError('The gdf must be provided as GeoDataFrame')

    return gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)],
                            crs=gdf.crs)


def calculate_hd(hd_gdf: gpd.GeoDataFrame,
                 mask_gdf: gpd.GeoDataFrame,
                 hd_data_column: str = ''):
    """Calculate Heat Demand.

    Parameters
    __________
        hd_gdf : gpd.GeoDataFrame
            Heat demand data as GeoDataFrame.
        mask_gdf : gpd.GeoDataFrame
            Mask for the output Heat Demand Data.
        hd_data_column : str
            Name of the column that contains the Heat Demand Data, e.g. ``hd_data_column='HD'``.

    Returns
    _______

        gdf_hd : gpd.GeoDataFrame
            Output GeoDataFrame with Heat Demand Data.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_hd = processing.calculate_hd(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column='HD')
        >>> gdf_hd
            HD           geometry
        0   111.620963   POLYGON ((3726770.877 2671399.353, 3726870.877...
        1   142.831789   POLYGON ((3726770.877 2671499.353, 3726870.877...
        2   20.780601    POLYGON ((3726770.877 2671699.353, 3726870.877...

    """

    # Checking that the hd_gdf is of type GeoDataFrame
    if not isinstance(hd_gdf, gpd.GeoDataFrame):
        raise TypeError('The heat demand gdf must be provided as GeoDataFrame')

    # Checking that the HD Data Column is in the HD GeoDataFrame
    if not hd_data_column in hd_gdf:
        raise ValueError('%s is not a column in the GeoDataFrame' % hd_data_column)

    # Checking that the mask_gdf is of type GeoDataFrame
    if not isinstance(mask_gdf, gpd.GeoDataFrame):
        raise TypeError('The mask gdf must be provided as GeoDataFrame')

    # Checking that the Heat Demand Data Column is provided as string
    if not isinstance(hd_data_column, str):
        raise TypeError('The heat demand data column must be provided as string')

    # Reprojecting Data if necessary
    if mask_gdf.crs != hd_gdf.crs:
        hd_gdf = hd_gdf.to_crs(mask_gdf.crs)

    if any(shapely.get_type_id(hd_gdf.geometry) == 6):
        hd_gdf = hd_gdf.explode(index_parts=True).reset_index(drop=True)

    if all(shapely.get_type_id(hd_gdf.geometry) == 3):
        # Assigning area of original geometries to GeoDataFrame
        hd_gdf['area'] = hd_gdf.area

    elif all(shapely.get_type_id(hd_gdf.geometry) == 1):
        # Assigning length of original geometries to GeoDataFrame
        hd_gdf['length'] = hd_gdf.length

    # Overlaying Heat Demand Data with Mask
    overlay = gpd.overlay(df1=hd_gdf,
                          df2=mask_gdf)
    if all(shapely.get_type_id(hd_gdf.geometry) == 3):
        # Assigning area of splitted geometries to GeoDataFrame
        overlay['area_new'] = overlay.area

        # Calculating the share of the original Heat Demand for each splitted geometry
        overlay['HD'] = overlay[hd_data_column] * overlay['area_new'] / overlay['area']

    elif all(shapely.get_type_id(hd_gdf.geometry) == 1):
        # Assigning length of splitted geometries to GeoDataFrame
        overlay['length_new'] = overlay.length

        # Calculating the share of the original Heat Demand for each splitted geometry
        overlay['HD'] = overlay[hd_data_column] * overlay['length_new'] / overlay['length']

    elif all(shapely.get_type_id(hd_gdf.geometry) == 0):
        overlay['HD'] = overlay[hd_data_column]

    # Assigning centroid as geometry for spatial join
    overlay['geometry'] = overlay.centroid

    # Spatial join of overlay and mask
    leftjoin_gdf = gpd.sjoin(left_df=overlay,
                             right_df=mask_gdf,
                             how='left')

    # Adding the heat demand for each raster cell
    gdf_grouped = (leftjoin_gdf.groupby('index_right')['HD'].sum())

    # Concatenating cut polygons with mask polygons
    gdf_hd = pd.concat([gdf_grouped,
                        mask_gdf],
                       axis=1)

    # Creating GeoDataFrame
    gdf_hd = gpd.GeoDataFrame(geometry=gdf_hd['geometry'],
                              data=gdf_hd,
                              crs=mask_gdf.crs)

    # Filling NaNs
    gdf_hd.dropna(inplace=True)

    # Dropping duplicate values
    gdf_hd = gdf_hd.drop_duplicates()

    # Resetting index
    gdf_hd = gdf_hd.reset_index().drop('index',
                                       axis=1)

    return gdf_hd


def rasterize_gdf_hd(gdf_hd: gpd.GeoDataFrame,
                     path_out: str,
                     crs: Union[str, pyproj.crs.crs.CRS] = 'EPSG:3034',
                     xsize: int = 100,
                     ysize: int = 100):
    """Rasterize Heat Demand GeoDataFrame and save as raster.

    Parameters
    ___________
        gdf_hd : GeoDataFrame
            GeoDataFrame with Heat Demand Data.
        path_out : str
            Output file path for the heat demand raster, e.g. ``path_out='raster.tif'``.
        crs : str, pyproj.crs.crs.CRS, default: ``'EPSG:3034'``
            Output coordinate reference system, e.g. ``crs='EPSG:3034'``.
        xsize : int, default: ``100``
            Cell size of the output raster, e.g. ``xsize=100``.
        ysize : int, default: ``100``
            Cell size of the output raster, e.g. ``ysize=100``.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> rasterize_gdf_hd(gdf_hd=gdf_hd, path_out='raster.tif', crs='EPSG:3034', xsize=100, ysize=100)

    """

    # Checking that the gdf_hd if of type GeoDataFrame
    if not isinstance(gdf_hd, gpd.GeoDataFrame):
        raise TypeError('The gdf_hd must be provided as GeoDataFrame')

    # Checking that the output path is of type string
    if not isinstance(path_out, str):
        raise TypeError('The output path must be provided as string')

    # Checking that the CRS is provided as string or Pyproj CRS
    if not isinstance(crs, (str, pyproj.crs.crs.CRS)):
        raise TypeError('The CRS must be provided as string or PyProj CRS')

    # Checking that the xsize is of type int
    if not isinstance(xsize, int):
        raise TypeError('The xsize must be provided as int')

    # Checking that the ysize is of type int
    if not isinstance(ysize, int):
        raise TypeError('The ysize must be provided as int')

    # Creating array with the length of polygons in x and y direction
    x = np.arange(gdf_hd.total_bounds[0], gdf_hd.total_bounds[2], xsize)
    y = np.arange(gdf_hd.total_bounds[1], gdf_hd.total_bounds[3], ysize)

    # Creating matrix
    matrix = np.zeros(len(y) * len(x)).reshape(len(y),
                                               len(x))
    # Creating transform
    transform = rasterio.transform.from_origin(x[0], y[1], xsize, -ysize)

    # Saving mask raster
    with rasterio.open(
            path_out.split('.tif')[0] + '_temp.tif',
            'w',
            driver='GTiff',
            height=matrix.shape[0],
            width=matrix.shape[1],
            count=1,
            dtype=matrix.dtype,
            crs=crs,
            transform=transform,
            nodata=-9999
    ) as dst:
        dst.write(matrix, 1)

    # Copy meta data
    rst = rasterio.open(path_out.split('.tif')[0] + '_temp.tif')
    meta = rst.meta.copy()
    meta.update(compress='lzw')

    # Rasterisation of the quadratic-polygon-shapefile using the rasterize-function from rasterio
    with rasterio.open(path_out, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where the code creates a generator of geom, value pairs (geometry and HD_new) to use in rasterizing
        shapes = [(geom, value) for geom, value in zip(gdf_hd['geometry'], gdf_hd['HD'])]

        burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)

    # Closing and deleting dataset
    rst.close()
    os.remove(path_out.split('.tif')[0] + '_temp.tif')
    out.close()


def obtain_coordinates_from_addresses(df: pd.DataFrame,
                                      street_column: str,
                                      house_number_column: str,
                                      postal_code_column: str,
                                      location_column: str,
                                      output_crs: Union[str, pyproj.crs.crs.CRS]) -> gpd.GeoDataFrame:
    """Obtain coordinates from building addresses.

    Parameters
    ___________
        df : pd.DataFrame
            DataFrame containing the address data.
        street_column : str
            Name for the column containing the street name, e.g. ``street_column='Street'``.
        house_number_column : str
            Name for the column containing the house number, e.g. ``house_number_column='Number'``.
        postal_code_column : str
            Name for the column containing the postal code, e.g. ``postal_code_column='Postal Code'``.
        location_column : str
            Name for the column containing the location name, e.g. ``location_column='City'``.
        output_crs : str, pyproj.crs.crs.CRS
            Output coordinate reference system, e.g. ``crs='EPSG:3034'``

    Returns
    ________
        gdf : gpd.GeoDataFrame
            Output GeoDataFrame containing the Coordinates of the street addresses.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_addresses = obtain_coordinates_from_addresses(df=df, street_column='Street',
        ... house_number_column='Number', postal_code_column='Postal Code', location_column='City',
        ... output_crs='EPSG:3034')
        >>> gdf_addresses
            Unnamed: 0         HeatDemand     Street        Number   Postal Code    City    address                     geometry
        0   0                  431905.208696  Rathausplatz  1        59174          Kamen   Rathausplatz 1 59174 Kamen  POINT (3843562.447 2758094.896)
        1   1                  1858.465217    Rathausplatz  1        59174          Kamen   Rathausplatz 1 59174 Kamen  POINT (3843562.447 2758094.896)
        2   2                  28594.673913   Rathausplatz  4        59174          Kamen   Rathausplatz 4 59174 Kamen  POINT (3843569.733 2758193.784)

    """

    # Checking that the address DataFrame is of type DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Addresses must be provided as Pandas DataFrame')

    # Checking that the column is of type string
    if not isinstance(street_column, str):
        raise TypeError('Column names must be provided as string')

    # Checking that the column is of type string
    if not isinstance(house_number_column, str):
        raise TypeError('Column names must be provided as string')

    # Checking that the column is of type string
    if not isinstance(postal_code_column, str):
        raise TypeError('Column names must be provided as string')

    # Checking that the column is of type string
    if not isinstance(location_column, str):
        raise TypeError('Column names must be provided as string')

    # Checking that the output crs is of type string or PyProj CRS
    if not isinstance(output_crs, (str, pyproj.crs.crs.CRS)):
        raise TypeError('The output CRS must be provided as string or PyProj CRS')

    # Converting the data types of the columns
    df = df.astype({street_column: 'str',
                    house_number_column: 'str',
                    postal_code_column: 'str',
                    location_column: 'str'})

    # Modifying the addresses
    df['address'] = df[[street_column, house_number_column, postal_code_column, location_column]].apply(
        lambda x: ' '.join(x), axis=1)

    # Extracting the coordinates from the addresses
    coordinates = [geopy.geocoders.Nominatim(user_agent=df['address'].iloc[i]).geocode(df['address'].iloc[i]) for i in
                   tqdm(range(len(df)))]

    # Creating GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([coordinates[i][1][1] for i in range(len(coordinates))],
                                                       [coordinates[i][1][0] for i in range(len(coordinates))],
                                                       crs='EPSG:4326'),
                           data=df).to_crs(output_crs)

    return gdf


def get_building_footprint(point: shapely.geometry.Point,
                           dist: int) -> gpd.GeoDataFrame:
    """Get Building footprint from Shapely Point.

    Parameters
    ___________
        point : shapely.geometry.Point
            Point that corresponds to a building. CRS must be 'EPSG:4326', e.g. ``point=Point(6.54, 51.23)``.
        dist : int
            Distance around the point to get features, e.g. ``dist=25``.

    Returns
    ________
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the building footprint.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_building = get_building_footprint(point=Point(6.54, 51.23), dist=25)
        >>> gdf_building
             element_type  osmid    nodes                                              addr:city  addr:housenumber   addr:postcode  addr:street   amenity
        0    way           60170820 [747404971, 1128780263, 1128780085, 1128780530...  Kamen      1                  59174          Rathausplatz  townhall

    """

    # Checking that the point is a Shapely Point
    if not isinstance(point, shapely.geometry.Point):
        raise TypeError('Point must be provided as Shapely Point')

    # Checking that the distance is provided as int
    if not isinstance(dist, int):
        raise TypeError('Distance must be provided as int')

    try:
        gdf = osmnx.features.features_from_point(center_point=(list(point.coords)[0][1],
                                                               list(point.coords)[0][0]),
                                                 tags={'building': True},
                                                 dist=dist)
    except:
        gdf = None

    return gdf


def get_building_footprints(points: gpd.GeoDataFrame,
                            dist: int,
                            perform_sjoin: bool = True):
    """Get Building footprints from GeoDataFrame.

    Parameters
    ___________
        points : gpd.GeoDataFrame
            GeoDataFrame containing the Points.
        dist : int
            Distance around the points to get features, e.g. ``dist=25``.
        perform_sjoin : bool, default: ``True``
            Boolean to perform a spatial join to filter out additional buildings, e.g. ``perform_sjoin=True``.

    Returns
    ________
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the building footprints.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_buildings = get_building_footprints(points=gdf_addresses, dist=25)
        >>> gdf_buildings
             element_type  osmid    nodes                                              addr:city  addr:housenumber   addr:postcode  addr:street   amenity
        0    way           60170820 [747404971, 1128780263, 1128780085, 1128780530...  Kamen      1                  59174          Rathausplatz  townhall
        1    way           60170821 [747405971, 1128781263, 1128784085, 1128786530...  Kamen      5                  59174          Rathausplatz  townhall

    """

    # Checking that the points are provided as GeoDataFrame
    if not isinstance(points, gpd.GeoDataFrame):
        raise TypeError('Points must be provided as GeoDataFrame')

    # Checking that the distance is provided as int
    if not isinstance(dist, int):
        raise TypeError('Distance must be provided as int')

    # Checking that the perform_sjoin is provided as bool
    if not isinstance(perform_sjoin, bool):
        raise TypeError('perform_sjoin must be provided as bool')

    # Reprojecting GeoDataFrame
    if points.crs != 'EPSG:4326':
        crs = points.crs
        points = points.to_crs('EPSG:4326')
    else:
        crs = 'EPSG:4326'

    # Getting GeoDataFrames
    gdfs = [get_building_footprint(points['geometry'].iloc[i], dist=dist) for i in tqdm(range(len(points)))]

    # Concatenate GeoDataFrames
    gdf = pd.concat(gdfs)

    # Filtering Buildings
    if perform_sjoin:
        gdf = gpd.sjoin(gdf, points).reset_index()

    # Reprojecting GeoDataFrame to original CRS
    gdf = gdf.to_crs(crs)

    return gdf


def merge_rasters(file_names: list,
                  path_out: str) -> rasterio.io.DatasetReader:
    """Merge rasters.

    Parameters
    ___________
        file_names : list
            List of file names, e.g. ``file_names=['raster1.tif', 'raster2.tif']``.
        path_out : str
            Output path for merged raster, e.g. ``path_out='raster_merged.tif'``.

    Returns
    ________
        mosaic : rasterio.io.DatasetReader
            Merged raster.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> merge_rasters(file_names=['raster1.tif', 'raster2.tif'], path_out='raster_merged.tif')

    """

    # Checking that the file names are stored in a list
    if not isinstance(file_names, list):
        raise TypeError('The file names must be provided as list')

    # Opening Files
    files = [rasterio.open(path) for path in file_names]

    # Creating Mosaic
    mosaic, out_trans = merge(files)

    # Copying Meta Data
    out_meta = files[0].meta.copy()

    # Updating Meta Data
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": files[0].crs
                     }
                    )

    # Removing existing file
    os.remove(path_out)

    # Saving file
    with rasterio.open(path_out,
                       "w",
                       **out_meta) as dest:
        dest.write(mosaic)

    # Closing file
    dest.close()

    print('Raster successfully merged')


def calculate_zonal_stats(path_mask: str,
                          path_raster: str,
                          crs: Union[str, pyproj.crs.crs.CRS],
                          calculate_heated_area: bool = True
                          ) -> gpd.GeoDataFrame:
    """Calculate zonal statistics and return GeoDataFrame.

    Parameters
    ___________
        path_mask : str
            Path to the mask for the zonal statistics, e.g. ``path_mask='mask.shp'``.
        path_raster : str
            Path to the raster for the zonal statistics, e.g. ``path_raster='raster.tif'``.
        crs : str, pyproj.crs.crs.CRS
            Coordinate Reference System to pass to the GeoDataFrame, e.g. ``crs='EPSG:3034'``.
        calculate_heated_area : bool, default: ``True``
            Boolean value to calculate the heated area, e.g. ``calculate_heated_area=True``.

    Returns
    ________
        gdf : gpd.GeoDataFrame
            Output GeoDataFrame containing the input geometries and the output zonal statistics.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_stats = calculate_zonal_stats(path_mask='mask.shp', path_raster='raster.tif', crs='EPSG:3034')
        >>> gdf_stats
            geometry                                           min          max          std        median      Area (planimetric) Total Heat Demand   Average Heat demand per unit area    Share of Total HD [%]  Share of Total Area [%]   Heated Area   Share of Heated Area [%]
        0   POLYGON ((3854043.358 2686588.658, 3854042.704...  3.024974e-06 21699.841028 351.107975 88.114117   7.471599e+09       4.689531e+07        206.001944                           21.437292              23.485618                 2.276161e+09  30.464174
        1   POLYGON ((3922577.630 2751867.434, 3922590.877...  6.662710e-08 40566.944918 265.277509 46.066755   6.086689e+09       2.959064e+07        134.484551                           13.526791              19.132405                 2.200020e+09  36.144783
        2   MULTIPOLYGON (((3815551.417 2711668.010, 38155...  3.148388e-06 71665.631370 382.872868 106.194020  6.866552e+09       5.063581e+07        217.321986                           23.147186              21.583762                 2.329694e+09  33.928151

    """

    # Checking that the path to the mask is of type string
    if not isinstance(path_mask, str):
        raise TypeError('The path to the mask must be provided as string')

    # Checking that the path to the raster is of type string
    if not isinstance(path_raster, str):
        raise TypeError('The path to the raster must be provided as string')

    # Checking that the CRS is of type string or a pyproj CRS object
    if not isinstance(crs, (str, pyproj.crs.crs.CRS)):
        raise TypeError('The CRS must be provided as string or pyproj object')

    # Checking that the boolean value for calculate_heated_area is a boolean
    if not isinstance(calculate_heated_area, bool):
        raise TypeError('calculate_heatead_area value must be provided as bool')

    # Calculating zonal statistics
    stats = zonal_stats(vectors=path_mask,
                        raster=path_raster,
                        stats="count min mean max median sum std",
                        geojson_out=True)

    # Converting zonal statistics to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(stats)

    # Calculating total heat demand
    total_hd = sum(gdf['sum'])

    # Calculating total area
    total_area = sum(gdf.area)

    # Assigning the area of the Polygons to the DataFrame
    # NB: GeoPandas calculated the planimetric area; for larger regions, the ellipsoidal area should be calculated
    gdf['Area (planimetric)'] = gdf.area

    # Calculating the total heat demand per geometry
    gdf['Total Heat Demand'] = gdf['sum']

    # Calculating the average heat demand per unit area
    gdf['Average Heat demand per unit area'] = gdf['mean']

    # Calculating share of total heat demand for every geometry
    gdf['Share of Total HD [%]'] = gdf['sum']*100/total_hd

    # Calculating share of total area for every geometry
    gdf['Share of Total Area [%]'] = gdf.area*100/total_area

    if calculate_heated_area:
        # Opening raster to get resolution
        raster = rasterio.open(path_raster)

        # Calculating for heated area
        gdf['Heated Area'] = gdf['count']*raster.res[0]*raster.res[1]

        # Calculating share for heated area
        gdf['Share of Heated Area [%]'] = gdf['Heated Area']*100/gdf.area

    # Adding CRS manually as it is not passed from rasterstats,
    # see also https://github.com/perrygeo/python-rasterstats/issues/295
    gdf.crs = crs

    # Dropping columns
    gdf = gdf.drop(['sum', 'mean', 'count'], axis=1)

    return gdf
