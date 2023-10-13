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

    Parameters:
    ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame over which a mask is created.
        step_size : int
            Size of the rasterized squares in meters.
        crop_gdf : bool
            Boolean to either crop the GeoDataFrame to the outline or return it as is.

    Returns:
    --------
        gdf_mask : gpd.GeoDataFrame
            GeoDataFrame containing the masked polygons.

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

    if crop_gdf:
        gdf_mask = gdf_mask.sjoin(gdf).reset_index()[['geometry']]

    return gdf_mask


def vectorize_raster(path: str) -> gpd.GeoDataFrame:
    """Vectorize Raster.

    Parameters:
    ___________
        path : str
            Path to raster file.

    Returns:
    ________
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the Polygons of the vectorized raster.

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

    Parameters:
    ___________
        gdf : gpd.GeoDataFrame
            GeoDataFrame holding the Heat Demand Data.

    Returns:
    ________
        outline : gpd.GeoDataFrame
            Outline GeoDataFrame.
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

    Parameters:
    ___________
        hd_gdf : gpd.GeoDataFrame
            Heat demand data as GeoDataFrame.
        mask_gdf : gpd.GeoDataFrame
            Mask for the output Heat Demand Data.
        hd_data_str : str
            Name of the column that contains the Heat Demand Data.

    Returns:
    ________

        gdf_hd : gpd.GeoDataFrame
            Output GeoDataFrame with Heat Demand Data.
    """

    # Checking that the hd_gdf is of type GeoDataFrame
    if not isinstance(hd_gdf, gpd.GeoDataFrame):
        raise TypeError('The heat demand gdf must be provided as GeoDataFrame')

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
        hd_gdf = hd_gdf.explode().reset_index(drop=True)

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

    Parameters:
    ___________
        gdf_hd : GeoDataFrame
            GeoDataFrame with Heat Demand Data.
        path_out : str
            Output file path for the heat demand raster.
        crs : str, pyproj.crs.crs.CRS
            Output coordinate reference system.
        xsize : int
            Cell size of the output raster.
        ysize : int
            Cell size of the output raster.
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
    if not isinstance(xsize, int):
        raise TypeError('The ysize must be provided as int')

    # Creating array with the length of polygons in x and y direction
    x = np.arange(gdf_hd.total_bounds[0], gdf_hd.total_bounds[2], 100)
    y = np.arange(gdf_hd.total_bounds[1], gdf_hd.total_bounds[3], 100)

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

    Parameters:
    ___________
        df : pd.DataFrame
            DataFrame containing the address data.
        street_column : str
            Name for the column containing the street name.
        house_number_column : str
            Name for the column containing the house number.
        postal_code_column : str
            Name for the column containing the postal code.
        location_column : str
            Name for the column containing the location name.
        output_crs : str, pyproj.crs.crs.CRS
            Output coordinate reference system.

    Returns:
    ________
        gdf = gpd.GeoDataFrame
            Output GeoDataFrame containing the Coordinates of the street addresses.

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

    Parameters:
    ___________
        point : shapely.geometry.Point
            Point that corresponds to a building. CRS must be 'EPSG:4326'.
        dist : int
            Distance around the point to get features.

    Returns:
    ________
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the building footprint.

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

    Parameters:
    ___________
        points : gpd.GeoDataFrame
            GeoDataFrame containing the Points.
        dist : int
            Distance around the points to get features.
        perform_sjoin : bool
            Boolean to perform a spatial join to filter out additional buildings.

    Returns:
    ________
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the building footprints.

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

#def merge_rasters(rasters: list):
    #"""Merge rasters with different heat demand values.

    #:return:
    #"""
#def stitch_raster():

# def calculate_final_hd(hd_data: Union[rasterio.io.DatasetReader,
#                                      gpd.GeoDataFrame,
#                                      pd.DataFrame,
#                                      str],
#                       global_mask_data: gpd.GeoDataFrame,
#                       output_crs: Union[str,
#                                         pyproj.crs.crs.CRS],
#                       output_resolution: int,
#                       save_as_raster: bool = False) -> gpd.GeoDataFrame:
#    """Calculate Heat Demand for any kind of provided input data.
#
#    Parameters:
#    ___________
#        hd_data : Union[rasterio.io.DatasetReader, gpd.GeoDataFrame, pd.DataFrame, str]
#            Input Heat Demand Data
#        global_mask_data : gpd.GeoDataFrame
#            Global mask provided as GeoDataFrame
#        output_crs : Union[str, pyproj.crs.crs.CRS]
#            Output Coordinate Reference System
#        output_resolution : int
#            Cell size of the output
#        save_as_raster : bool
#            Boolean value to store the result as raster
#    Returns:
#         gdf_hd : gpd.GeoDataFrame
#            Resulting Heat Demand Data as GeoDataFrame
#    """

#    if not isinstance(hd_data, (rasterio.io.DatasetReader,
#                                      gpd.GeoDataFrame,
#                                      pd.DataFrame,
#                                      str)):
#        raise TypeError('Heat Demand Data must be provided as Raster data through rasterio, Vector data through '
#                        'GeoPandas, DataFrame through Pandas or as string')
#
#    if
