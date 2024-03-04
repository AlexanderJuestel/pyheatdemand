"""
Contributors: Alexander Jüstel, Elias Khashfe, Eileen Herbst

"""

import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import Polygon, shape, box, Point, LineString
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


def create_polygon_mask(
    gdf: Union[gpd.GeoDataFrame, Polygon],
    step_size: int,
    crop_gdf: bool = False,
    crs: Union[str, pyproj.crs.crs.CRS] = None,
) -> gpd.GeoDataFrame:
    """Create a mask GeoDataFrame consisting of squares with a defined step_size.

    Parameters
    ----------
        gdf : Union[gpd.GeoDataFrame, shapely.geometry.Polygon]
            GeoDataFrame/Polygon over which a mask is created.
        step_size : int
            Size of the rasterized squares in meters, e.g. ``step_size=100``.
        crop_gdf : bool, default: ``False``
            Boolean to either crop the GeoDataFrame to the outline or return it as is, e.g. ``crop_gdf=False``.
        crs : Union[str, pyproj.crs.crs.CRS], default: ``None``
            Coordinate Reference System when providing Shapely Polygons as input.

    Returns
    --------
        gdf_mask : gpd.GeoDataFrame
            GeoDataFrame containing the masked polygons.

            ========== ===============================
            Index      Index of each mask polygon
            geometry   Geometry of each mask polygon
            ========== ===============================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> mask = create_polygon_mask(gdf=gdf, step_size=100, crop_gdf=True)
        >>> mask

        =======  ===================================================
        Index    geometry
        =======  ===================================================
        0        POLYGON ((2651470.877 2135999.353, 2661470.877...
        1        POLYGON ((2651470.877 2145999.353, 2661470.877...
        2        POLYGON ((2651470.877 2155999.353, 2661470.877...
        =======  ===================================================

    See Also
    ________


    """
    # Converting Polygon to GeoDataFrame
    if isinstance(gdf, Polygon):
        if not isinstance(crs, (str, pyproj.crs.crs.CRS)):
            raise TypeError("CRS must be provided as string or pyproj CRS object")
        gdf = gpd.GeoDataFrame(geometry=[gdf], crs=crs)

    # Checking that the gdf is of type GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be provided as GeoDataFrame")

    # Checking that the step_size is of type int
    if not isinstance(step_size, int):
        raise TypeError("The step_size must be provided as int")

    # Checking that crop_gdf is of type bool
    if not isinstance(crop_gdf, bool):
        raise TypeError("crop_gdf must be provided as bool")

    # Creating arrays
    x = np.arange(gdf.total_bounds[0], gdf.total_bounds[2], step_size)

    y = np.arange(gdf.total_bounds[1], gdf.total_bounds[3], step_size)

    # Creating polygons
    polygons = [
        Polygon(
            [
                (a, b),
                (a + step_size, b),
                (a + step_size, b + step_size),
                (a, b + step_size),
            ]
        )
        for a, b in product(x, y)
    ]

    # Converting polygons to GeoDataFrame
    gdf_mask = gpd.GeoDataFrame(geometry=polygons, crs=gdf.crs)

    # Dropping duplicate cells
    gdf_mask = gdf_mask.drop_duplicates(ignore_index=True)

    # Cropping the gdf if crop_gdf is True
    if crop_gdf:
        gdf_mask = gdf_mask.sjoin(gdf).reset_index()[["geometry"]]

    return gdf_mask


def refine_mask(
    mask: gpd.GeoDataFrame,
    data: gpd.GeoDataFrame,
    num_of_points: int,
    cell_size: int,
    area_limit: Union[float, int] = None,
) -> gpd.GeoDataFrame:
    """Refine polygon mask.

    Parameters
    __________
        mask : gpd.GeoDataFrame
            Original mask.
        data : gpd.GeoDataFrame
            Heat demand data, usually polygons.
        num_of_points : int
            Number of points that need to be in one cell for refinement, e.g. ``num_of_points=100``.
        cell_size : int
            Cell size of the new cells, cell size should be a divisor of the original cell size, e.g. ``cell_size=1000``.
        area_limit : Union[float, int]
            For multiple refinements, the area limit can be defined to only further refine already refined cells, e.g. ``area_limit=10000``.

    Returns
    _______
        grid_refs : gpd.GeoDataFrame
            GeoDataFrame containing the refined mask polygons. Data columns are as follows:

            ========== ===============================
            Index      Index of each mask polygon
            geometry   Geometry of each mask polygon
            ========== ===============================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    .. versionadded:: 0.0.9

    Examples
    ________

        >>> mask = refine_mask(mask=mask, data=data, num_of_points=100, cell_size=1000)
        >>> mask

        =======  ===================================================
        Index    geometry
        =======  ===================================================
        0        POLYGON ((2651470.877 2135999.353, 2661470.877...
        1        POLYGON ((2651470.877 2145999.353, 2661470.877...
        2        POLYGON ((2651470.877 2155999.353, 2661470.877...
        =======  ===================================================

    """
    # Checking that the mask is of type GeoDataFrame
    if not isinstance(mask, gpd.GeoDataFrame):
        raise TypeError("Mask must be provided as GeoPandas GeoDataFrame.")

    # Checking that the data is of type GeoDataFrame
    if not isinstance(data, gpd.GeoDataFrame):
        raise TypeError("Input data must be provided as GeoPandas GeoDataFrame.")

    # Checking that the number of points is of type int
    if not isinstance(num_of_points, int):
        raise TypeError("Number of points must be provided as int.")

    # Checking that the cell size is of type int
    if not isinstance(cell_size, int):
        raise TypeError("Cell size must be provided as int.")

    # Checking that the area limit is of type float
    if area_limit:
        if not isinstance(area_limit, (float, int)):
            raise TypeError("Area limit must be provided as type float.")

    # Only select already redefined polygons for further refinement
    mask_above_limit = pd.DataFrame()
    if area_limit is not None:
        mask_above_limit = mask[mask.area > area_limit].reset_index(drop=True)
        mask = mask[mask.area <= area_limit].reset_index(drop=True)

    # Create centroids of geometries
    data["geometry"] = data.centroid

    # Join data with grid
    grid_joined = gpd.sjoin(left_df=data, right_df=mask)

    # Count number of points per polygon
    df_value_counts = pd.DataFrame(data=grid_joined["index_right"].value_counts())

    # Filter df_value_counts by threshold number of data points within one Polygon
    df_sel = df_value_counts[df_value_counts["count"] >= num_of_points]

    # Select Polygons from mask
    grid_sel = mask.iloc[df_sel.index].sort_index().reset_index(drop=True)

    # Create masks within selected polygons
    grid_ref = pd.concat(
        [
            create_polygon_mask(
                gdf=grid_sel.iloc[i]["geometry"], step_size=cell_size, crs=mask.crs
            )
            for i in range(len(grid_sel))
        ]
    ).reset_index(drop=True)

    # Drop old polygons
    grid_dropped = mask.drop(index=df_sel.index).reset_index(drop=True)

    # Merge GeoDataFrames
    grid_refs = pd.concat([mask_above_limit, grid_dropped, grid_ref]).reset_index(
        drop=True
    )

    return grid_refs


def quad_tree_mask_refinement(
    mask: gpd.GeoDataFrame,
    data: gpd.GeoDataFrame,
    max_depth: int = 4,
    num_of_points: Union[int, list] = 100,
) -> gpd.GeoDataFrame:
    """Quad Tree Mask Refinement.

    Parameters
    __________
        mask : gpd.GeoDataFrame
            Original mask.
        data : gpd.GeoDataFrame
            Heat demand data, usually polygons.
        max_depth : int, default: ``4``
            Number of refinements, e.g. ``max_depth=4``.
        num_of_points : Union[int, list], default: ``100``
            Number of points that need to be in one cell for refinement, e.g. ``num_of_points=100``.

    Returns
    _______
        grid_refs : gpd.GeoDataFrame
            GeoDataFrame containing the refined mask polygons. Data columns are as follows:

            ========== ===============================
            Index      Index of each mask polygon
            geometry   Geometry of each mask polygon
            ========== ===============================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    .. versionadded:: 0.0.9

    Examples
    ________

        >>> mask = quad_tree_mask_refinement(mask=mask, data=data, max_depth=4, num_of_points=[150, 150, 100, 50])
        >>> mask

        =======  ===================================================
        Index    geometry
        =======  ===================================================
        0        POLYGON ((2651470.877 2135999.353, 2661470.877...
        1        POLYGON ((2651470.877 2145999.353, 2661470.877...
        2        POLYGON ((2651470.877 2155999.353, 2661470.877...
        =======  ===================================================

    """
    # Checking that the mask is of type GeoDataFrame
    if not isinstance(mask, gpd.GeoDataFrame):
        raise TypeError("The mask must be provided as GeoDataFrame.")

    # Checking that the data is of type GeoDataFrame
    if not isinstance(data, gpd.GeoDataFrame):
        raise TypeError("The data must be provided as GeoDataFrame.")

    # Checking that the max_depth is of type int
    if not isinstance(max_depth, int):
        raise TypeError("The max_depth must be provided as int.")

    # Checking that the number of points is either an integer or a list
    if not isinstance(num_of_points, (int, list)):
        raise TypeError(
            "The number of points must be provided as integer or list of integers."
        )

    # Getting the original size of the cells
    original_cell_size = np.sqrt(mask.iloc[0].geometry.area)

    # Creating list of points if only a single point is provided
    if isinstance(num_of_points, int):
        num_of_points = [num_of_points] * max_depth

    for i in range(max_depth):
        # Setting the area limit
        if i == 0:
            area_limit = None
        else:
            area_limit = int(original_cell_size / 2**i) * int(original_cell_size / 2**i)

        # Refining mask
        mask = refine_mask(
            mask=mask,
            data=data,
            num_of_points=num_of_points[i],
            cell_size=int(original_cell_size / 2 ** (i + 1)),
            area_limit=area_limit,
        )
    return mask


def vectorize_raster(path: str, merge_polygons: bool = True) -> gpd.GeoDataFrame:
    """Vectorize Raster.

    Parameters
    ___________
        path : str
            Path to raster file, e.g. ``path='raster.tif'``.
        merge_polygons : bool, default: ``True``
            Boolean to state if the polygons should be merged or if every single pixel should be return as polygon,
            e.g. ``merge_polygons=True``.

    Returns
    ________
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the Polygons of the vectorized raster. Data columns are as follows:

            ========== ===============================
            Index      Index of each raster cell
            geometry   Geometry of each raster cell
            class      Value of each raster cell
            ========== ===============================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    .. versionchanged:: 0.0.9

    Examples
    ________

        >>> gdf = vectorize_raster(path='raster.tif')
        >>> gdf

        =======  =================================================== ===========
        Index    geometry                                            class
        =======  =================================================== ===========
        0        POLYGON ((4038305.864 3086142.360, 4038305.864...   0.292106
        1        POLYGON ((4038405.844 3086142.360, 4038405.844...   41.289803
        2        POLYGON ((4038505.823 3086142.360, 4038505.823...   61.701653
        =======  =================================================== ===========

    """
    # Checking that the path is provided as str
    if not isinstance(path, str):
        raise TypeError("The path must be provided as string")

    # Checking that merge_polygon is provided as bool
    if not isinstance(merge_polygons, bool):
        raise TypeError("merge_polygons must be either True or False")

    # Opening raster
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)

        # Adding small random value to raster value to split pixels into separate polygons
        if not merge_polygons:
            generator = np.random.default_rng(42)  # set seed number for reproducibility
            val = generator.uniform(0, 0.001, size=data.shape).astype(np.float32)
            data = data + val

        # Using a generator instead of a list
        shape_gen = ((shape(s), v) for s, v in shapes(data, transform=src.transform))

        # or build a dict from unpacked shapes
        gdf = gpd.GeoDataFrame(
            dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs
        )

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
            Outline GeoDataFrame. Data columns are as follows:

            ==========  =============================
            Index       Index of the outline
            geometry    Geometry of the outline
            ==========  =============================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf = processing.create_outline(gdf=gdf)
        >>> gdf

        ======= ===================================================
        Index   geometry
        ======= ===================================================
        0       POLYGON ((3744005.190 2671457.082, 3744005.190...
        ======= ===================================================

    """
    # Checking that the gdf is of type GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("The gdf must be provided as GeoDataFrame")

    return gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=gdf.crs)


def _check_hd_input(
    hd_gdf: gpd.GeoDataFrame,
    mask_gdf: Union[gpd.GeoDataFrame, Polygon],
    hd_data_column: str = "",
):
    """Check heat demand input data.

    Parameters
    __________
        hd_gdf : gpd.GeoDataFrame
            Heat demand data as GeoDataFrame.
        mask_gdf : Union[gpd.GeoDataFrame, shapely.geometry.Polygon]
            Mask for the output Heat Demand Data.
        hd_data_column : str, default: ``''``
            Name of the column that contains the Heat Demand Data, e.g. ``hd_data_column='HD'``.

    Returns
    _______
        hd_gdf : gpd.GeoDataFrame
            Heat demand data as GeoDataFrame.
        mask_gdf : Union[gpd.GeoDataFrame, shapely.geometry.Polygon]
            Mask for the output Heat Demand Data.
        hd_data_column : str, default: ``''``
            Name of the column that contains the Heat Demand Data, e.g. ``hd_data_column='HD'``.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    """
    # Converting Shapely Polygon to GeoDataFrame
    if isinstance(mask_gdf, Polygon):
        mask_gdf = gpd.GeoDataFrame(geometry=[mask_gdf], crs=hd_gdf.crs)

    # Checking that the hd_gdf is of type GeoDataFrame
    if not isinstance(hd_gdf, gpd.GeoDataFrame):
        raise TypeError("The heat demand gdf must be provided as GeoDataFrame")

    # Checking that the mask_gdf is of type GeoDataFrame
    if not isinstance(mask_gdf, gpd.GeoDataFrame):
        raise TypeError("The mask gdf must be provided as GeoDataFrame")

    # Checking that the Heat Demand Data Column is provided as string
    if not isinstance(hd_data_column, str):
        raise TypeError("The heat demand data column must be provided as string")

    # Checking that the HD Data Column is in the HD GeoDataFrame
    if hd_data_column not in hd_gdf:
        raise ValueError("%s is not a column in the GeoDataFrame" % hd_data_column)

    # Reprojecting Data if necessary
    if mask_gdf.crs != hd_gdf.crs:
        hd_gdf = hd_gdf.to_crs(mask_gdf.crs)

    # Exploding MultiPolygons
    if any(shapely.get_type_id(hd_gdf.geometry) == 6):
        hd_gdf = hd_gdf.explode(index_parts=True).reset_index(drop=True)

    # Assigning area to Polygons
    if all(shapely.get_type_id(hd_gdf.geometry) == 3):
        # Assigning area of original geometries to GeoDataFrame
        hd_gdf["area"] = hd_gdf.area

    # Assigning lengths to LineStrings
    elif all(shapely.get_type_id(hd_gdf.geometry) == 1):
        # Assigning length of original geometries to GeoDataFrame
        hd_gdf["length"] = hd_gdf.length

    return hd_gdf, mask_gdf, hd_data_column


def calculate_hd(
    hd_gdf: gpd.GeoDataFrame,
    mask_gdf: Union[gpd.GeoDataFrame, Polygon],
    hd_data_column: str = "",
) -> gpd.GeoDataFrame:
    """Calculate Heat Demand.

    Parameters
    __________
        hd_gdf : gpd.GeoDataFrame
            Heat demand data as GeoDataFrame.
        mask_gdf : Union[gpd.GeoDataFrame, shapely.geometry.Polygon]
            Mask for the output Heat Demand Data.
        hd_data_column : str, default: ``''``
            Name of the column that contains the Heat Demand Data, e.g. ``hd_data_column='HD'``.

    Returns
    _______

        gdf_hd : gpd.GeoDataFrame
            Output GeoDataFrame with Heat Demand Data. Data columns are as follows:

            ============ ==================================
            Index        Index of each heat demand cell
            HD           Heat demand of each cell
            geometry     Geometry of the heat demand cell
            ....         Other columns
            ============ ==================================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_hd = processing.calculate_hd(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column='HD')
        >>> gdf_hd

        =======  ==============  ===================================================
        Index    HD              geometry
        =======  ==============  ===================================================
        0        111.620963      POLYGON ((3726770.877 2671399.353, 3726870.877...
        1        142.831789      POLYGON ((3726770.877 2671499.353, 3726870.877...
        2        20.780601       POLYGON ((3726770.877 2671699.353, 3726870.877...
        =======  ==============  ===================================================

    See Also
    ________
        calculate_hd_sindex : Calculate Heat Demand using Spatial Indices.

    """
    # Checking input data
    hd_gdf, mask_gdf, hd_data_column = _check_hd_input(
        hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column=hd_data_column
    )

    # Overlaying Heat Demand Data with Mask
    overlay = gpd.overlay(df1=hd_gdf, df2=mask_gdf)

    # Calculate HD for Polygons
    if all(shapely.get_type_id(hd_gdf.geometry) == 3):
        # Assigning area of splitted geometries to GeoDataFrame
        overlay["area_new"] = overlay.area

        # Calculating the share of the original Heat Demand for each splitted geometry
        overlay["HD"] = overlay[hd_data_column] * overlay["area_new"] / overlay["area"]

    #  Calculate HD for LineStrings
    elif all(shapely.get_type_id(hd_gdf.geometry) == 1):
        # Assigning length of splitted geometries to GeoDataFrame
        overlay["length_new"] = overlay.length

        # Calculating the share of the original Heat Demand for each splitted geometry
        overlay["HD"] = (
            overlay[hd_data_column] * overlay["length_new"] / overlay["length"]
        )

    # Calculate HD for Points
    elif all(shapely.get_type_id(hd_gdf.geometry) == 0):
        overlay["HD"] = overlay[hd_data_column]

    # Assigning centroid as geometry for spatial join
    overlay["geometry"] = overlay.centroid

    # Spatial join of overlay and mask
    leftjoin_gdf = gpd.sjoin(left_df=overlay, right_df=mask_gdf, how="left")

    # Adding the heat demand for each raster cell
    gdf_grouped = leftjoin_gdf.groupby("index_right")["HD"].sum()

    # Concatenating cut polygons with mask polygons
    gdf_hd = pd.concat([gdf_grouped, mask_gdf], axis=1)

    # Creating GeoDataFrame
    gdf_hd = gpd.GeoDataFrame(
        geometry=gdf_hd["geometry"], data=gdf_hd, crs=mask_gdf.crs
    )

    # Filling NaNs
    gdf_hd.dropna(inplace=True)

    # Dropping duplicate values
    gdf_hd = gdf_hd.drop_duplicates()

    # Resetting index
    gdf_hd = gdf_hd.reset_index().drop("index", axis=1)

    return gdf_hd


def calculate_hd_sindex(
    hd_gdf: gpd.GeoDataFrame,
    mask_gdf: Union[gpd.GeoDataFrame, Polygon],
    hd_data_column: str = "",
) -> gpd.GeoDataFrame:
    """Calculate Heat Demand using Spatial Indices.

    Parameters
    __________
        hd_gdf : gpd.GeoDataFrame
            Heat demand data as GeoDataFrame.
        mask_gdf : Union[gpd.GeoDataFrame, shapely.geometry.Polygon]
            Mask for the output Heat Demand Data.
        hd_data_column : str, default: ``''``
            Name of the column that contains the Heat Demand Data, e.g. ``hd_data_column='HD'``.

    Returns
    _______
        gdf_hd : gpd.GeoDataFrame
            Output GeoDataFrame with Heat Demand Data. Data columns are as follows:

            ============ ==================================
            Index        Index of each heat demand cell
            HD           Heat demand of each cell
            geometry     Geometry of the heat demand cell
            ....         Other columns
            ============ ==================================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_hd = processing.calculate_hd_sindex(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column='HD')
        >>> gdf_hd

        =======  ==============  ===================================================
        Index    HD              geometry
        =======  ==============  ===================================================
        0        111.620963      POLYGON ((3726770.877 2671399.353, 3726870.877...
        1        142.831789      POLYGON ((3726770.877 2671499.353, 3726870.877...
        2        20.780601       POLYGON ((3726770.877 2671699.353, 3726870.877...
        =======  ==============  ===================================================

    """
    # Checking input data
    hd_gdf, mask_gdf, hd_data_column = _check_hd_input(
        hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column=hd_data_column
    )

    # Querying spatial index
    grid_ix, buildings_ix = hd_gdf.sindex.query(mask_gdf.geometry, predicate=None)

    # Getting heat demand per mask cell
    heat_per_grid_cell = (
        hd_gdf[hd_data_column].iloc[buildings_ix].groupby(grid_ix).sum()
    )

    # Creating GeoDataFrame
    gdf_hd = mask_gdf.iloc[pd.DataFrame(heat_per_grid_cell).index]

    # Assigning Heat Demand Values
    gdf_hd["HD"] = heat_per_grid_cell.values

    # Resetting index
    gdf_hd = gdf_hd.reset_index().drop("index", axis=1)

    return gdf_hd


def rasterize_gdf_hd(
    gdf_hd: gpd.GeoDataFrame,
    path_out: str,
    crs: Union[str, pyproj.crs.crs.CRS] = "EPSG:3034",
    xsize: int = 100,
    ysize: int = 100,
    flip_raster: bool = True,
):
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
        flip_raster : bool, default: ``True``
            Boolean value to flip the raster.

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
        raise TypeError("The gdf_hd must be provided as GeoDataFrame")

    # Checking that the output path is of type string
    if not isinstance(path_out, str):
        raise TypeError("The output path must be provided as string")

    # Checking that the CRS is provided as string or Pyproj CRS
    if not isinstance(crs, (str, pyproj.crs.crs.CRS)):
        raise TypeError("The CRS must be provided as string or PyProj CRS")

    # Checking that the xsize is of type int
    if not isinstance(xsize, int):
        raise TypeError("The xsize must be provided as int")

    # Checking that the ysize is of type int
    if not isinstance(ysize, int):
        raise TypeError("The ysize must be provided as int")

    # Checking that the flip_raster variable is of type bool
    if not isinstance(flip_raster, int):
        raise TypeError("The flip_raster value must be provided as bool")

    # Creating array with the length of polygons in x and y direction
    x = np.arange(gdf_hd.total_bounds[0], gdf_hd.total_bounds[2], xsize)
    y = np.arange(gdf_hd.total_bounds[1], gdf_hd.total_bounds[3], ysize)

    # Creating matrix
    matrix = np.zeros(len(y) * len(x)).reshape(len(y), len(x))
    # Creating transform
    if flip_raster:
        transform = rasterio.transform.from_origin(x[0], y[-1], xsize, ysize)
    else:
        transform = rasterio.transform.from_origin(x[0], y[0], xsize, -ysize)

    # Saving mask raster
    with rasterio.open(
        path_out.split(".tif")[0] + "_temp.tif",
        "w",
        driver="GTiff",
        height=matrix.shape[0],
        width=matrix.shape[1],
        count=1,
        dtype=matrix.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(matrix, 1)

    # Copy meta data
    rst = rasterio.open(path_out.split(".tif")[0] + "_temp.tif")
    meta = rst.meta.copy()
    meta.update(compress="lzw")

    # Rasterization of the quadratic-polygon-shapefile using the rasterize-function from rasterio
    with rasterio.open(path_out, "w+", **meta) as out:
        out_arr = out.read(1)

        # this is where the code creates a generator of geom, value pairs (geometry and HD_new) to use in rasterizing
        shapes = list(zip(gdf_hd["geometry"], gdf_hd["HD"]))

        burned = rasterio.features.rasterize(
            shapes=shapes, fill=0, out=out_arr, transform=out.transform
        )
        out.write_band(1, burned)

    # Closing and deleting dataset
    rst.close()
    os.remove(path_out.split(".tif")[0] + "_temp.tif")
    out.close()


def obtain_coordinates_from_addresses(
    df: pd.DataFrame,
    street_column: str,
    house_number_column: str,
    postal_code_column: str,
    location_column: str,
    output_crs: Union[str, pyproj.crs.crs.CRS],
) -> gpd.GeoDataFrame:
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
            Output GeoDataFrame containing the Coordinates of the street addresses. Data columns are as follows:

            ============== ==============================
            Index          Index of each address
            Unnamed: 0     Index of each address
            HeatDemand     Heat Demand of each address
            Street         Street name of each address
            Number         House number of each address
            Postal Code    Postal code of each address
            City           City of each address
            address        Address
            geometry       Geometry of each address
            ============== ==============================

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

        ======== ================   ============== ============= ======== =============  ======  =========================== =================================
        Index    Unnamed: 0         HeatDemand     Street        Number   Postal Code    City    address                     geometry
        ======== ================   ============== ============= ======== =============  ======  =========================== =================================
        0        0                  431905.208696  Rathausplatz  1        59174          Kamen   Rathausplatz 1 59174 Kamen  POINT (3843562.447 2758094.896)
        1        1                  1858.465217    Rathausplatz  1        59174          Kamen   Rathausplatz 1 59174 Kamen  POINT (3843562.447 2758094.896)
        2        2                  28594.673913   Rathausplatz  4        59174          Kamen   Rathausplatz 4 59174 Kamen  POINT (3843569.733 2758193.784)
        ======== ================   ============== ============= ======== =============  ======  =========================== =================================

    """
    # Checking that the address DataFrame is of type DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Addresses must be provided as Pandas DataFrame")

    # Checking that the column is of type string
    if not isinstance(street_column, str):
        raise TypeError("Column names must be provided as string")

    # Checking that the column is of type string
    if not isinstance(house_number_column, str):
        raise TypeError("Column names must be provided as string")

    # Checking that the column is of type string
    if not isinstance(postal_code_column, str):
        raise TypeError("Column names must be provided as string")

    # Checking that the column is of type string
    if not isinstance(location_column, str):
        raise TypeError("Column names must be provided as string")

    # Checking that the output crs is of type string or PyProj CRS
    if not isinstance(output_crs, (str, pyproj.crs.crs.CRS)):
        raise TypeError("The output CRS must be provided as string or PyProj CRS")

    # Converting the data types of the columns
    df = df.astype(
        {
            street_column: "str",
            house_number_column: "str",
            postal_code_column: "str",
            location_column: "str",
        }
    )

    # Modifying the addresses
    df["address"] = df[
        [street_column, house_number_column, postal_code_column, location_column]
    ].apply(lambda x: " ".join(x), axis=1)

    # Extracting the coordinates from the addresses
    coordinates = [
        geopy.geocoders.Nominatim(user_agent=df["address"].iloc[i]).geocode(
            df["address"].iloc[i]
        )
        for i in tqdm(range(len(df)))
    ]

    # Creating GeoDataFrame
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [coordinates[i][1][1] for i in range(len(coordinates))],
            [coordinates[i][1][0] for i in range(len(coordinates))],
            crs="EPSG:4326",
        ),
        data=df,
    ).to_crs(output_crs)

    return gdf


def get_building_footprint(
    point: shapely.geometry.Point, dist: int
) -> gpd.GeoDataFrame:
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
            GeoDataFrame containing the building footprint. Data columns are as follows:

            =================  ===============================================
            Index              Index of the building footprint
            element_type       Element type of bulding gootprint
            osmid              OpenStreetMap ID number
            nodes              Nodes of the building footprint
            addr:city          City where the building footprint is located
            addr:housenumber   Housenumber of the building footpring
            addr:postcode      Post code of the building footprint
            addr:street        Street of the building footprint
            amenity            Feature of the building footprint
            geometry           Geometry of the building footprint
            =================  ===============================================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_building = get_building_footprint(point=Point(6.54, 51.23), dist=25)
        >>> gdf_building

        ========  ============= ======== ================================================== ========== ================== ============== ============= ==========
        Index     element_type  osmid    nodes                                              addr:city  addr:housenumber   addr:postcode  addr:street   amenity
        ========  ============= ======== ================================================== ========== ================== ============== ============= ==========
        0         way           60170820 [747404971, 1128780263, 1128780085, 1128780530...  Kamen      1                  59174          Rathausplatz  townhall
        ========  ============= ======== ================================================== ========== ================== ============== ============= ==========

    """

    # Checking that the point is a Shapely Point
    if not isinstance(point, shapely.geometry.Point):
        raise TypeError("Point must be provided as Shapely Point")

    # Checking that the distance is provided as int
    if not isinstance(dist, int):
        raise TypeError("Distance must be provided as int")

    try:
        gdf = osmnx.features.features_from_point(
            center_point=(list(point.coords)[0][1], list(point.coords)[0][0]),
            tags={"building": True},
            dist=dist,
        )
    except:
        gdf = gpd.GeoDataFrame(columns=["id", "geometry"], geometry="geometry")

    return gdf


def get_building_footprints(
    points: gpd.GeoDataFrame, dist: int, perform_sjoin: bool = True
) -> gpd.GeoDataFrame:
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
            GeoDataFrame containing the building footprints. Data columns are as follows:

            =================  ===============================================
            Index              Index of the building footprint
            element_type       Element type of bulding gootprint
            osmid              OpenStreetMap ID number
            nodes              Nodes of the building footprint
            addr:city          City where the building footprint is located
            addr:housenumber   Housenumber of the building footpring
            addr:postcode      Post code of the building footprint
            addr:street        Street of the building footprint
            amenity            Feature of the building footprint
            geometry           Geometry of the building footprint
            =================  ===============================================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_buildings = get_building_footprints(points=gdf_addresses, dist=25)
        >>> gdf_buildings

        ========  ============= ======== ================================================== ========== ================== ============== ============= ==========
        Index     element_type  osmid    nodes                                              addr:city  addr:housenumber   addr:postcode  addr:street   amenity
        ========  ============= ======== ================================================== ========== ================== ============== ============= ==========
        0         way           60170820 [747404971, 1128780263, 1128780085, 1128780530...  Kamen      1                  59174          Rathausplatz  townhall
        1         way           60170821 [747405971, 1128781263, 1128784085, 1128786530...  Kamen      5                  59174          Rathausplatz  townhall
        ========  ============= ======== ================================================== ========== ================== ============== ============= ==========

    """
    # Checking that the points are provided as GeoDataFrame
    if not isinstance(points, gpd.GeoDataFrame):
        raise TypeError("Points must be provided as GeoDataFrame")

    # Checking that the distance is provided as int
    if not isinstance(dist, int):
        raise TypeError("Distance must be provided as int")

    # Checking that the perform_sjoin is provided as bool
    if not isinstance(perform_sjoin, bool):
        raise TypeError("perform_sjoin must be provided as bool")

    # Reprojecting GeoDataFrame
    if points.crs != "EPSG:4326":
        crs = points.crs
        points = points.to_crs("EPSG:4326")
    else:
        crs = "EPSG:4326"

    # Getting GeoDataFrames
    gdfs = [
        get_building_footprint(points["geometry"].iloc[i], dist=dist)
        for i in tqdm(range(len(points)))
    ]

    # Concatenate GeoDataFrames
    gdf = pd.concat(gdfs)

    # Filtering Buildings
    if perform_sjoin:
        gdf = gpd.sjoin(gdf, points).reset_index()

    # Reprojecting GeoDataFrame to original CRS
    gdf = gdf.to_crs(crs)

    return gdf


def merge_rasters(file_names: list, path_out: str) -> rasterio.io.DatasetReader:
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
        raise TypeError("The file names must be provided as list")

    # Opening Files
    files = [rasterio.open(path) for path in file_names]

    # Creating Mosaic
    mosaic, out_trans = merge(files)

    # Copying Meta Data
    out_meta = files[0].meta.copy()

    # Updating Meta Data
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": files[0].crs,
        }
    )

    # Removing existing file
    os.remove(path_out)

    # Saving file
    with rasterio.open(path_out, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Closing file
    dest.close()

    print("Raster successfully merged")


def calculate_zonal_stats(
    path_mask: str,
    path_raster: str,
    crs: Union[str, pyproj.crs.crs.CRS],
    calculate_heated_area: bool = True,
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

            ===================================== ===================================================================
            Index                                 Index of each geometry
            geometry                              Geometry of the input Vector Data Set
            min                                   Minimum Heat Demand value within the geometry
            max                                   Maximum Heat Demand value within the geometry
            std                                   Standard Deviation of the Heat Demand values within the geometry
            median                                Median Heat Demand value within the geometry
            Area (planimetric)                    Area of the geometry
            Total Heat Demand                     Total Heat Demand of the geometry
            Average Heat demand per unit area     Average Heat Demand per unit area
            Share of Total HD [%]                 Share of the total Heat Demand of this geometry
            Share of Total Area [%]               Share of the total area of this geometry
            Heated Area                           Area that actually contains heat demand values within the geometry
            Share of Heated Area [%]              Share of the area that actually contains heat demand values
            ===================================== ===================================================================


    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    Examples
    ________

        >>> gdf_stats = calculate_zonal_stats(path_mask='mask.shp', path_raster='raster.tif', crs='EPSG:3034')
        >>> gdf_stats

        =======  ====================================================  =============  ============== ============ ============  ==================== ==================    ===================================  ====================== ========================  ============= ==========================
        Index    geometry                                              min            max            std          median        Area (planimetric)   Total Heat Demand     Average Heat demand per unit area    Share of Total HD [%]  Share of Total Area [%]   Heated Area   Share of Heated Area [%]
        =======  ====================================================  =============  ============== ============ ============  ==================== ==================    ===================================  ====================== ========================  ============= ==========================
        0        POLYGON ((3854043.358 2686588.658, 3854042.704...     3.024974e-06   21699.841028   351.107975   88.114117     7.471599e+09         4.689531e+07          206.001944                           21.437292              23.485618                 2.276161e+09  30.464174
        1        POLYGON ((3922577.630 2751867.434, 3922590.877...     6.662710e-08   40566.944918   265.277509   46.066755     6.086689e+09         2.959064e+07          134.484551                           13.526791              19.132405                 2.200020e+09  36.144783
        2        MULTIPOLYGON (((3815551.417 2711668.010, 38155...     3.148388e-06   71665.631370   382.872868   106.194020    6.866552e+09         5.063581e+07          217.321986                           23.147186              21.583762                 2.329694e+09  33.928151
        =======  ====================================================  =============  ============== ============ ============  ==================== ==================    ===================================  ====================== ========================  ============= ==========================

    """
    # Checking that the path to the mask is of type string
    if not isinstance(path_mask, str):
        raise TypeError("The path to the mask must be provided as string")

    # Checking that the path to the raster is of type string
    if not isinstance(path_raster, str):
        raise TypeError("The path to the raster must be provided as string")

    # Checking that the CRS is of type string or a pyproj CRS object
    if not isinstance(crs, (str, pyproj.crs.crs.CRS)):
        raise TypeError("The CRS must be provided as string or pyproj object")

    # Checking that the boolean value for calculate_heated_area is a boolean
    if not isinstance(calculate_heated_area, bool):
        raise TypeError("calculate_heatead_area value must be provided as bool")

    # Calculating zonal statistics
    stats = zonal_stats(
        vectors=path_mask,
        raster=path_raster,
        stats="count min mean max median sum std",
        geojson_out=True,
    )

    # Converting zonal statistics to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(stats)

    # Calculating total heat demand
    total_hd = sum(gdf["sum"])

    # Calculating total area
    total_area = sum(gdf.area)

    # Assigning the area of the Polygons to the DataFrame
    # NB: GeoPandas calculated the planimetric area; for larger regions, the ellipsoidal area should be calculated
    gdf["Area (planimetric)"] = gdf.area

    # Calculating the total heat demand per geometry
    gdf["Total Heat Demand"] = gdf["sum"]

    # Calculating the average heat demand per unit area
    gdf["Average Heat demand per unit area"] = gdf["mean"]

    # Calculating share of total heat demand for every geometry
    gdf["Share of Total HD [%]"] = gdf["sum"] * 100 / total_hd

    # Calculating share of total area for every geometry
    gdf["Share of Total Area [%]"] = gdf.area * 100 / total_area

    if calculate_heated_area:
        # Opening raster to get resolution
        raster = rasterio.open(path_raster)

        # Calculating for heated area
        gdf["Heated Area"] = gdf["count"] * raster.res[0] * raster.res[1]

        # Calculating share for heated area
        gdf["Share of Heated Area [%]"] = gdf["Heated Area"] * 100 / gdf.area

    # Adding CRS manually as it is not passed from rasterstats,
    # see also https://github.com/perrygeo/python-rasterstats/issues/295
    gdf.crs = crs

    # Dropping columns
    gdf = gdf.drop(["sum", "mean", "count"], axis=1)

    return gdf


def create_connection(
    linestring: shapely.geometry.LineString, point: shapely.geometry.Point
) -> shapely.geometry.LineString:
    """Create LineString between Point and LineString.

    Parameters
    __________
        linestring : shapely.geometry.LineString
            LineString representing a street segment.
        point : shapely.geometry.Point
            Point representing the centroid of a building footprint, e.g. ``point= POINT(100, 100)``.

    Returns
    _______
        linestring_connection : shapely.geometry.LineString
            Shortest connection between the centroid of a building footprint and a street segment.

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    .. versionadded:: 0.0.9

    Examples
    ________
        >>> linestring_connection = processing.create_connection(linestring=linestring, point=point)
        >>> linestring_connection.wkt
        'LINESTRING (292607.59635341103 5627766.391411121, 292597.58816236566 5627776.171055705)'

    """
    # Checking that the street segment is of type LineString
    if not isinstance(linestring, shapely.geometry.LineString):
        raise TypeError("Street segment must be provided as Shapely LineString")

    # Checking that the building footprint is represented by a point
    if not isinstance(point, shapely.geometry.Point):
        raise TypeError("Building footprint must be provided as Shapely Point")

    # Find distance on line that is closest to the point
    projected_point = linestring.project(point)

    # Find point at the distance on the line that is closest to the point
    interpolated_point = linestring.interpolate(projected_point)

    # Create LineString from interpolated point and original point
    linestring_connection = LineString([interpolated_point, point])

    return linestring_connection


def create_connections(
    gdf_buildings: gpd.GeoDataFrame,
    gdf_roads: gpd.GeoDataFrame,
    hd_data_column: str = None,
) -> gpd.GeoDataFrame:
    """Create LineString between Points and LineStrings.

    Parameters
    __________
        gdf_buildings : gpd.GeoDataFrame
            GeoDataFrame holding the building footprints.
        gdf_roads : gpd.GeoDataFrame
            GeoDataFrame holding the street segments.
        hd_data_column : str, default: ``None``
            Name of the column that contains the Heat Demand Data, e.g. ``hd_data_column='HD'``.

    Returns
    _______
        gdf_connections : gpd.GeoDataFrame
            GeoDataFrame holding the connections between the houses and the street segments. Data columns are as follows:

            ==========  =============================
            Index       Index of each connection
            geometry    Geometry of each connection
            ==========  =============================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    .. versionadded:: 0.0.9

    Examples
    ________
        >>> gdf_connections = create_connections(gdf_buildings=buildings, gdf_roads=roads)
        >>> gdf_connections

        ======= ===================================================
        Index   geometry
        ======= ===================================================
        0	    LINESTRING (292726.502 5627866.823, 292705.144...
        1	    LINESTRING (292725.657 5627862.826, 292705.613...
        2	    LINESTRING (292726.502 5627866.823, 292705.144...
        ======= ===================================================

    """
    # Checking that gdf_building is of type GeoDataFrame
    if not isinstance(gdf_buildings, gpd.GeoDataFrame):
        raise TypeError("Building footprints must be provided as GeoDataFrame")

    # Checking that gdf_roads is of type GeoDataFrame
    if not isinstance(gdf_roads, gpd.GeoDataFrame):
        raise TypeError("Road segments must be provided as GeoDataFrame")

    # Getting centroids of the buildings
    gdf_buildings["geometry"] = gdf_buildings.centroid

    # Performing spatial join
    gdf_joined = gpd.sjoin_nearest(gdf_buildings, gdf_roads)

    # Creating connections between building footprints and road segments
    linestrings_connections = [
        create_connection(
            linestring=gdf_roads.iloc[row["index_right"]].geometry,
            point=row["geometry"],
        )
        for index, row in gdf_joined.iterrows()
    ]

    # Creating GeoDataFrame from list of LineStrings
    gdf_connections = gpd.GeoDataFrame(
        geometry=linestrings_connections, crs=gdf_roads.crs
    )

    if hd_data_column:

        # Checking that the Heat Demand Data Column is provided as string
        if not isinstance(hd_data_column, str):
            raise TypeError("The heat demand data column must be provided as string")

        # Checking that the HD Data Column is in the HD GeoDataFrame
        if hd_data_column not in gdf_buildings:
            raise ValueError("%s is not a column in the GeoDataFrame" % hd_data_column)

        gdf_connections[hd_data_column] = gdf_joined.reset_index()[hd_data_column]

    return gdf_connections


def calculate_hd_street_segments(
    gdf_buildings: gpd.GeoDataFrame, gdf_roads: gpd.GeoDataFrame, hd_data_column: str
) -> gpd.GeoDataFrame:
    """Calculate heat demand for street segments based on the heat demand of the nearest houses.

    Parameters
    ----------
        gdf_buildings : gpd.GeoDataFrame
            GeoDataFrame holding the building footprints.
        gdf_roads : gpd.GeoDataFrame
            GeoDataFrame holding the street segments.
        hd_data_column : str
            Name of the column that contains the Heat Demand Data, e.g. ``hd_data_column='HD'``.

    Returns
    -------
        gdf_hd : gpd.GeoDataFrame
            GeoDataFrame consisting of the street segments and the cumulated heat demand. Data columns are as follows:

            ============== ========================================
            Index          Index of each heat demand cell
            HD_normalized  Heat demand of each LineString
            geometry       Geometry of the heat demand LineString
            ....           Other columns
            ============== ========================================

    Raises
    ______
        TypeError
            If the wrong input data types are provided.

    .. versionadded:: 0.0.9

    Examples
    ________
        >>> gdf_hd = calculate_hd_street_segments(gdf_buildings=buildings, gdf_roads=roads, hd_data_column='HD')
        >>> gdf_hd

        =======  ==============  =====================================================
        Index    HD_normalized   geometry
        =======  ==============  =====================================================
        0        111.620963      LINESTRING ((3726770.877 2671399.353, 3726870.877...
        1        142.831789      LINESTRING ((3726770.877 2671499.353, 3726870.877...
        2        20.780601       LINESTRING ((3726770.877 2671699.353, 3726870.877...
        =======  ==============  =====================================================


    """
    # Checking that gdf_building is of type GeoDataFrame
    if not isinstance(gdf_buildings, gpd.GeoDataFrame):
        raise TypeError("Building footprints must be provided as GeoDataFrame")

    # Checking that gdf_roads is of type GeoDataFrame
    if not isinstance(gdf_roads, gpd.GeoDataFrame):
        raise TypeError("Road segments must be provided as GeoDataFrame")

    # Checking that the Heat Demand Data Column is provided as string
    if not isinstance(hd_data_column, str):
        raise TypeError("The heat demand data column must be provided as string")

    # Checking that the HD Data Column is in the HD GeoDataFrame
    if hd_data_column not in gdf_buildings:
        raise ValueError("%s is not a column in the GeoDataFrame" % hd_data_column)

    # Getting centroids of the buildings
    gdf_buildings["geometry"] = gdf_buildings.centroid

    # Performing spatial join
    gdf_joined = gpd.sjoin_nearest(gdf_buildings, gdf_roads)
    # Group Heat Demands
    heat_demands = gdf_joined.groupby("index_right")[hd_data_column].sum()

    # Concatenating data
    gdf_hd = pd.concat([heat_demands, gdf_roads], axis=1)

    # Creating GeoDataFrame
    gdf_hd = gpd.GeoDataFrame(
        geometry=gdf_hd["geometry"], data=gdf_hd, crs=gdf_roads.crs
    )

    # Assigning normalized heat demand
    gdf_hd["HD_normalized"] = gdf_hd[hd_data_column] / gdf_hd.length

    return gdf_hd


def convert_dtype(path_in: str, path_out: str):
    """Convert dtype of raster.

    Parameters
    ----------
        path_in : str
            Input path of the raster, e.g. ``path_in='input.tif'``.
        path_out : str
            Output path of the converted raster, e.g. ``path_out='output.tif'``

    Examples
    ________
        >>> processing.convert_dtype(path_in='input.tif', path_out='output.tif')

    .. versionadded:: 0.0.9
    """
    # Checking that the input path is of type string
    if not isinstance(path_in, str):
        raise TypeError("Input path must be provided as string")

    # Checking that the output path is of type string
    if not isinstance(path_out, str):
        raise TypeError("Output path must be provided as string")

    # Opening dataset
    with rasterio.open(path_in) as src:

        # Converting data
        band1 = src.read(1)
        resband = np.uint16(band1)

        # Editing meta data
        m = src.meta
        m["count"] = 1
        m["dtype"] = "uint16"

        # Saving converted dataset to file
        with rasterio.open(path_out, "w", **m) as dst:
            dst.write(resband, 1)
