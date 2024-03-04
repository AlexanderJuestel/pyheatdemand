import pytest
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
import sys
import os

sys.path.insert(0, "../")


def test_create_polygon_mask():
    from pyheatdemand.processing import create_polygon_mask

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (0, 0),  # Bottom-left corner
                    (0, 100),  # Top-left corner
                    (100, 100),  # Top-right corner
                    (100, 0),  # Bottom-right corner
                    (0, 0),  # Close the polygon by repeating the first point
                ]
            )
        ],
        crs="EPSG:3034",
    )

    mask = create_polygon_mask(gdf=gdf, step_size=10, crop_gdf=False)

    assert isinstance(mask, gpd.GeoDataFrame)
    assert len(mask) == 100
    assert all(mask.area == 100) == True
    assert mask.crs == "EPSG:3034"

    mask = create_polygon_mask(gdf=gdf, step_size=10, crop_gdf=True)

    assert isinstance(mask, gpd.GeoDataFrame)
    assert len(mask) == 100
    assert all(mask.area == 100) == True
    assert mask.crs == "EPSG:3034"

    mask = create_polygon_mask(
        gdf=gdf.iloc[0].geometry, step_size=10, crop_gdf=True, crs="EPSG:3034"
    )

    assert isinstance(mask, gpd.GeoDataFrame)
    assert len(mask) == 100
    assert all(mask.area == 100) == True
    assert mask.crs == "EPSG:3034"


def test_create_polygon_mask_error():
    from pyheatdemand.processing import create_polygon_mask

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (0, 0),  # Bottom-left corner
                    (0, 100),  # Top-left corner
                    (100, 100),  # Top-right corner
                    (100, 0),  # Bottom-right corner
                    (0, 0),  # Close the polygon by repeating the first point
                ]
            )
        ],
        crs="EPSG:3034",
    )

    with pytest.raises(TypeError):
        create_polygon_mask(gdf=[gdf], step_size=10, crop_gdf=False)
    with pytest.raises(TypeError):
        create_polygon_mask(gdf=gdf, step_size=10.5, crop_gdf=False)
    with pytest.raises(TypeError):
        create_polygon_mask(gdf=gdf, step_size=10, crop_gdf="False")

    with pytest.raises(TypeError):
        create_polygon_mask(
            gdf=gdf.iloc[0].geometry, step_size=10, crop_gdf=True, crs=["EPSG:3034"]
        )


@pytest.mark.parametrize("path", ["data/Data_Type_I_Raster.tif"])
def test_vectorize_raster(path):
    from pyheatdemand.processing import vectorize_raster

    data = rasterio.open(path)

    gdf = vectorize_raster(path=path)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert data.crs == gdf.crs

    gdf = vectorize_raster(path=path, merge_polygons=False)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert data.crs == gdf.crs


@pytest.mark.parametrize("path", ["data/Data_Type_I_Raster.tif"])
def test_vectorize_raster_error(path):
    from pyheatdemand.processing import vectorize_raster

    with pytest.raises(TypeError):
        vectorize_raster(path=[path])

    with pytest.raises(TypeError):
        vectorize_raster(path=path, merge_polygons="False")


def test_create_outline():
    from pyheatdemand.processing import create_outline

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (0, 0),  # Bottom-left corner
                    (0, 100),  # Top-left corner
                    (100, 100),  # Top-right corner
                    (100, 0),  # Bottom-right corner
                    (0, 0),  # Close the polygon by repeating the first point
                ]
            )
        ],
        crs="EPSG:3034",
    )

    outline = create_outline(gdf=gdf)

    assert isinstance(outline, gpd.GeoDataFrame)
    assert outline.crs == gdf.crs


def test_create_outline_error():
    from pyheatdemand.processing import create_outline

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (0, 0),  # Bottom-left corner
                    (0, 100),  # Top-left corner
                    (100, 100),  # Top-right corner
                    (100, 0),  # Bottom-right corner
                    (0, 0),  # Close the polygon by repeating the first point
                ]
            )
        ],
        crs="EPSG:3034",
    )

    with pytest.raises(TypeError):
        create_outline(gdf=[gdf])


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_I_Vector_HD_Data.shp")]
)
def test_calculate_hd(mask_gdf, hd_gdf):
    from pyheatdemand.processing import calculate_hd

    gdf_hd = calculate_hd(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB")

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    hd_gdf_multi = gpd.GeoDataFrame(
        geometry=[MultiPolygon([hd_gdf.geometry.values])], crs="EPSG:3034"
    )
    hd_gdf_multi["WOHNGEB_WB"] = 100

    gdf_hd = calculate_hd(
        hd_gdf=hd_gdf_multi, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    gdf_hd = calculate_hd(
        hd_gdf=hd_gdf.to_crs("EPSG:25832"),
        mask_gdf=mask_gdf,
        hd_data_column="WOHNGEB_WB",
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    gdf_hd = calculate_hd(
        hd_gdf=hd_gdf, mask_gdf=mask_gdf.iloc[0].geometry, hd_data_column="WOHNGEB_WB"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_I_Vector_HD_Data.shp")]
)
def test_calculate_hd_points(mask_gdf, hd_gdf):
    from pyheatdemand.processing import calculate_hd

    hd_gdf["geometry"] = hd_gdf["geometry"].centroid
    gdf_hd = calculate_hd(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB")

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    gdf_hd = calculate_hd(
        hd_gdf=hd_gdf.to_crs("EPSG:25832"),
        mask_gdf=mask_gdf,
        hd_data_column="WOHNGEB_WB",
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_II_Vector_Lines.shp")]
)
def test_calculate_hd_lines(mask_gdf, hd_gdf):
    from pyheatdemand.processing import calculate_hd

    gdf_hd = calculate_hd(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="RW_WW_WBED")

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.to_crs("EPSG:3034").crs

    gdf_hd = calculate_hd(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="RW_WW_WBED")

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.to_crs("EPSG:3034").crs


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_I_Vector_HD_Data.shp")]
)
def test_calculate_hd_error(mask_gdf, hd_gdf):
    from pyheatdemand.processing import calculate_hd

    with pytest.raises(TypeError):
        gdf_hd = calculate_hd(
            hd_gdf=[hd_gdf], mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB"
        )
    with pytest.raises(TypeError):
        gdf_hd = calculate_hd(
            hd_gdf=hd_gdf, mask_gdf=[mask_gdf], hd_data_column="WOHNGEB_WB"
        )
    with pytest.raises(TypeError):
        gdf_hd = calculate_hd(
            hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column=["WOHNGEB_WB"]
        )

    with pytest.raises(ValueError):
        gdf_hd = calculate_hd(
            hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_W"
        )


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_I_Vector_HD_Data.shp")]
)
def test_rasterize_gdf_hd(mask_gdf, hd_gdf):
    from pyheatdemand.processing import rasterize_gdf_hd, calculate_hd

    gdf_hd = calculate_hd(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB")

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    try:
        os.remove("data/Data_Type_I_Vector.tif")
    except FileNotFoundError:
        pass

    rasterize_gdf_hd(
        gdf_hd=gdf_hd,
        path_out="data/Data_Type_I_Vector.tif",
        crs="EPSG:3034",
        xsize=100,
        ysize=100,
    )

    os.remove("data/Data_Type_I_Vector.tif")


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_I_Vector_HD_Data.shp")]
)
def test_rasterize_gdf_hd_error(mask_gdf, hd_gdf):
    from pyheatdemand.processing import rasterize_gdf_hd, calculate_hd

    gdf_hd = calculate_hd(hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB")

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    with pytest.raises(TypeError):
        rasterize_gdf_hd(
            gdf_hd=[gdf_hd],
            path_out="data/Data_Type_I_Vector.tif",
            crs="EPSG:3034",
            xsize=100,
            ysize=100,
        )
    with pytest.raises(TypeError):
        rasterize_gdf_hd(
            gdf_hd=gdf_hd,
            path_out=["data/Data_Type_I_Vector.tif"],
            crs="EPSG:3034",
            xsize=100,
            ysize=100,
        )
    with pytest.raises(TypeError):
        rasterize_gdf_hd(
            gdf_hd=gdf_hd,
            path_out="data/Data_Type_I_Vector.tif",
            crs=["EPSG:3034"],
            xsize=100,
            ysize=100,
        )
    with pytest.raises(TypeError):
        rasterize_gdf_hd(
            gdf_hd=gdf_hd,
            path_out="data/Data_Type_I_Vector.tif",
            crs="EPSG:3034",
            xsize=[100],
            ysize=100,
        )
    with pytest.raises(TypeError):
        rasterize_gdf_hd(
            gdf_hd=gdf_hd,
            path_out="data/Data_Type_I_Vector.tif",
            crs="EPSG:3034",
            xsize=100,
            ysize=[100],
        )


@pytest.mark.parametrize("df", [pd.read_csv("data/Data_Type_III_Point_Addresses.csv")])
def test_obtain_coordinates_from_addresses(df):
    from pyheatdemand.processing import obtain_coordinates_from_addresses

    df["PLZ"] = df["PLZ"].astype(int)
    df["Strasse"] = df["Strasse"].apply(
        lambda x: "".join(" " + char if char.isupper() else char for char in x).strip()
    )

    gdf = obtain_coordinates_from_addresses(
        df[:2], "Strasse", "Nummer", "PLZ", "Ort", "EPSG:3034"
    )

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == "EPSG:3034"


@pytest.mark.parametrize("df", [pd.read_csv("data/Data_Type_III_Point_Addresses.csv")])
def test_obtain_coordinates_from_addresses_error(df):
    from pyheatdemand.processing import obtain_coordinates_from_addresses

    df["PLZ"] = df["PLZ"].astype(int)
    df["Strasse"] = df["Strasse"].apply(
        lambda x: "".join(" " + char if char.isupper() else char for char in x).strip()
    )

    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(
            [df[:2]], "Strasse", "Nummer", "PLZ", "Ort", "EPSG:3034"
        )
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(
            df[:2], ["Strasse"], "Nummer", "PLZ", "Ort", "EPSG:3034"
        )
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(
            df[:2], "Strasse", ["Nummer"], "PLZ", "Ort", "EPSG:3034"
        )
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(
            df[:2], "Strasse", "Nummer", ["PLZ"], "Ort", "EPSG:3034"
        )
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(
            df[:2], "Strasse", "Nummer", "PLZ", ["Ort"], "EPSG:3034"
        )
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(
            df[:2], "Strasse", "Nummer", "PLZ", "Ort", ["EPSG:3034"]
        )


def test_get_building_footprint():
    from pyheatdemand.processing import get_building_footprint

    gdf = get_building_footprint(point=Point(6.07868, 50.77918), dist=25)

    assert isinstance(gdf, gpd.GeoDataFrame)

    gdf = get_building_footprint(point=Point(6.07868, 50.77918), dist=1)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 0


def test_get_building_footprint_error():
    from pyheatdemand.processing import get_building_footprint

    with pytest.raises(TypeError):
        get_building_footprint(point=[Point(6.07868, 50.77918)], dist=25)

    with pytest.raises(TypeError):
        get_building_footprint(point=Point(6.07868, 50.77918), dist=[25])


def test_get_building_footprints():
    from pyheatdemand.processing import get_building_footprints

    points = gpd.GeoDataFrame(geometry=[Point(6.07868, 50.77918)], crs="EPSG:4326")

    gdf = get_building_footprints(points=points, dist=250, perform_sjoin=True)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1

    gdf = get_building_footprints(points=points, dist=250, perform_sjoin=False)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) != 1

    gdf = get_building_footprints(
        points=points.to_crs("EPSG:3034"), dist=250, perform_sjoin=False
    )

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) != 1


def test_get_building_footprints_error():
    from pyheatdemand.processing import get_building_footprints

    points = gpd.GeoDataFrame(geometry=[Point(6.07868, 50.77918)], crs="EPSG:4326")

    with pytest.raises(TypeError):
        gdf = get_building_footprints(points=[points], dist=250, perform_sjoin=True)

    with pytest.raises(TypeError):
        gdf = get_building_footprints(points=points, dist=[250], perform_sjoin=True)

    with pytest.raises(TypeError):
        gdf = get_building_footprints(points=points, dist=250, perform_sjoin=[True])


def test_merge_rasters():
    from pyheatdemand.processing import merge_rasters

    raster_list = os.listdir("data/rasters/")
    raster_list = [
        os.path.join(os.path.abspath("data/rasters/"), path) for path in raster_list
    ]

    merge_rasters(raster_list, "data/Raster_merged.tif")


def test_merge_rasters_error():
    from pyheatdemand.processing import merge_rasters

    raster_list = os.listdir("data/rasters/")
    raster_list = [
        os.path.join(os.path.abspath("data/rasters/"), path) for path in raster_list
    ]

    with pytest.raises(TypeError):
        merge_rasters(raster_list[0], "data/Raster_merged.tif")

    with pytest.raises(TypeError):
        merge_rasters(raster_list, ["data/Raster_merged.tif"])


def test_calculate_zonal_stats():
    from pyheatdemand.processing import calculate_zonal_stats

    gdf_stats = calculate_zonal_stats(
        "data/nw_dvg_krs.shp",
        "data/HD_RBZ_Köln.tif",
        "EPSG:3034",
        calculate_heated_area=True,
    )

    assert isinstance(gdf_stats, gpd.GeoDataFrame)

    gdf_stats = calculate_zonal_stats(
        "data/nw_dvg_krs.shp",
        "data/HD_RBZ_Köln.tif",
        "EPSG:3034",
        calculate_heated_area=False,
    )

    assert isinstance(gdf_stats, gpd.GeoDataFrame)


def test_calculate_zonal_stats_error():
    from pyheatdemand.processing import calculate_zonal_stats

    with pytest.raises(TypeError):
        calculate_zonal_stats(
            ["data/nw_dvg_krs.shp"],
            "data/HD_RBZ_Köln.tif",
            "EPSG:3034",
            calculate_heated_area=False,
        )

    with pytest.raises(TypeError):
        calculate_zonal_stats(
            "data/nw_dvg_krs.shp",
            ["data/HD_RBZ_Köln.tif"],
            "EPSG:3034",
            calculate_heated_area=False,
        )

    with pytest.raises(TypeError):
        calculate_zonal_stats(
            "data/nw_dvg_krs.shp",
            "data/HD_RBZ_Köln.tif",
            ["EPSG:3034"],
            calculate_heated_area=False,
        )

    with pytest.raises(TypeError):
        calculate_zonal_stats(
            "data/nw_dvg_krs.shp",
            "data/HD_RBZ_Köln.tif",
            "EPSG:3034",
            calculate_heated_area="False",
        )


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_I_Vector_HD_Data.shp")]
)
def test_calculate_hd_sindex(mask_gdf, hd_gdf):
    from pyheatdemand.processing import calculate_hd_sindex

    gdf_hd = calculate_hd_sindex(
        hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    hd_gdf_multi = gpd.GeoDataFrame(
        geometry=[MultiPolygon([hd_gdf.geometry.values])], crs="EPSG:3034"
    )
    hd_gdf_multi["WOHNGEB_WB"] = 100

    gdf_hd = calculate_hd_sindex(
        hd_gdf=hd_gdf_multi, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    gdf_hd = calculate_hd_sindex(
        hd_gdf=hd_gdf.to_crs("EPSG:25832"),
        mask_gdf=mask_gdf,
        hd_data_column="WOHNGEB_WB",
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    gdf_hd = calculate_hd_sindex(
        hd_gdf=hd_gdf, mask_gdf=mask_gdf.iloc[0].geometry, hd_data_column="WOHNGEB_WB"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_I_Vector_HD_Data.shp")]
)
def test_calculate_hd_sindex_points(mask_gdf, hd_gdf):
    from pyheatdemand.processing import calculate_hd_sindex

    hd_gdf["geometry"] = hd_gdf["geometry"].centroid
    gdf_hd = calculate_hd_sindex(
        hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    gdf_hd = calculate_hd_sindex(
        hd_gdf=hd_gdf.to_crs("EPSG:25832"),
        mask_gdf=mask_gdf,
        hd_data_column="WOHNGEB_WB",
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_II_Vector_Lines.shp")]
)
def test_calculate_hd_sindex_lines(mask_gdf, hd_gdf):
    from pyheatdemand.processing import calculate_hd_sindex

    gdf_hd = calculate_hd_sindex(
        hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="RW_WW_WBED"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.to_crs("EPSG:3034").crs

    gdf_hd = calculate_hd_sindex(
        hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="RW_WW_WBED"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.to_crs("EPSG:3034").crs


@pytest.mark.parametrize(
    "mask_gdf", [gpd.read_file("data/Interreg_NWE_mask_500m_EPSG3034.shp")]
)
@pytest.mark.parametrize(
    "hd_gdf", [gpd.read_file("data/Data_Type_I_Vector_HD_Data.shp")]
)
def test_calculate_hd_sindex_error(mask_gdf, hd_gdf):
    from pyheatdemand.processing import calculate_hd_sindex

    with pytest.raises(TypeError):
        gdf_hd = calculate_hd_sindex(
            hd_gdf=[hd_gdf], mask_gdf=mask_gdf, hd_data_column="WOHNGEB_WB"
        )
    with pytest.raises(TypeError):
        gdf_hd = calculate_hd_sindex(
            hd_gdf=hd_gdf, mask_gdf=[mask_gdf], hd_data_column="WOHNGEB_WB"
        )
    with pytest.raises(TypeError):
        gdf_hd = calculate_hd_sindex(
            hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column=["WOHNGEB_WB"]
        )

    with pytest.raises(ValueError):
        gdf_hd = calculate_hd_sindex(
            hd_gdf=hd_gdf, mask_gdf=mask_gdf, hd_data_column="WOHNGEB_W"
        )


def test_refine_mask():
    from pyheatdemand.processing import refine_mask

    mask = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (0, 0),  # Bottom-left corner
                    (0, 100),  # Top-left corner
                    (100, 100),  # Top-right corner
                    (100, 0),  # Bottom-right corner
                    (0, 0),  # Close the polygon by repeating the first point
                ]
            )
        ],
        crs="EPSG:3034",
    )

    generator = np.random.default_rng(42)  # set seed number for reproducibility
    x = generator.integers(low=0, high=100, size=1000)

    y = generator.integers(low=0, high=100, size=1000)

    data = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=x, y=y, crs="EPSG:3034"))

    gdf_refined = refine_mask(
        mask=mask,
        data=data,
        num_of_points=10,
        cell_size=50,
    )

    gdf_refined = refine_mask(
        mask=gdf_refined, data=data, num_of_points=10, cell_size=50, area_limit=50 * 50
    )

    assert isinstance(gdf_refined, gpd.GeoDataFrame)


def test_refine_mask_error():
    from pyheatdemand.processing import refine_mask

    mask = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (0, 0),  # Bottom-left corner
                    (0, 100),  # Top-left corner
                    (100, 100),  # Top-right corner
                    (100, 0),  # Bottom-right corner
                    (0, 0),  # Close the polygon by repeating the first point
                ]
            )
        ],
        crs="EPSG:3034",
    )
    generator = np.random.default_rng(42)  # set seed number for reproducibility
    x = generator.integers(low=0, high=100, size=1000)

    y = generator.integers(low=0, high=100, size=1000)
    data = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=x, y=y, crs="EPSG:3034"))

    with pytest.raises(TypeError):
        refine_mask(
            mask=[mask],
            data=data,
            num_of_points=10,
            cell_size=50,
        )
    with pytest.raises(TypeError):
        refine_mask(
            mask=mask,
            data=[data],
            num_of_points=10,
            cell_size=50,
        )
    with pytest.raises(TypeError):
        refine_mask(
            mask=mask,
            data=data,
            num_of_points=[10],
            cell_size=50,
        )
    with pytest.raises(TypeError):
        refine_mask(
            mask=mask,
            data=data,
            num_of_points=10,
            cell_size=[50],
        )
    with pytest.raises(TypeError):
        refine_mask(
            mask=mask, data=data, num_of_points=10, cell_size=50, area_limit=[100]
        )


def test_quad_tree_mask_refinement():
    from pyheatdemand.processing import quad_tree_mask_refinement, create_polygon_mask

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (0, 0),  # Bottom-left corner
                    (0, 100),  # Top-left corner
                    (100, 100),  # Top-right corner
                    (100, 0),  # Bottom-right corner
                    (0, 0),  # Close the polygon by repeating the first point
                ]
            )
        ],
        crs="EPSG:3034",
    )

    mask = create_polygon_mask(gdf=gdf, step_size=10, crop_gdf=False)

    generator = np.random.default_rng(42)  # set seed number for reproducibility
    x = generator.integers(low=0, high=100, size=1000)

    y = generator.integers(low=0, high=100, size=1000)
    data = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=x, y=y, crs="EPSG:3034"))

    gdf_refined = quad_tree_mask_refinement(
        mask=mask, data=data, max_depth=2, num_of_points=10
    )

    assert isinstance(gdf_refined, gpd.GeoDataFrame)


def test_quad_tree_mask_refinement_error():
    from pyheatdemand.processing import quad_tree_mask_refinement, create_polygon_mask

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (0, 0),  # Bottom-left corner
                    (0, 100),  # Top-left corner
                    (100, 100),  # Top-right corner
                    (100, 0),  # Bottom-right corner
                    (0, 0),  # Close the polygon by repeating the first point
                ]
            )
        ],
        crs="EPSG:3034",
    )

    mask = create_polygon_mask(gdf=gdf, step_size=10, crop_gdf=False)

    generator = np.random.default_rng(42)  # set seed number for reproducibility
    x = generator.integers(low=0, high=100, size=1000)

    y = generator.integers(low=0, high=100, size=1000)
    data = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=x, y=y, crs="EPSG:3034"))

    with pytest.raises(TypeError):
        gdf_refined = quad_tree_mask_refinement(
            mask=[mask], data=data, max_depth=2, num_of_points=10
        )
    with pytest.raises(TypeError):
        gdf_refined = quad_tree_mask_refinement(
            mask=mask, data=[data], max_depth=2, num_of_points=10
        )
    with pytest.raises(TypeError):
        gdf_refined = quad_tree_mask_refinement(
            mask=mask, data=data, max_depth=[2], num_of_points=10
        )
    with pytest.raises(TypeError):
        gdf_refined = quad_tree_mask_refinement(
            mask=mask, data=data, max_depth=2, num_of_points="10"
        )


def test_create_connection():
    from pyheatdemand.processing import create_connection

    point = Point(5, 5)

    linestring = LineString([(0, 0), (10, 0)])

    connection = create_connection(linestring=linestring, point=point)

    assert isinstance(connection, LineString)
    assert connection.wkt == "LINESTRING (5 0, 5 5)"


def test_create_connection_error():
    from pyheatdemand.processing import create_connection

    point = Point(5, 5)

    linestring = LineString([(0, 0), (10, 0)])

    with pytest.raises(TypeError):
        connection = create_connection(linestring=[linestring], point=point)

    with pytest.raises(TypeError):
        connection = create_connection(linestring=linestring, point=[point])


def test_create_connections():
    from pyheatdemand.processing import create_connections

    point1 = Point(5, 5)
    point2 = Point(5, 5)

    linestring1 = LineString([(10, 10), (10, 0)])

    linestring2 = LineString([(0, 0), (10, 0)])

    gdf_points = gpd.GeoDataFrame(geometry=[point1, point2], crs="EPSG:25832")

    gdf_points["HD"] = [10, 10]

    gdf_linestrings = gpd.GeoDataFrame(
        geometry=[linestring1, linestring2], crs="EPSG:25832"
    )

    gdf_connections = create_connections(
        gdf_buildings=gdf_points, gdf_roads=gdf_linestrings
    )

    assert isinstance(gdf_connections, gpd.GeoDataFrame)

    gdf_connections = create_connections(
        gdf_buildings=gdf_points, gdf_roads=gdf_linestrings, hd_data_column="HD"
    )

    assert isinstance(gdf_connections, gpd.GeoDataFrame)


def test_create_connections_error():
    from pyheatdemand.processing import create_connections

    point1 = Point(5, 5)
    point2 = Point(5, 5)

    linestring1 = LineString([(10, 10), (10, 0)])

    linestring2 = LineString([(0, 0), (10, 0)])

    gdf_points = gpd.GeoDataFrame(geometry=[point1, point2], crs="EPSG:25832")
    gdf_points["HD"] = [10, 10]

    gdf_linestrings = gpd.GeoDataFrame(
        geometry=[linestring1, linestring2], crs="EPSG:25832"
    )
    with pytest.raises(TypeError):
        gdf_connections = create_connections(
            gdf_buildings=[gdf_points], gdf_roads=gdf_linestrings, hd_data_column="HD"
        )

    with pytest.raises(TypeError):
        gdf_connections = create_connections(
            gdf_buildings=gdf_points, gdf_roads=[gdf_linestrings], hd_data_column="HD"
        )

    with pytest.raises(TypeError):
        gdf_connections = create_connections(
            gdf_buildings=gdf_points, gdf_roads=gdf_linestrings, hd_data_column=["HD"]
        )

    with pytest.raises(ValueError):
        gdf_connections = create_connections(
            gdf_buildings=gdf_points, gdf_roads=gdf_linestrings, hd_data_column="HD1"
        )


@pytest.mark.parametrize("path", ["data/Data_Type_I_Raster.tif"])
def test_convert_dtype(path):
    from pyheatdemand.processing import convert_dtype

    convert_dtype(path_in=path, path_out="data/Data_Type_I_Raster_out.tif")


@pytest.mark.parametrize("path", ["data/Data_Type_I_Raster.tif"])
def test_convert_dtype_error(path):
    from pyheatdemand.processing import convert_dtype

    with pytest.raises(TypeError):
        convert_dtype(path_in=path, path_out=["data/Data_Type_I_Raster_out.tif"])

    with pytest.raises(TypeError):
        convert_dtype(path_in=[path], path_out="data/Data_Type_I_Raster_out.tif")


@pytest.mark.parametrize("gdf_buildings", [gpd.read_file("data/Aachen_Buildings.shp")])
@pytest.mark.parametrize("gdf_roads", [gpd.read_file("data/Aachen_Streets.shp")])
def test_calculate_hd_street_segments(gdf_buildings, gdf_roads):
    from pyheatdemand.processing import calculate_hd_street_segments

    gdf_roads = gdf_roads.set_crs("EPSG:4326").to_crs("EPSG:25832")

    gdf_hd = calculate_hd_street_segments(
        gdf_buildings=gdf_buildings, gdf_roads=gdf_roads, hd_data_column="WB_HU"
    )

    assert isinstance(gdf_hd, gpd.GeoDataFrame)


@pytest.mark.parametrize("gdf_buildings", [gpd.read_file("data/Aachen_Buildings.shp")])
@pytest.mark.parametrize("gdf_roads", [gpd.read_file("data/Aachen_Streets.shp")])
def test_calculate_hd_street_segments_error(gdf_buildings, gdf_roads):
    from pyheatdemand.processing import calculate_hd_street_segments

    gdf_roads = gdf_roads.set_crs("EPSG:4326").to_crs("EPSG:25832")

    with pytest.raises(TypeError):
        gdf_hd = calculate_hd_street_segments(
            gdf_buildings=[gdf_buildings], gdf_roads=gdf_roads, hd_data_column="WB_HU"
        )

    with pytest.raises(TypeError):
        gdf_hd = calculate_hd_street_segments(
            gdf_buildings=gdf_buildings, gdf_roads=[gdf_roads], hd_data_column="WB_HU"
        )

    with pytest.raises(TypeError):
        gdf_hd = calculate_hd_street_segments(
            gdf_buildings=gdf_buildings, gdf_roads=gdf_roads, hd_data_column=5
        )

    with pytest.raises(ValueError):
        gdf_hd = calculate_hd_street_segments(
            gdf_buildings=gdf_buildings, gdf_roads=gdf_roads, hd_data_column="WB_HU1"
        )
