import pytest
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon
import sys
import os

sys.path.insert(0, '../')


def test_create_polygon_mask():
    from pyhd.processing import create_polygon_mask

    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0),  # Bottom-left corner
                                              (0, 100),  # Top-left corner
                                              (100, 100),  # Top-right corner
                                              (100, 0),  # Bottom-right corner
                                              (0, 0)  # Close the polygon by repeating the first point
                                              ])],
                           crs='EPSG:3034'
                           )

    mask = create_polygon_mask(gdf=gdf,
                               step_size=10,
                               crop_gdf=False)

    assert isinstance(mask, gpd.GeoDataFrame)
    assert len(mask) == 100
    assert all(mask.area == 100) == True
    assert mask.crs == 'EPSG:3034'

    mask = create_polygon_mask(gdf=gdf,
                               step_size=10,
                               crop_gdf=True)

    assert isinstance(mask, gpd.GeoDataFrame)
    assert len(mask) == 100
    assert all(mask.area == 100) == True
    assert mask.crs == 'EPSG:3034'


def test_create_polygon_mask_error():
    from pyhd.processing import create_polygon_mask

    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0),  # Bottom-left corner
                                              (0, 100),  # Top-left corner
                                              (100, 100),  # Top-right corner
                                              (100, 0),  # Bottom-right corner
                                              (0, 0)  # Close the polygon by repeating the first point
                                              ])],
                           crs='EPSG:3034'
                           )

    with pytest.raises(TypeError):
        create_polygon_mask(gdf=[gdf],
                            step_size=10,
                            crop_gdf=False)
    with pytest.raises(TypeError):
        create_polygon_mask(gdf=gdf,
                            step_size=10.5,
                            crop_gdf=False)
    with pytest.raises(TypeError):
        create_polygon_mask(gdf=gdf,
                            step_size=10,
                            crop_gdf='False')


@pytest.mark.parametrize('path',
                         ['data/Data_Type_I_Raster.tif'])
def test_vectorize_raster(path):
    from pyhd.processing import vectorize_raster

    data = rasterio.open(path)

    gdf = vectorize_raster(path=path)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert data.crs == gdf.crs


@pytest.mark.parametrize('path',
                         ['data/Data_Type_I_Raster.tif'])
def test_vectorize_raster_error(path):
    from pyhd.processing import vectorize_raster

    with pytest.raises(TypeError):
        vectorize_raster(path=[path])


def test_create_outline():
    from pyhd.processing import create_outline

    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0),  # Bottom-left corner
                                              (0, 100),  # Top-left corner
                                              (100, 100),  # Top-right corner
                                              (100, 0),  # Bottom-right corner
                                              (0, 0)  # Close the polygon by repeating the first point
                                              ])],
                           crs='EPSG:3034'
                           )

    outline = create_outline(gdf=gdf)

    assert isinstance(outline, gpd.GeoDataFrame)
    assert outline.crs == gdf.crs


def test_create_outline_error():
    from pyhd.processing import create_outline

    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0),  # Bottom-left corner
                                              (0, 100),  # Top-left corner
                                              (100, 100),  # Top-right corner
                                              (100, 0),  # Bottom-right corner
                                              (0, 0)  # Close the polygon by repeating the first point
                                              ])],
                           crs='EPSG:3034'
                           )

    with pytest.raises(TypeError):
        create_outline(gdf=[gdf])


@pytest.mark.parametrize('mask_gdf',
                         [gpd.read_file('data/Interreg_NWE_mask_500m_EPSG3034.shp')])
@pytest.mark.parametrize('hd_gdf',
                         [gpd.read_file('data/Data_Type_I_Vector_HD_Data.shp')])
def test_calculate_hd(mask_gdf, hd_gdf):
    from pyhd.processing import calculate_hd

    gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                          mask_gdf=mask_gdf,
                          hd_data_column='WOHNGEB_WB')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    hd_gdf_multi = gpd.GeoDataFrame(geometry=[MultiPolygon([hd_gdf.geometry.values])],
                                    crs='EPSG:3034')
    hd_gdf_multi['WOHNGEB_WB'] = 100

    gdf_hd = calculate_hd(hd_gdf=hd_gdf_multi,
                          mask_gdf=mask_gdf,
                          hd_data_column='WOHNGEB_WB')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    gdf_hd = calculate_hd(hd_gdf=hd_gdf.to_crs('EPSG:25832'),
                          mask_gdf=mask_gdf,
                          hd_data_column='WOHNGEB_WB')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs


@pytest.mark.parametrize('mask_gdf',
                         [gpd.read_file('data/Interreg_NWE_mask_500m_EPSG3034.shp')])
@pytest.mark.parametrize('hd_gdf',
                         [gpd.read_file('data/Data_Type_I_Vector_HD_Data.shp')])
def test_calculate_hd_points(mask_gdf, hd_gdf):
    from pyhd.processing import calculate_hd

    hd_gdf['geometry'] = hd_gdf['geometry'].centroid
    gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                          mask_gdf=mask_gdf,
                          hd_data_column='WOHNGEB_WB')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    gdf_hd = calculate_hd(hd_gdf=hd_gdf.to_crs('EPSG:25832'),
                          mask_gdf=mask_gdf,
                          hd_data_column='WOHNGEB_WB')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs


@pytest.mark.parametrize('mask_gdf',
                         [gpd.read_file('data/Interreg_NWE_mask_500m_EPSG3034.shp')])
@pytest.mark.parametrize('hd_gdf',
                         [gpd.read_file('data/Data_Type_II_Vector_Lines.shp')])
def test_calculate_hd_lines(mask_gdf, hd_gdf):
    from pyhd.processing import calculate_hd

    gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                          mask_gdf=mask_gdf,
                          hd_data_column='RW_WW_WBED')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.to_crs('EPSG:3034').crs

    gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                          mask_gdf=mask_gdf,
                          hd_data_column='RW_WW_WBED')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.to_crs('EPSG:3034').crs


@pytest.mark.parametrize('mask_gdf',
                         [gpd.read_file('data/Interreg_NWE_mask_500m_EPSG3034.shp')])
@pytest.mark.parametrize('hd_gdf',
                         [gpd.read_file('data/Data_Type_I_Vector_HD_Data.shp')])
def test_calculate_hd_error(mask_gdf, hd_gdf):
    from pyhd.processing import calculate_hd

    with pytest.raises(TypeError):
        gdf_hd = calculate_hd(hd_gdf=[hd_gdf],
                              mask_gdf=mask_gdf,
                              hd_data_column='WOHNGEB_WB')
    with pytest.raises(TypeError):
        gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                              mask_gdf=[mask_gdf],
                              hd_data_column='WOHNGEB_WB')
    with pytest.raises(TypeError):
        gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                              mask_gdf=mask_gdf,
                              hd_data_column=['WOHNGEB_WB'])

    with pytest.raises(ValueError):
        gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                              mask_gdf=mask_gdf,
                              hd_data_column='WOHNGEB_W')


@pytest.mark.parametrize('mask_gdf',
                         [gpd.read_file('data/Interreg_NWE_mask_500m_EPSG3034.shp')])
@pytest.mark.parametrize('hd_gdf',
                         [gpd.read_file('data/Data_Type_I_Vector_HD_Data.shp')])
def test_rasterize_gdf_hd(mask_gdf, hd_gdf):
    from pyhd.processing import rasterize_gdf_hd, calculate_hd

    gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                          mask_gdf=mask_gdf,
                          hd_data_column='WOHNGEB_WB')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    try:
        os.remove('data/Data_Type_I_Vector.tif')
    except FileNotFoundError:
        pass

    rasterize_gdf_hd(gdf_hd=gdf_hd,
                     path_out='data/Data_Type_I_Vector.tif',
                     crs='EPSG:3034',
                     xsize=100,
                     ysize=100)

    os.remove('data/Data_Type_I_Vector.tif')


@pytest.mark.parametrize('mask_gdf',
                         [gpd.read_file('data/Interreg_NWE_mask_500m_EPSG3034.shp')])
@pytest.mark.parametrize('hd_gdf',
                         [gpd.read_file('data/Data_Type_I_Vector_HD_Data.shp')])
def test_rasterize_gdf_hd_error(mask_gdf, hd_gdf):
    from pyhd.processing import rasterize_gdf_hd, calculate_hd

    gdf_hd = calculate_hd(hd_gdf=hd_gdf,
                          mask_gdf=mask_gdf,
                          hd_data_column='WOHNGEB_WB')

    assert isinstance(gdf_hd, gpd.GeoDataFrame)
    assert gdf_hd.crs == hd_gdf.crs

    with pytest.raises(TypeError):
        rasterize_gdf_hd(gdf_hd=[gdf_hd],
                         path_out='data/Data_Type_I_Vector.tif',
                         crs='EPSG:3034',
                         xsize=100,
                         ysize=100)
    with pytest.raises(TypeError):
        rasterize_gdf_hd(gdf_hd=gdf_hd,
                         path_out=['data/Data_Type_I_Vector.tif'],
                         crs='EPSG:3034',
                         xsize=100,
                         ysize=100)
    with pytest.raises(TypeError):
        rasterize_gdf_hd(gdf_hd=gdf_hd,
                         path_out='data/Data_Type_I_Vector.tif',
                         crs=['EPSG:3034'],
                         xsize=100,
                         ysize=100)
    with pytest.raises(TypeError):
        rasterize_gdf_hd(gdf_hd=gdf_hd,
                         path_out='data/Data_Type_I_Vector.tif',
                         crs='EPSG:3034',
                         xsize=[100],
                         ysize=100)
    with pytest.raises(TypeError):
        rasterize_gdf_hd(gdf_hd=gdf_hd,
                         path_out='data/Data_Type_I_Vector.tif',
                         crs='EPSG:3034',
                         xsize=100,
                         ysize=[100])


@pytest.mark.parametrize('df',
                         [pd.read_csv('data/Data_Type_III_Point_Addresses.csv')])
def test_obtain_coordinates_from_addresses(df):
    from pyhd.processing import obtain_coordinates_from_addresses
    df['PLZ'] = df['PLZ'].astype(int)
    df['Strasse'] = df['Strasse'].apply(
        lambda x: ''.join(' ' + char if char.isupper() else char for char in x).strip())

    gdf = obtain_coordinates_from_addresses(df[:2],
                                            'Strasse',
                                            'Nummer',
                                            'PLZ',
                                            'Ort',
                                            'EPSG:3034')

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == 'EPSG:3034'


@pytest.mark.parametrize('df',
                         [pd.read_csv('data/Data_Type_III_Point_Addresses.csv')])
def test_obtain_coordinates_from_addresses_error(df):
    from pyhd.processing import obtain_coordinates_from_addresses
    df['PLZ'] = df['PLZ'].astype(int)
    df['Strasse'] = df['Strasse'].apply(
        lambda x: ''.join(' ' + char if char.isupper() else char for char in x).strip())

    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses([df[:2]],
                                                'Strasse',
                                                'Nummer',
                                                'PLZ',
                                                'Ort',
                                                'EPSG:3034')
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(df[:2],
                                                ['Strasse'],
                                                'Nummer',
                                                'PLZ',
                                                'Ort',
                                                'EPSG:3034')
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(df[:2],
                                                'Strasse',
                                                ['Nummer'],
                                                'PLZ',
                                                'Ort',
                                                'EPSG:3034')
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(df[:2],
                                                'Strasse',
                                                'Nummer',
                                                ['PLZ'],
                                                'Ort',
                                                'EPSG:3034')
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(df[:2],
                                                'Strasse',
                                                'Nummer',
                                                'PLZ',
                                                ['Ort'],
                                                'EPSG:3034')
    with pytest.raises(TypeError):
        gdf = obtain_coordinates_from_addresses(df[:2],
                                                'Strasse',
                                                'Nummer',
                                                'PLZ',
                                                'Ort',
                                                ['EPSG:3034'])


def test_get_building_footprint():
    from pyhd.processing import get_building_footprint

    gdf = get_building_footprint(point=Point(6.07868, 50.77918),
                                 dist=25)

    assert isinstance(gdf, gpd.GeoDataFrame)

    gdf = get_building_footprint(point=Point(6.07868, 50.77918),
                                 dist=1)

    assert isinstance(gdf, type(None))


def test_get_building_footprint_error():
    from pyhd.processing import get_building_footprint

    with pytest.raises(TypeError):
        get_building_footprint(point=[Point(6.07868, 50.77918)],
                               dist=25)

    with pytest.raises(TypeError):
        get_building_footprint(point=Point(6.07868, 50.77918),
                               dist=[25])


def test_get_building_footprints():
    from pyhd.processing import get_building_footprints

    points = gpd.GeoDataFrame(geometry=[Point(6.07868, 50.77918)],
                              crs='EPSG:4326')

    gdf = get_building_footprints(points=points,
                                  dist=250,
                                  perform_sjoin=True)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1

    gdf = get_building_footprints(points=points,
                                  dist=250,
                                  perform_sjoin=False)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) != 1

    gdf = get_building_footprints(points=points.to_crs('EPSG:3034'),
                                  dist=250,
                                  perform_sjoin=False)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) != 1


def test_get_building_footprints_error():
    from pyhd.processing import get_building_footprints
    points = gpd.GeoDataFrame(geometry=[Point(6.07868, 50.77918)],
                              crs='EPSG:4326')

    with pytest.raises(TypeError):
        gdf = get_building_footprints(points=[points],
                                      dist=250,
                                      perform_sjoin=True)

    with pytest.raises(TypeError):
        gdf = get_building_footprints(points=points,
                                      dist=[250],
                                      perform_sjoin=True)

    with pytest.raises(TypeError):
        gdf = get_building_footprints(points=points,
                                      dist=250,
                                      perform_sjoin=[True])


def test_merge_rasters():
    from pyhd.processing import merge_rasters

    raster_list = os.listdir('data/rasters/')
    raster_list = [os.path.join(os.path.abspath('data/rasters/'), path) for path in raster_list]

    merge_rasters(raster_list, 'data/Raster_merged.tif')


def test_merge_rasters_error():
    from pyhd.processing import merge_rasters

    raster_list = os.listdir('data/rasters/')
    raster_list = [os.path.join(os.path.abspath('data/rasters/'), path) for path in raster_list]

    with pytest.raises(TypeError):
        merge_rasters(raster_list[0], 'data/Raster_merged.tif')

    with pytest.raises(TypeError):
        merge_rasters(raster_list, ['data/Raster_merged.tif'])


def test_calculate_zonal_stats():
    from pyhd.processing import calculate_zonal_stats

    gdf_stats = calculate_zonal_stats("data/nw_dvg_krs.shp",
                                      "data/HD_RBZ_Köln.tif",
                                      'EPSG:3034',
                                      calculate_heated_area=True)

    assert isinstance(gdf_stats, gpd.GeoDataFrame)

    gdf_stats = calculate_zonal_stats("data/nw_dvg_krs.shp",
                                      "data/HD_RBZ_Köln.tif",
                                      'EPSG:3034',
                                      calculate_heated_area=False)

    assert isinstance(gdf_stats, gpd.GeoDataFrame)


def test_calculate_zonal_stats_error():
    from pyhd.processing import calculate_zonal_stats

    with pytest.raises(TypeError):
        calculate_zonal_stats(["data/nw_dvg_krs.shp"],
                              "data/HD_RBZ_Köln.tif",
                              'EPSG:3034',
                              calculate_heated_area=False)

    with pytest.raises(TypeError):
        calculate_zonal_stats("data/nw_dvg_krs.shp",
                              ["data/HD_RBZ_Köln.tif"],
                              'EPSG:3034',
                              calculate_heated_area=False)

    with pytest.raises(TypeError):
        calculate_zonal_stats("data/nw_dvg_krs.shp",
                              "data/HD_RBZ_Köln.tif",
                              ['EPSG:3034'],
                              calculate_heated_area=False)

    with pytest.raises(TypeError):
        calculate_zonal_stats("data/nw_dvg_krs.shp",
                              "data/HD_RBZ_Köln.tif",
                              'EPSG:3034',
                              calculate_heated_area='False')
