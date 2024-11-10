import rasterio
import matplotlib.pyplot as plt
import json
from scipy.interpolate import make_interp_spline
import numpy as np
import geojson
from shapely.geometry import shape, Point
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.mask import mask
import os
import glob


def extract_bbox_json(coord_list):
     """
     Extracts bbox from the list of coordinates.
     Args:
        coord_list: list of coordinates
     """
     box = []
     for i in (0,1):
         res = sorted(coord_list, key=lambda x:x[i])
         box.append((res[0][i],res[-1][i]))
     ret = f"({box[0][0]} {box[1][0]}, {box[0][1]} {box[1][1]})"
     return ret


def plot_tensorboard_logs(path_to_file, graph_name="Graph", x_name="X", y_name="Y", intepolate=True):
    """
    Takes .json with Tensorboard logs and plots the results
    """
    with open(path_to_file) as f:
        contents = json.load(f)
        x = [i[1] for i in contents]
        y = [j[2] for j in contents]
    
    if intepolate:
        x = np.asarray(x)
        y = np.asarray(y)
        spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 10)
        Y_ = spline(X_)
        plt.plot(X_, Y_)
        plt.xlabel(x_name)  
        plt.ylabel(y_name)   
        plt.title(graph_name)
        plt.show() 

    else:
        plt.plot(x, y) 
        plt.xlabel(x_name)  
        plt.ylabel(y_name)   
        plt.title(graph_name)
        plt.show()

    
def plot_tensorboard_logs_two_files(path_to_file_1, path_to_file_2, graph_name="Graph", x_name="X", y_name="Y", intepolate=True):
    """
    Takes .json with Tensorboard logs and plots the results
    """
    with open(path_to_file_1) as f:
        contents = json.load(f)
        x_1 = [i[1] for i in contents]
        y_1 = [j[2] for j in contents]
        print(len(y_1))
    
    with open(path_to_file_2) as f:
        contents = json.load(f)
        x_2 = [i[1] for i in contents]
        y_2 = [j[2] for j in contents]
        print(len(y_2))

    if intepolate:
        x_1 = np.asarray(x_1)
        x_2 = np.asarray(x_2)
        y_1 = np.asarray(y_1)
        y_2 = np.asarray(y_2)
        spline_1 = make_interp_spline(x_1, y_1)
        spline_2 = make_interp_spline(x_2, y_2)
        X_1 = np.linspace(x_1.min(), x_1.max(), 200)
        X_2 = np.linspace(x_2.min(), x_2.max(), 200)
        Y_1 = spline_1(X_1)
        Y_2 = spline_2(X_2)
        plt.plot(X_1, Y_1, label='Train')
        plt.plot(X_2, Y_2, label='Val')
        plt.xlabel(x_name)  
        plt.ylabel(y_name)   
        plt.title(graph_name)
        plt.legend()
        plt.show() 

    else:
        plt.plot(x_1, y_1, label='Train') 
        plt.plot(x_2, y_2, label='Val')
        plt.xlabel(x_name)  
        plt.ylabel(y_name)   
        plt.title(graph_name)
        plt.legend()
        plt.show()


def get_bbox_from_geojson(geojson_file):
    """
    Getting the bbox from the Multipolygon in the geojson file

    Args:
        geojson_file: path to geojson file
    """
    with open(geojson_file, 'r') as f:
        data = geojson.load(f)
    for feature in data['features']:
        geometry = feature['geometry']
        # checking if the feature is a MultiPolygon
        if geometry['type'] == 'MultiPolygon':
            # converting the geometry to a shapely shape
            multi_polygon = shape(geometry)
            # getting the bounding box of the MultiPolygon
            bbox = multi_polygon.bounds  # (minx, miny, maxx, maxy)
            
            return bbox
        

def get_bbox_geotiff(geotiff):
    """
    Returns bbox for geotiff file.
    """
    with rasterio.open(geotiff) as src:
        return src.bounds
        

def reproject_and_rasterize(geojson_file, output_crs, raster_output_file, pixel_size=10):
    """
    Reprojects and rasterizes geojson file.

    Args:
        geojson_file: path to geojson file
        output_crs: CRS of the output raster
        raster_output_file: output raster filepath
        pixel_size: resolution (default: 10m)
    """

    gdf = gpd.read_file(geojson_file)
    gdf_projected = gdf.to_crs(output_crs)
    bounds = gdf_projected.total_bounds
    width = int((bounds[2] - bounds[0]) / pixel_size)
    height = int((bounds[3] - bounds[1]) / pixel_size)

    transform = rasterio.transform.from_origin(bounds[0], bounds[3], pixel_size, pixel_size)
    shapes = [(geom, 1) for geom in gdf_projected.geometry]  # Assign a value of 1 to all features
    raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype='uint8')

    with rasterio.open(
        raster_output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=output_crs,
        transform=transform,
        compress="LZW"
    ) as dst:
        dst.write(raster, 1)

    print(f"Rasterized output saved to {raster_output_file}")


def get_centroid_geotiff(tiff_file):
    """
    Gets the centroid of the bounding box of a GeoTIFF file and its CRS
    """
    with rasterio.open(tiff_file) as src:
        bounds = src.bounds
        # Compute the centroid from the bounds
        centroid_x = (bounds.left + bounds.right) / 2
        centroid_y = (bounds.top + bounds.bottom) / 2
        tiff_crs = src.crs
        return Point(centroid_x, centroid_y), tiff_crs
    
    
def centroid_in_polygon(centroid, centroid_crs, polygon_gdf):
    """
    Check if the centroid is within the polygon stored in the GeoDataFrame.

    Args:
        centroid: centorid coordinates
        centroid_crs: centorid CRS
        polygon_gdf: GeoDataFrame object containing the polygon
    """
    # ensuring the CRS is the same
    if polygon_gdf.crs != centroid_crs:
        polygon_gdf = polygon_gdf.to_crs(centroid_crs)

    return polygon_gdf.contains(centroid).values[0]


def get_corners_of_geotiff(tiff_file):
    """
    Gets the corner points (coordinates) of the bounding box of a GeoTIFF file.
    """
    with rasterio.open(tiff_file) as src:
        bounds = src.bounds
        # Get the four corners of the bounding box
        corners = [
            Point(bounds.left, bounds.top),    # top-left
            Point(bounds.right, bounds.top),   # top-right
            Point(bounds.right, bounds.bottom),# bottom-right
            Point(bounds.left, bounds.bottom)  # bottom-left
        ]
        return corners


def reproject_mask(mask_src, original_src):
    """
    Reprojects the mask file to match the CRS and resolution of the original GeoTIFF file.

    Args:
        mask_src: rasterio dataset object containing mask
        original_src: rasterio dataset object to match the mask with
    """
    # reading the original file's dimensions, transform, and CRS
    original_transform = original_src.transform
    original_crs = original_src.crs
    original_shape = (original_src.height, original_src.width)
    
    # creating an empty array to hold the reprojected mask data
    reprojected_mask = np.empty(original_shape, dtype=mask_src.dtypes[0])
    
    # reprojecting the mask to match the original GeoTIFF's CRS and transform
    rasterio.warp.reproject(
        source=rasterio.band(mask_src, 1),
        destination=reprojected_mask,
        src_transform=mask_src.transform,
        src_crs=mask_src.crs,
        dst_transform=original_transform,
        dst_crs=original_crs,
        resampling=rasterio.warp.Resampling.nearest
    )
    
    return reprojected_mask


def extract_last_two_numbers(filename):
    """
    Splitting the filename by underscores and extract the second-to-last and last parts (before '.tif')
    """
    parts = filename.split('_')
    second_last = int(parts[-2])  # Second-to-last number
    last = int(parts[-1].split('.')[0])  # Last number before '.tif'
    return (second_last, last)


def compress_tiff(folder):
    """
    Compresses all the files in the directory with LZW compression.
    Files should contain 1 band.

    Args:
        fodler: root path containing files
    """
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        with rasterio.open(path, "r") as src:
            data = src.read(1)
            profile = src.profile
            profile.update({"compress": "lzw"})
            print(src.count)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
    print("Processing finished!")     


def modify_geotiff_with_mask(original_file, mask_file, output_file):
    """
    Crops the geotiff to match the mask.

    Args:
        original_file: path to Geotiff file to be modified
        mask_file: path to Geotiff file to be used as a mask
        output_file: path to save the output cropped Geotiff file
    """
    # opening the original Geotiff file (with 12 channels)
    with rasterio.open(original_file) as src:
        profile = src.profile 
        original_data = src.read()
        
    # opening the mask Geotiff file
    with rasterio.open(mask_file) as mask_src:
        # if the CRS of the mask does not match the original, reprojecting the mask
        if mask_src.crs != src.crs:
            print(f"Reprojecting mask to match CRS of the original GeoTIFF...")
            mask_data = reproject_mask(mask_src, src)
        else:
            mask_data = mask_src.read(1)
            
        with rasterio.open(output_file, 'w', **profile) as dst:            
            mask_window_data = mask_data                
            # finding the pixels in the mask that have a value of 0
            zero_mask = mask_window_data == 0
            # replacing corresponding pixels in all channels of the original data with 0
            original_data[:, zero_mask] = 0
            dst.write(original_data)

    print(f"Modified GeoTIFF saved to {output_file}")


def crop_gtiff_by_polygon(geojson_path, geotiff_path, geotiff_out_path):
    """
    Crops the GeoTIFF file (geotiff_path) by using geojson file which should contain 1 polygon.
    
    Args:
        geojson_path: path to geojson file with 1 polygon
        geotiff_path: path to GeoTIFF file
        geotiff_out_path: output geotiff filepath
    """
    polygon = gpd.read_file(geojson_path)
    src = rasterio.open(geotiff_path)
    if src.crs != polygon.crs:
        print("Changing CRS...")
        polygon = polygon.to_crs(src.crs)

    geometry = [polygon.geometry[0]]
    out_image, out_transform = mask(src, geometry, crop=True)
    out_meta = src.meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    src.close()

    with rasterio.open(geotiff_out_path, 'w', **out_meta) as dest:
        dest.write(out_image)

    print("Cropped!")


def get_perc_of_utc(gtiff_path=None, value=1, batch_proc=False, root_path=None, print_dict=True):
    """
    Returns the dict, containing the percentage of elements of value (value) in GeoTIFF. 
    Assuming the raster is normalized to contain the values between 0 and 1.
    Takes only single-band rasters. 
    
    Args:
        gtiff_path: path of GeoTIFF to process
        value: target value
        batch_proc: True if processing all the GeoTIFF files in the root_path
        root_path: path of all GeoTIFFs to process if batch_proc
        print_dict: print dict in console
    """
    if not batch_proc:
        dict_out = {}
        with rasterio.open(gtiff_path) as src:
            img = src.read(1)
            num_value = (img == value).sum()
            num_zeroes = (img == 0).sum()
            total = num_value + num_zeroes
            dict_out[gtiff_path] = num_value / total
            if print_dict:
                print(dict_out)
    if batch_proc:
        print("Starting batch processing...")
        dict_out = {}
        for filepath in glob.glob(f"{root_path}/*.tif"):
            with rasterio.open(filepath) as src:
                img = src.read(1)
                num_value = (img == value).sum()
                num_zeroes = (img == 0).sum()
                total = num_value + num_zeroes
                perc = num_value / total
                dict_out[filepath] = f"{perc:.4f}"
                if print_dict:
                    print(filepath, dict_out[filepath])
    
    if not print_dict:
        return dict_out
