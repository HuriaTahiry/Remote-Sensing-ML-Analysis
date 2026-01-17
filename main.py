# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import os
# from rasterio.plot import show
# from rasterio.merge import merge
# from rasterio.mask import mask
# import geopandas as gpd
# import json
#
#
# data_path = "/Users/huriatahiry/PycharmProjects/GIS/LT05_L2SP_015035_19900116_20200916_02_T1"
# band_files = sorted(glob.glob(os.path.join(data_path, "*B[1-7].TIF")))
#
# print("Bands found:", band_files)
#
# # Create output directories
# out_dir = "/Users/huriatahiry/Desktop"
#
# clipped_out_dir = "/Users/huriatahiry/Desktop/research/clipped_bands/"
# os.makedirs(out_dir, exist_ok=True)
# os.makedirs(clipped_out_dir, exist_ok=True)
#
#
#
# def clip_bands_to_aoi():
#
#     try:
#         # study area boundary
#
#         # Have to make sure this is write
#         aoi = gpd.read_file("../data/study_area.geojson")  # .shp the file tupe
#         aoi = aoi.to_crs(epsg=4326)  # has to macht CRS
#
#         print("Clipping bands to study area...")
#
#         for band_path in band_files:
#             with rasterio.open(band_path) as src:
#                 out_image, out_transform = mask(src, aoi.geometry, crop=True)
#                 out_meta = src.meta.copy()
#                 out_meta.update({
#                     "height": out_image.shape[1],
#                     "width": out_image.shape[2],
#                     "transform": out_transform
#                 })
#
#                 # Save clipped band
#                 out_name = os.path.join(clipped_out_dir,
#                                         os.path.basename(band_path).replace(".TIF", "_clipped.TIF"))
#                 with rasterio.open(out_name, "w", **out_meta) as dest:
#                     dest.write(out_image)
#                 print("Saved:", out_name)
#
#         print(" All bands clipped successfully")
#         return True
#
#     except Exception as e:
#         print(f" Error clipping bands: {e}")
#         return False
#
#
#
# def read_band(band_path):
#     with rasterio.open(band_path) as src:
#         return src.read(1).astype('float32'), src.profile
#
#
# def load_bands():
#
#     # Landsat 5 TM bands:
#     # 1 = Blue, 2 = Green, 3 = Red, 4 = NIR, 5 = SWIR1, 7 = SWIR2
#
#     band1, meta = read_band(band_files[0])  # Blue
#     band2, _ = read_band(band_files[1])  # Green
#     band3, _ = read_band(band_files[2])  # Red
#     band4, _ = read_band(band_files[3])  # NIR
#     band5, _ = read_band(band_files[4])  # SWIR1
#     band7, _ = read_band(band_files[5])  # SWIR2
#
#     return band1, band2, band3, band4, band5, band7, meta
#
#
#
# def calculate_indices(band3, band4, band5):
#     # Calculate NDVI and NDBI indices
#     # NDVI = (NIR - RED) / (NIR + RED)
#     ndvi = (band4 - band3) / (band4 + band3)
#
#     # NDBI = (SWIR - NIR) / (SWIR + NIR)
#     ndbi = (band5 - band4) / (band5 + band4)
#
#     return ndvi, ndbi
#
#
#
# def visualize_results(band2, band3, band4, ndvi, ndbi):
#
#     plt.figure(figsize=(15, 5))
#
#     # False color composite
#     plt.subplot(1, 3, 1)
#     plt.imshow(np.clip(np.dstack([band4, band3, band2]) / np.percentile(band4, 98), 0, 1))
#     plt.title("False Color (NIR, Red, Green)")
#     plt.axis('off')
#
#     # NDVI
#     plt.subplot(1, 3, 2)
#     plt.imshow(ndvi, cmap='Greens')
#     plt.title("NDVI (Vegetation)")
#     plt.colorbar(shrink=0.7)
#
#     # NDBI
#     plt.subplot(1, 3, 3)
#     plt.imshow(ndbi, cmap='OrRd')
#     plt.title("NDBI (Built-up)")
#     plt.colorbar(shrink=0.7)
#
#     plt.tight_layout()
#     plt.show()
#
#
#
# def save_outputs(ndvi, ndbi, meta):
#     meta.update(dtype=rasterio.float32, count=1)
#
#     with rasterio.open(os.path.join(out_dir, "NDVI_1999.tif"), "w", **meta) as dest:
#         dest.write(ndvi, 1)
#
#     with rasterio.open(os.path.join(out_dir, "NDBI_1999.tif"), "w", **meta) as dest:
#         dest.write(ndbi, 1)
#
#     print("NDVI and NDBI maps saved to:", out_dir)
#
#
#
# def main():
#
#     print("Starting Landsat 1999 Analysis...")
#
#     # Clip bands to study area
#     clip_success = clip_bands_to_aoi()
#
#     # Load bands
#     print("Loading bands...")
#     band1, band2, band3, band4, band5, band7, meta = load_bands()
#
#     # calculate indices
#     print("Calculating indices...")
#     ndvi, ndbi = calculate_indices(band3, band4, band5)
#
#     # Visualize results
#     print("Creating visualizations...")
#     visualize_results(band2, band3, band4, ndvi, ndbi)
#
#     # Save outputs
#     print("Saving outputs...")
#     save_outputs(ndvi, ndbi, meta)
#
#     print(" Analysis completed successfully!")
#
# if __name__ == "__main__":
#     main()
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from rasterio.plot import show
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box, mapping
import json
import matplotlib.colors as mcolors

# Define Raleigh bounding box coordinates (more accurate for downtown Raleigh)
RALEIGH_BBOX = {
    "min_lon": -78.70,  # only 0.02 farther west
    "max_lon": -78.56,  # only 0.02 farther east
    "min_lat": 35.74,   # small south expansion
    "max_lat": 35.86    # small north expansion
}

# /Users/huriatahiry/PycharmProjects/GIS/LE07_L2SP_135047_20240119_20240213_02_T1 NO
# /Users/huriatahiry/PycharmProjects/GIS/LT05_L2SP_015035_19900116_20200916_02_T1 always yes
# /Users/huriatahiry/PycharmProjects/GIS/LT05_L2SP_015035_19981208_20200908_02_T1 yes
# /Users/huriatahiry/PycharmProjects/GIS/LT05_L2SP_039038_20120426_20200820_02_T1
# /Users/huriatahiry/PycharmProjects/GIS/LC08_L2SP_015035_20231229_20240108_02_T1
data_path = "/Users/huriatahiry/PycharmProjects/GIS/LC08_L2SP_015035_20231229_20240108_02_T1"
band_files = sorted(glob.glob(os.path.join(data_path, "*B[1-7].TIF")))

print("Bands found:", [os.path.basename(b) for b in band_files])

# Create output directories
out_dir = "/Users/huriatahiry/Desktop"
clipped_out_dir = "/Users/huriatahiry/Desktop/research/clipped_bands/"
raleigh_out_dir = os.path.join(out_dir, "raleigh_analysis")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(clipped_out_dir, exist_ok=True)
os.makedirs(raleigh_out_dir, exist_ok=True)


def create_raleigh_geometry():
    """Create Raleigh geometry directly"""
    raleigh_bbox = box(
        RALEIGH_BBOX["min_lon"],
        RALEIGH_BBOX["min_lat"],
        RALEIGH_BBOX["max_lon"],
        RALEIGH_BBOX["max_lat"]
    )

    # Create a GeoDataFrame
    aoi = gpd.GeoDataFrame(
        {'geometry': [raleigh_bbox], 'name': ['Raleigh_NC']},
        crs='EPSG:4326'
    )

    return aoi


def clip_bands_to_raleigh():
    """Clip bands specifically to Raleigh, NC bounding box"""
    try:
        # Create Raleigh geometry
        print("Creating Raleigh geometry...")
        aoi = create_raleigh_geometry()

        print(f"Clipping bands to Raleigh, NC area...")
        print(f"Raleigh bounds: {aoi.total_bounds}")

        # Save the boundary as a shapefile for reference
        boundary_file = os.path.join(raleigh_out_dir, "raleigh_boundary.shp")
        aoi.to_file(boundary_file)
        print(f"Saved Raleigh boundary to: {boundary_file}")

        for band_path in band_files:
            with rasterio.open(band_path) as src:
                print(f"\nProcessing: {os.path.basename(band_path)}")
                print(f"Source CRS: {src.crs}")
                print(f"Source bounds: {src.bounds}")

                # Reproject AOI to match raster CRS if needed
                if src.crs != aoi.crs:
                    print(f"Reprojecting AOI from {aoi.crs} to {src.crs}")
                    aoi = aoi.to_crs(src.crs)

                # Clip the band
                out_image, out_transform = mask(src, aoi.geometry, crop=True, all_touched=True)

                # Check if we got any data
                if out_image.shape[1] == 0 or out_image.shape[2] == 0:
                    print(f"Warning: No data in clipped area for {os.path.basename(band_path)}")
                    continue

                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Save clipped band
                out_name = os.path.join(
                    clipped_out_dir,
                    os.path.basename(band_path).replace(".TIF", "_raleigh.TIF")
                )

                with rasterio.open(out_name, "w", **out_meta) as dest:
                    dest.write(out_image)
                print(f"Saved: {out_name}")
                print(f"Shape: {out_image.shape}")
                print(f"Data range: [{out_image.min():.2f}, {out_image.max():.2f}]")

        print("\nAll bands clipped to Raleigh successfully")
        return True

    except Exception as e:
        print(f"Error clipping to Raleigh: {e}")
        import traceback
        traceback.print_exc()
        return False


def read_band(band_path):
    """Read a single band from a TIFF file with proper scaling"""
    with rasterio.open(band_path) as src:
        data = src.read(1).astype('float32')
        # Apply scale factor if it exists in metadata (common in Landsat SR products)
        if 'scale_factor' in src.tags():
            scale_factor = float(src.tags()['scale_factor'])
            data = data * scale_factor

        # Remove nodata values
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)

        return data, src.profile


def load_raleigh_bands():
    """Load Raleigh-clipped bands with proper band ordering"""
    # Get all Raleigh-clipped bands
    raleigh_bands = sorted(glob.glob(os.path.join(clipped_out_dir, "*_raleigh.TIF")))

    if not raleigh_bands:
        print("No Raleigh-clipped bands found. Clipping first...")
        if not clip_bands_to_raleigh():
            raise FileNotFoundError("Failed to clip bands to Raleigh area")
        raleigh_bands = sorted(glob.glob(os.path.join(clipped_out_dir, "*_raleigh.TIF")))

    print(f"\nFound {len(raleigh_bands)} Raleigh-clipped bands")

    # Sort bands properly: B1, B2, B3, B4, B5, B7
    # Landsat bands are: 1=Blue, 2=Green, 3=Red, 4=NIR, 5=SWIR1, 7=SWIR2
    band_dict = {}
    for band_path in raleigh_bands:
        band_num = os.path.basename(band_path).split('_B')[1][0]  # Extract band number
        band_dict[int(band_num)] = band_path

    # Read bands in correct order
    bands = []
    for band_num in [1, 2, 3, 4, 5, 7]:  # Skip thermal band (6)
        if band_num in band_dict:
            print(f"Loading Band {band_num}: {os.path.basename(band_dict[band_num])}")
            band_data, meta = read_band(band_dict[band_num])
            bands.append(band_data)

    if len(bands) < 6:
        raise ValueError(f"Expected 6 bands but found {len(bands)}")

    return bands[0], bands[1], bands[2], bands[3], bands[4], bands[5], meta


def calculate_indices(band3, band4, band5):
    """Calculate NDVI and NDBI indices with proper handling"""
    # Avoid division by zero and handle NaN values
    with np.errstate(divide='ignore', invalid='ignore'):
        # NDVI = (NIR - RED) / (NIR + RED)
        denominator_ndvi = band4 + band3
        ndvi = np.where(denominator_ndvi != 0, (band4 - band3) / denominator_ndvi, np.nan)

        # NDBI = (SWIR - NIR) / (SWIR + NIR)
        denominator_ndbi = band5 + band4
        ndbi = np.where(denominator_ndbi != 0, (band5 - band4) / denominator_ndbi, np.nan)

    # Replace inf and NaN with 0
    ndvi = np.nan_to_num(ndvi, nan=0, posinf=0, neginf=0)
    ndbi = np.nan_to_num(ndbi, nan=0, posinf=0, neginf=0)

    # Clip values to valid range
    ndvi = np.clip(ndvi, -1, 1)
    ndbi = np.clip(ndbi, -1, 1)

    print(f"NDVI statistics: min={ndvi.min():.3f}, max={ndvi.max():.3f}, mean={ndvi.mean():.3f}")
    print(f"NDBI statistics: min={ndbi.min():.3f}, max={ndbi.max():.3f}, mean={ndbi.mean():.3f}")

    return ndvi, ndbi


def create_false_color_composite(band2, band3, band4):
    """Create a properly normalized false color composite"""

    # Remove extreme values (2-98 percentile stretch)
    def stretch_band(band, percentiles=(2, 98)):
        p_low, p_high = np.nanpercentile(band, percentiles)
        band_stretched = np.clip(band, p_low, p_high)
        band_stretched = (band_stretched - p_low) / (p_high - p_low)
        return np.clip(band_stretched, 0, 1)

    # Apply stretching to each band
    red_stretched = stretch_band(band3)
    green_stretched = stretch_band(band2)
    nir_stretched = stretch_band(band4)

    # Create RGB composite (NIR, Red, Green for false color)
    return np.dstack([nir_stretched, red_stretched, green_stretched])


def save_individual_images(band2, band3, band4, ndvi, ndbi, false_color):
    """Save individual images as separate files"""
    print("\nSaving individual image files...")

    # 1. False Color Composite
    plt.figure(figsize=(10, 8))
    plt.imshow(false_color)
    plt.title("False Color Composite (NIR, Red, Green) - Raleigh, NC 1999")
    plt.axis('off')
    plt.tight_layout()
    false_color_path = os.path.join(raleigh_out_dir, "false_color_raleigh.png")
    plt.savefig(false_color_path, dpi=300, bbox_inches='tight')
    print(f"Saved False Color: {false_color_path}")
    plt.close()

    # 2. NDVI Map
    plt.figure(figsize=(10, 8))
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title("NDVI - Vegetation Index - Raleigh, NC 1999")
    plt.colorbar(label='NDVI Value', shrink=0.8)
    plt.axis('off')
    plt.tight_layout()
    ndvi_img_path = os.path.join(raleigh_out_dir, "ndvi_raleigh.png")
    plt.savefig(ndvi_img_path, dpi=300, bbox_inches='tight')
    print(f"Saved NDVI image: {ndvi_img_path}")
    plt.close()

    # 3. NDBI Map
    plt.figure(figsize=(10, 8))
    plt.imshow(ndbi, cmap='RdYlBu_r', vmin=-1, vmax=1)
    plt.title("NDBI - Built-up Index - Raleigh, NC 1999")
    plt.colorbar(label='NDBI Value', shrink=0.8)
    plt.axis('off')
    plt.tight_layout()
    ndbi_img_path = os.path.join(raleigh_out_dir, "ndbi_raleigh.png")
    plt.savefig(ndbi_img_path, dpi=300, bbox_inches='tight')
    print(f"Saved NDBI image: {ndbi_img_path}")
    plt.close()

    # 4. Natural Color Composite (if we have blue band)
    try:
        blue_bands = sorted(glob.glob(os.path.join(clipped_out_dir, "*_B1_raleigh.TIF")))
        if blue_bands:
            blue_band, _ = read_band(blue_bands[0])

            # Create natural color composite
            def stretch_natural(band, percentiles=(5, 95)):
                p_low, p_high = np.nanpercentile(band, percentiles)
                band_stretched = np.clip(band, p_low, p_high)
                band_stretched = (band_stretched - p_low) / (p_high - p_low)
                return np.clip(band_stretched, 0, 1)

            red_natural = stretch_natural(band3)
            green_natural = stretch_natural(band2)
            blue_natural = stretch_natural(blue_band)

            natural_color = np.dstack([red_natural, green_natural, blue_natural])

            plt.figure(figsize=(10, 8))
            plt.imshow(natural_color)
            plt.title("Natural Color Composite - Raleigh, NC 1999")
            plt.axis('off')
            plt.tight_layout()
            natural_color_path = os.path.join(raleigh_out_dir, "natural_color_raleigh.png")
            plt.savefig(natural_color_path, dpi=300, bbox_inches='tight')
            print(f"Saved Natural Color: {natural_color_path}")
            plt.close()
    except Exception as e:
        print(f"Could not create natural color composite: {e}")


def visualize_results(band2, band3, band4, ndvi, ndbi, false_color):
    """Visualize all results in a single figure"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # False Color Composite
    ax1 = axes[0, 0]
    ax1.imshow(false_color)
    ax1.set_title("False Color (NIR, Red, Green) - Raleigh, NC", fontsize=12)
    ax1.axis('off')

    # NDVI Map
    ax2 = axes[0, 1]
    im2 = ax2.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.set_title("NDVI - Vegetation Index", fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='NDVI Value')

    # NDBI Map
    ax3 = axes[1, 0]
    im3 = ax3.imshow(ndbi, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax3.set_title("NDBI - Built-up Index", fontsize=12)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='NDBI Value')

    # NDVI Histogram
    ax4 = axes[1, 1]
    ax4.hist(ndvi.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('NDVI Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('NDVI Distribution')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Landsat 5 Analysis - Raleigh, NC - 1999', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def save_outputs(ndvi, ndbi, meta):
    """Save NDVI and NDBI as GeoTIFF files"""
    print("\nSaving GeoTIFF files...")

    # Update metadata for single band output
    meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    # Save NDVI as GeoTIFF
    ndvi_path = os.path.join(raleigh_out_dir, "NDVI_Raleigh_1999.tif")
    with rasterio.open(ndvi_path, "w", **meta) as dest:
        dest.write(ndvi.astype(np.float32), 1)
    print(f"Saved NDVI GeoTIFF: {ndvi_path}")

    # Save NDBI as GeoTIFF
    ndbi_path = os.path.join(raleigh_out_dir, "NDBI_Raleigh_1999.tif")
    with rasterio.open(ndbi_path, "w", **meta) as dest:
        dest.write(ndbi.astype(np.float32), 1)
    print(f"Saved NDBI GeoTIFF: {ndbi_path}")

    # Also save a combined visualization
    combined_path = os.path.join(raleigh_out_dir, "combined_analysis_raleigh.png")

    return raleigh_out_dir


def main():
    """Main function to run Raleigh analysis"""
    print("=" * 60)
    print("LANDSAT 5 ANALYSIS - RALEIGH, NC - 1999")
    print("=" * 60)

    # Step 1: Clip bands to Raleigh area
    print("\n1. CLIPPING TO RALEIGH BOUNDARY")
    print("-" * 40)
    clip_success = clip_bands_to_raleigh()

    if not clip_success:
        print("Failed to clip bands. Exiting.")
        return

    # Step 2: Load Raleigh-clipped bands
    print("\n2. LOADING RALEIGH-CLIPPED BANDS")
    print("-" * 40)
    try:
        band1, band2, band3, band4, band5, band7, meta = load_raleigh_bands()
        print(f"\nLoaded bands successfully!")
        print(f"Image dimensions: {band3.shape}")
        print(f"Pixel values range:")
        print(f"  Red band (B3): [{band3.min():.2f}, {band3.max():.2f}]")
        print(f"  NIR band (B4): [{band4.min():.2f}, {band4.max():.2f}]")
    except Exception as e:
        print(f"Error loading bands: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Calculate indices
    print("\n3. CALCULATING INDICES")
    print("-" * 40)
    ndvi, ndbi = calculate_indices(band3, band4, band5)

    # Step 4: Create false color composite
    print("\n4. CREATING FALSE COLOR COMPOSITE")
    print("-" * 40)
    false_color = create_false_color_composite(band2, band3, band4)

    # Step 5: Save individual images
    print("\n5. SAVING INDIVIDUAL IMAGE FILES")
    print("-" * 40)
    save_individual_images(band2, band3, band4, ndvi, ndbi, false_color)

    # Step 6: Save GeoTIFF outputs
    print("\n6. SAVING GEOTIFF OUTPUTS")
    print("-" * 40)
    output_dir = save_outputs(ndvi, ndbi, meta)

    # Step 7: Visualize
    print("\n7. CREATING VISUALIZATION")
    print("-" * 40)
    visualize_results(band2, band3, band4, ndvi, ndbi, false_color)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles created:")
    print("1. raleigh_boundary.shp - Raleigh boundary shapefile")
    print("2. false_color_raleigh.png - False color composite")
    print("3. ndvi_raleigh.png - NDVI visualization")
    print("4. ndbi_raleigh.png - NDBI visualization")
    print("5. NDVI_Raleigh_1999.tif - NDVI GeoTIFF")
    print("6. NDBI_Raleigh_1999.tif - NDBI GeoTIFF")
    print("7. natural_color_raleigh.png - Natural color image (if available)")
    print("\nYou can find these files on your Desktop in the 'raleigh_analysis' folder.")


if __name__ == "__main__":
    main()