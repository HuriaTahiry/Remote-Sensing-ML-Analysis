import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RALEIGH_BBOX = {
    "min_lon": -78.70,
    "max_lon": -78.58,
    "min_lat": 35.73,
    "max_lat": 35.83
}

# Paths - UPDATE THIS TO YOUR DATA PATH
data_path = "/Users/huriatahiry/PycharmProjects/GIS/LC08_L2SP_015035_20231229_20240108_02_T1"
band_files = sorted(glob.glob(os.path.join(data_path, "*B[1-7].TIF")))

# Output directories
out_dir = "/Users/huriatahiry/Desktop/random_forest_classification"
os.makedirs(out_dir, exist_ok=True)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

def load_raster_bands(band_files, aoi):
    """Load all bands and clip to AOI"""
    bands_data = {}
    reference_crs = None
    
    for band_path in band_files:
        with rasterio.open(band_path) as src:
            if reference_crs is None:
                reference_crs = src.crs
                aoi_proj = aoi.to_crs(reference_crs)
            
            # Clip to AOI
            out_image, out_transform = mask(src, aoi_proj.geometry, crop=True)
            
            # Extract band number from filename
            band_num = os.path.basename(band_path).split('_B')[1][0]
            
            # Landsat SR data: scale from 0-10000 to 0-1
            data = out_image[0].astype('float32')
            data = data / 10000.0
            data = np.clip(data, 0, 1)
            
            bands_data[f'B{band_num}'] = data
    
    return bands_data, out_transform, reference_crs

def calculate_indices(bands):
    """Calculate NDVI, NDBI, NDWI, MNDWI indices"""
    indices = {}
    
    # NDVI = (NIR - Red) / (NIR + Red)
    nir = bands.get('B4')
    red = bands.get('B3')
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red + 1e-10)
        indices['NDVI'] = np.clip(ndvi, -1, 1)
    
    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    swir1 = bands.get('B5')
    with np.errstate(divide='ignore', invalid='ignore'):
        ndbi = (swir1 - nir) / (swir1 + nir + 1e-10)
        indices['NDBI'] = np.clip(ndbi, -1, 1)
    
    # NDWI = (Green - NIR) / (Green + NIR) for water
    green = bands.get('B2')
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir + 1e-10)
        indices['NDWI'] = np.clip(ndwi, -1, 1)
    
    # MNDWI = (Green - SWIR1) / (Green + SWIR1) - better for water
    with np.errstate(divide='ignore', invalid='ignore'):
        mndwi = (green - swir1) / (green + swir1 + 1e-10)
        indices['MNDWI'] = np.clip(mndwi, -1, 1)
    
    # Brightness (average of all bands)
    all_bands = [bands[f'B{i}'] for i in [2, 3, 4, 5, 7] if f'B{i}' in bands]
    indices['Brightness'] = np.mean(all_bands, axis=0)
    
    return indices

# ============================================================================
# STEP 2: CREATE TRAINING SAMPLES (MANUAL LABELING)
# ============================================================================

def create_training_samples(bands, indices, num_samples_per_class=300):
    """
    Manually label training samples based on spectral signatures.
    In real application, you'd use field data or digitized polygons.
    """
    np.random.seed(42)
    
    rows, cols = bands['B3'].shape
    total_pixels = rows * cols
    
    # Class definitions with spectral rules
    # These thresholds are based on typical spectral signatures
    classes = {
        1: {'name': 'Urban', 'color': 'red', 'rule': lambda ndbi, ndvi, brightness: (ndbi > 0.05) & (ndvi < 0.2) & (brightness > 0.15)},
        2: {'name': 'Vegetation', 'color': 'green', 'rule': lambda ndbi, ndvi, brightness: (ndvi > 0.3) & (ndbi < 0.1)},
        3: {'name': 'Water', 'color': 'blue', 'rule': lambda ndbi, ndvi, brightness: (indices['MNDWI'] > 0.1)},
        4: {'name': 'Bare Soil', 'color': 'brown', 'rule': lambda ndbi, ndvi, brightness: (ndvi < 0.15) & (ndvi > -0.05) & (brightness > 0.2) & (ndbi < 0.05)},
        5: {'name': 'Agriculture', 'color': 'yellow', 'rule': lambda ndbi, ndvi, brightness: (ndvi > 0.15) & (ndvi <= 0.3) & (brightness > 0.1)}
    }
    
    X_train = []
    y_train = []
    
    print("\n" + "="*60)
    print("CREATING TRAINING SAMPLES")
    print("="*60)
    
    for class_id, class_info in classes.items():
        print(f"\nSampling {class_info['name']}...")
        
        # Find pixels that satisfy the rule
        mask = class_info['rule'](
            indices['NDBI'], 
            indices['NDVI'], 
            indices['Brightness']
        )
        
        # Get indices of valid pixels
        valid_indices = np.where(mask.flatten())[0]
        
        if len(valid_indices) < num_samples_per_class:
            print(f"  Warning: Only {len(valid_indices)} pixels found for {class_info['name']}")
            sample_indices = valid_indices
        else:
            sample_indices = np.random.choice(valid_indices, num_samples_per_class, replace=False)
        
        # Extract features for each sample
        for idx in sample_indices:
            row = idx // cols
            col = idx % cols
            
            # Feature vector: spectral bands + indices
            features = [
                bands['B2'][row, col],  # Blue
                bands['B3'][row, col],  # Green  
                bands['B4'][row, col],  # Red
                bands['B5'][row, col],  # NIR
                indices['NDVI'][row, col],
                indices['NDBI'][row, col],
                indices['NDWI'][row, col],
                indices['MNDWI'][row, col],
                indices['Brightness'][row, col]
            ]
            
            X_train.append(features)
            y_train.append(class_id)
        
        print(f"  Added {len(sample_indices)} samples for {class_info['name']}")
    
    return np.array(X_train), np.array(y_train), classes

# ============================================================================
# STEP 3: TRAIN RANDOM FOREST CLASSIFIER
# ============================================================================

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier with hyperparameter tuning"""
    
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Split into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize Random Forest with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        max_depth=15,            # Maximum depth of each tree
        min_samples_split=5,     # Minimum samples to split a node
        min_samples_leaf=2,      # Minimum samples in a leaf
        max_features='sqrt',     # Features to consider for best split
        class_weight='balanced', # HANDLES CLASS IMBALANCE automatically!
        random_state=42,
        n_jobs=-1                # Use all CPU cores
    )
    
    # Train the model
    print("Training model...")
    rf.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_pred = rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Number of Trees: {rf.n_estimators}")
    print(f"  Feature Importance: {rf.feature_importances_}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return rf

# ============================================================================
# STEP 4: APPLY MODEL TO ENTIRE STUDY AREA
# ============================================================================

def predict_full_image(bands, indices, model, rows, cols):
    """Apply trained Random Forest model to every pixel"""
    
    print("\n" + "="*60)
    print("APPLYING MODEL TO STUDY AREA")
    print("="*60)
    print(f"Processing {rows:,} × {cols:,} = {rows*cols:,} pixels...")
    
    # Prepare feature matrix for all pixels
    num_features = 9
    X_all = np.zeros((rows * cols, num_features))
    
    # Flatten all bands and indices
    X_all[:, 0] = bands['B2'].flatten()  # Blue
    X_all[:, 1] = bands['B3'].flatten()  # Green
    X_all[:, 2] = bands['B4'].flatten()  # Red
    X_all[:, 3] = bands['B5'].flatten()  # NIR
    X_all[:, 4] = indices['NDVI'].flatten()
    X_all[:, 5] = indices['NDBI'].flatten()
    X_all[:, 6] = indices['NDWI'].flatten()
    X_all[:, 7] = indices['MNDWI'].flatten()
    X_all[:, 8] = indices['Brightness'].flatten()
    
    # Remove NaN values (replace with 0)
    X_all = np.nan_to_num(X_all, nan=0)
    
    # Predict in batches to avoid memory issues
    batch_size = 100000
    predictions = np.zeros(rows * cols)
    
    for i in range(0, len(X_all), batch_size):
        batch = X_all[i:i+batch_size]
        predictions[i:i+batch_size] = model.predict(batch)
        
        if i % (batch_size * 5) == 0:
            print(f"  Processed {i+len(batch):,} / {len(X_all):,} pixels...")
    
    # Reshape back to image shape
    classification_map = predictions.reshape(rows, cols)
    
    return classification_map

# ============================================================================
# STEP 5: VISUALIZE RESULTS
# ============================================================================

def visualize_results(classification_map, classes, bands, indices, year="2023"):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Classification Map
    ax1 = axes[0, 0]
    cmap_colors = ['red', 'green', 'blue', 'brown', 'yellow']
    class_cmap = plt.matplotlib.colors.ListedColormap(cmap_colors[:len(classes)])
    
    im1 = ax1.imshow(classification_map, cmap=class_cmap, vmin=1, vmax=len(classes))
    ax1.set_title(f"Random Forest Classification\n{year} - {len(classes)} Classes", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Add legend
    legend_elements = [plt.matplotlib.patches.Patch(facecolor=c, label=classes[i+1]['name']) 
                       for i, c in enumerate(cmap_colors[:len(classes)])]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # 2. NDVI (for reference)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(indices['NDVI'], cmap='RdYlGn', vmin=-0.5, vmax=0.8)
    ax2.set_title("NDVI - Vegetation Index", fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.7, label='NDVI')
    
    # 3. NDBI (for reference)
    ax3 = axes[0, 2]
    im3 = ax3.imshow(indices['NDBI'], cmap='OrRd', vmin=-0.3, vmax=0.5)
    ax3.set_title("NDBI - Built-up Index", fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.7, label='NDBI')
    
    # 4. Class Distribution Pie Chart
    ax4 = axes[1, 0]
    unique, counts = np.unique(classification_map, return_counts=True)
    total_pixels = counts.sum()
    percentages = counts / total_pixels * 100
    
    ax4.pie(percentages, labels=[classes[int(u)]['name'] for u in unique], 
            colors=cmap_colors[:len(unique)], autopct='%1.1f%%', startangle=90)
    ax4.set_title("Land Cover Distribution", fontsize=12, fontweight='bold')
    
    # 5. False Color Composite
    ax5 = axes[1, 1]
    false_color = np.dstack([
        bands['B4'] / np.percentile(bands['B4'], 98),  # NIR as Red
        bands['B3'] / np.percentile(bands['B3'], 98),  # Red as Green
        bands['B2'] / np.percentile(bands['B2'], 98)   # Green as Blue
    ])
    false_color = np.clip(false_color, 0, 1)
    ax5.imshow(false_color)
    ax5.set_title("False Color Composite (NIR, Red, Green)", fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. Feature Importance
    ax6 = axes[1, 2]
    feature_names = ['Blue', 'Green', 'Red', 'NIR', 'NDVI', 'NDBI', 'NDWI', 'MNDWI', 'Brightness']
    importances = model.feature_importances_
    indices_sorted = np.argsort(importances)[::-1]
    
    ax6.barh(range(len(feature_names)), importances[indices_sorted])
    ax6.set_yticks(range(len(feature_names)))
    ax6.set_yticklabels([feature_names[i] for i in indices_sorted])
    ax6.set_xlabel('Importance')
    ax6.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)
    
    plt.suptitle(f"Land Cover Classification using Random Forest\nRaleigh, NC - {year}", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(out_dir, f"random_forest_classification_{year}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nVisualization saved to: {output_path}")
    
    return output_path

# ============================================================================
# STEP 6: SAVE OUTPUTS
# ============================================================================

def save_geotiff(classification_map, transform, crs, output_name):
    """Save classification map as GeoTIFF"""
    
    output_path = os.path.join(out_dir, output_name)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=classification_map.shape[0],
        width=classification_map.shape[1],
        count=1,
        dtype=classification_map.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(classification_map.astype('int16'), 1)
    
    print(f"GeoTIFF saved to: {output_path}")
    return output_path

def save_class_statistics(classification_map, classes):
    """Save area statistics for each class"""
    
    # Assuming 30m pixel resolution (Landsat)
    pixel_area_ha = (30 * 30) / 10000  # Convert to hectares
    
    unique, counts = np.unique(classification_map, return_counts=True)
    
    print("\n" + "="*60)
    print("LAND COVER STATISTICS")
    print("="*60)
    print(f"{'Class':<15} {'Pixels':<12} {'Area (ha)':<12} {'Percentage':<10}")
    print("-"*60)
    
    stats = []
    for u, c in zip(unique, counts):
        area_ha = c * pixel_area_ha
        percentage = (c / counts.sum()) * 100
        class_name = classes[int(u)]['name']
        print(f"{class_name:<15} {c:<12,} {area_ha:<12.2f} {percentage:<10.2f}%")
        stats.append({'Class': class_name, 'Pixels': c, 'Area_ha': area_ha, 'Percentage': percentage})
    
    # Save statistics to file
    import csv
    stats_path = os.path.join(out_dir, "classification_statistics.csv")
    with open(stats_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Class', 'Pixels', 'Area_ha', 'Percentage'])
        writer.writeheader()
        writer.writerows(stats)
    
    print(f"\nStatistics saved to: {stats_path}")
    
    return stats

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main workflow for Random Forest land cover classification"""
    
    print("="*70)
    print("RANDOM FOREST LAND COVER CLASSIFICATION")
    print("Raleigh, North Carolina")
    print("="*70)
    
    year = "2023"  # Extract from filename if needed
    
    # Create AOI
    print("\n1. Creating study area boundary...")
    aoi = gpd.GeoDataFrame(
        {'geometry': [box(RALEIGH_BBOX["min_lon"], RALEIGH_BBOX["min_lat"],
                         RALEIGH_BBOX["max_lon"], RALEIGH_BBOX["max_lat"])]},
        crs='EPSG:4326'
    )
    
    # Load raster bands
    print("\n2. Loading Landsat bands...")
    bands, transform, crs = load_raster_bands(band_files, aoi)
    print(f"   Bands loaded: {list(bands.keys())}")
    print(f"   Image shape: {bands['B3'].shape}")
    
    # Calculate spectral indices
    print("\n3. Calculating spectral indices...")
    indices = calculate_indices(bands)
    print(f"   Indices calculated: {list(indices.keys())}")
    
    # Create training samples
    rows, cols = bands['B3'].shape
    X_train, y_train, classes = create_training_samples(bands, indices, num_samples_per_class=300)
    print(f"\n   Total training samples: {len(X_train)}")
    print(f"   Feature dimensions: {X_train.shape[1]}")
    
    # Train Random Forest
    global model  # Make model accessible for visualization
    model = train_random_forest(X_train, y_train)
    
    # Apply model to full image
    classification_map = predict_full_image(bands, indices, model, rows, cols)
    
    # Visualize results
    print("\n4. Creating visualizations...")
    visualize_results(classification_map, classes, bands, indices, year)
    
    # Save outputs
    print("\n5. Saving outputs...")
    save_geotiff(classification_map, transform, crs, f"landcover_classification_{year}.tif")
    save_class_statistics(classification_map, classes)
    
    # Save model for future use
    model_path = os.path.join(out_dir, "random_forest_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "="*70)
    print("CLASSIFICATION COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {out_dir}")
    print("\nOutput files:")
    print("  1. random_forest_classification_2023.png - Visualization")
    print("  2. landcover_classification_2023.tif - GeoTIFF map")
    print("  3. classification_statistics.csv - Area statistics")
    print("  4. random_forest_model.pkl - Trained model")
    
    # Class distribution summary
    print("\n" + "="*70)
    print("CLASSIFICATION SCHEME")
    print("="*70)
    for class_id, info in classes.items():
        print(f"  Class {class_id}: {info['name']} ({info['color']})")

if __name__ == "__main__":
    main()
