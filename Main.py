# %%
import os
import numpy as np
import rasterio
import pickle
import skimage.feature

# %%

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# %% [markdown]
# Loaing S2 Data and Reading Shape

# %%
import os
import numpy as np
import pickle

# Correct file path
files_path = "c:\\Users\\Dell\\Desktop\\Sentinal 2\\"

# Check file sizes
file_sizes = {f: os.path.getsize(files_path + f) for f in os.listdir(files_path)}
print("File sizes (in bytes):", file_sizes)

# Load a small portion of the file to get the metadata
temp_bands = np.load(files_path + 'BANDS.npy', mmap_mode='r')
shape = temp_bands.shape
dtype = temp_bands.dtype
del temp_bands  # Release the temporary memory-mapped array

# Using memory mapping to load large .npy file
bands = np.memmap(files_path + 'BANDS.npy', dtype=dtype, mode='r', shape=shape)
clp = np.load(files_path + 'CLP.npy')
is_data = np.load(files_path + 'IS_DATA.npy')
norm_factors = np.load(files_path + 'NORM_FACTORS.npy')
scl = np.load(files_path + 'SCL.npy')

# Loading .pkl files with error handling
def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

bbox = load_pickle_file(files_path + 'bbox.pkl')
meta_info = load_pickle_file(files_path + 'meta_info.pkl')
timestamp = load_pickle_file(files_path + 'timestamp.pkl')

# Exploring the data
print("Bands shape:", bands.shape)
print("Bands size (in bytes):", bands.nbytes)
print("File size of BANDS.npy:", os.path.getsize(files_path + 'BANDS.npy'))
print("CLP shape:", clp.shape)
print("IS_DATA shape:", is_data.shape)
print("Normalization factors shape:", norm_factors.shape)
print("SCL shape:", scl.shape)

print("Bounding box:", bbox)
print("Metadata:", meta_info)
print("Timestamp:", timestamp)


# %% [markdown]
# Visualize S2

# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the file path to your bands data
file_path = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\BANDS.npy'

# Load the data using memory mapping
bands = np.load(file_path, mmap_mode='r')

# Function to normalize and extract RGB channels for a batch of images
def process_batch(bands, start, end):
    # Extract Red, Green, Blue bands (indices might need adjustment based on your data)
    red = bands[start:end, ..., 3]  # Assuming index 3 is Red
    green = bands[start:end, ..., 2]  # Assuming index 2 is Green
    blue = bands[start:end, ..., 1]  # Assuming index 1 is Blue

    # Stack and normalize
    rgb_batch = np.stack([red, green, blue], axis=-1)
    max_val = np.max(rgb_batch, axis=(1, 2, 3), keepdims=True)
    rgb_normalized = (rgb_batch / max_val).astype(float)
    
    return rgb_normalized

# Display function for RGB images
def display_images(rgb_images, start):
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(rgb_images):
        plt.subplot(1, len(rgb_images), i+1)
        plt.imshow(img)
        plt.title(f'Time Step: {start + i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Process and display in batches
batch_size = 10  # Process 10 time steps at a time
num_steps = bands.shape[0]

for start in range(0, num_steps, batch_size):
    end = min(start + batch_size, num_steps)
    rgb_images = process_batch(bands, start, end)
    display_images(rgb_images, start)


# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the file path to your bands data
file_path = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\BANDS.npy'

# Load the data using memory mapping
bands = np.load(file_path, mmap_mode='r')

# Function to normalize and enhance brightness for RGB channels for a batch of images
def process_batch(bands, start, end, brightness_factor=1.5):
    # Extract and stack RGB bands
    rgb_batch = bands[start:end, ..., [3, 2, 1]]  # Adjust indices if necessary
    max_val = np.max(rgb_batch, axis=(1, 2, 3), keepdims=True)
    rgb_normalized = (rgb_batch / max_val * 255 * brightness_factor).astype(np.uint8)
    rgb_clipped = np.clip(rgb_normalized, 0, 255)  # Ensure values remain within byte range
    return rgb_clipped

# Efficiently display a single representative image per batch to check output
def display_sample_image(rgb_image, step):
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title(f'Time Step: {step}')
    plt.axis('off')
    plt.show()

# Process and display in batches, visualizing only one image per batch for efficiency
batch_size = 10  # Adjust based on memory constraints and desired processing load
num_steps = bands.shape[0]

for start in range(0, num_steps, batch_size):
    end = min(start + batch_size, num_steps)
    rgb_images = process_batch(bands, start, end)
    display_sample_image(rgb_images[0], start)  # Display only the first image of each batch


# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the file path to your bands data
file_path = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\BANDS.npy'

# Load the data using memory mapping
bands = np.load(file_path, mmap_mode='r')

# Function to normalize and enhance brightness for RGB channels for specific images
def process_image(bands, index, brightness_factor=2.6):  # Increased brightness factor
    # Extract RGB bands for a single image
    rgb_image = bands[index, ..., [3, 2, 1]]  # Adjust indices if necessary for RGB channels
    max_val = np.max(rgb_image)
    rgb_normalized = (rgb_image / max_val * 255 * brightness_factor).astype(np.uint8)
    rgb_clipped = np.clip(rgb_normalized, 0, 255)  # Ensure values remain within byte range
    return rgb_clipped.transpose((1, 2, 0))  # Transpose the axes to match (2400, 2400, 3)

# Display function for a single image
def display_image(rgb_image, index):
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title(f'Time Step: {index}')
    plt.axis('off')
    plt.show()

# Specific timestamps to display
timestamps = [94, 133, 135, 139]
for ts in timestamps:
    rgb_image = process_image(bands, ts)
    display_image(rgb_image, ts)


# %% [markdown]
# Shape of S2 Data

# %%
import numpy as np

# Load data
bands = np.load('c:\\Users\\Dell\\Desktop\\Sentinal 2\\BANDS.npy', mmap_mode='r')

# Print information about the array
print(f"Shape of bands: {bands.shape}")
print(f"Size of bands: {bands.size}")
print(f"Data type of bands: {bands.dtype}")

# %%
# Calculate the number of bands (assuming 2400x2400 pixels and 144 time steps)
num_bands = 9953280000 // (2400 * 2400 * 144)

print(f"Calculated number of bands: {num_bands}")

# Try reshaping with the calculated number of bands
try:
    reshaped_bands = bands.reshape(144, num_bands, 2400, 2400)
    print("Reshape successful")
    print(f"New shape: {reshaped_bands.shape}")
except ValueError as e:
    print(f"Reshape failed: {e}")

# %% [markdown]
# Analyzing Invalid Pixels in chunks

# %%
import numpy as np

def analyze_invalid_pixels_chunk(chunk):
    # Find invalid (NaN or Inf) pixels in the chunk
    invalid_mask = np.isnan(chunk) | np.isinf(chunk)
    
    # If there are no invalid pixels in this chunk, return
    if not np.any(invalid_mask):
        return 0, 0
    
    total_invalid = np.sum(invalid_mask)
    fixable_invalid = 0
    
    # Iterate through each band in the chunk
    for band in range(chunk.shape[-1]):
        # Find invalid pixels in this band
        invalid_pixels = np.where(invalid_mask[..., band])
        
        if len(invalid_pixels[0]) > 0:
            print(f"Found {len(invalid_pixels[0])} invalid pixels in band {band}")
            
            # Check if these pixels can be fixed
            for i, j, k in zip(*invalid_pixels):
                i_min, i_max = max(0, i-1), min(chunk.shape[0], i+2)
                j_min, j_max = max(0, j-1), min(chunk.shape[1], j+2)
                k_min, k_max = max(0, k-1), min(chunk.shape[2], k+2)
                
                neighborhood = chunk[i_min:i_max, j_min:j_max, k_min:k_max, band]
                valid_neighbors = neighborhood[~(np.isnan(neighborhood) | np.isinf(neighborhood))]
                
                if len(valid_neighbors) > 0:
                    fixable_invalid += 1
    
    return total_invalid, fixable_invalid

def analyze_data(file_path, chunk_size=10):
    # Open the file in read-only mode
    bands = np.load(file_path, mmap_mode='r')
    shape = bands.shape
    total_invalid = 0
    total_fixable = 0
    
    try:
        # Process in chunks
        for i in range(0, shape[0], chunk_size):
            print(f"Analyzing chunk {i//chunk_size + 1} of {-(-shape[0]//chunk_size)}")
            chunk = bands[i:i+chunk_size]
            chunk_invalid, chunk_fixable = analyze_invalid_pixels_chunk(chunk)
            total_invalid += chunk_invalid
            total_fixable += chunk_fixable
    
    finally:
        # Make sure to close the mmap
        bands._mmap.close()
    
    print("Analysis complete.")
    print(f"Total invalid pixels: {total_invalid}")
    print(f"Fixable invalid pixels: {total_fixable}")
    print(f"Unfixable invalid pixels: {total_invalid - total_fixable}")

# File path
file_path = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\BANDS.npy'

# Analyze the data
analyze_data(file_path)

# Print array info
bands = np.load(file_path, mmap_mode='r')
print(f"Shape of array: {bands.shape}")
print(f"Data type of array: {bands.dtype}")
bands._mmap.close()

# %% [markdown]
# Normalizing Bands in Chunks

# %%
import numpy as np

def normalize_band(band):
    min_val = np.min(band)
    max_val = np.max(band)
    return ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def normalize_bands_chunked(bands, chunk_size=100):
    shape = bands.shape
    normalized_bands = np.memmap('normalized_bands_temp.npy', dtype=np.uint8, mode='w+', shape=shape)
    
    for z in range(0, shape[0], chunk_size):
        print(f"Processing chunk: Z({z}-{min(z + chunk_size, shape[0])})")
        for y in range(0, shape[1], chunk_size):
            for x in range(0, shape[2], chunk_size):
                chunk = bands[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]
                for i in range(chunk.shape[-1]):
                    normalized_bands[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size, i] = normalize_band(chunk[..., i])
    
    return normalized_bands

# Load data (using memory mapping)
bands = np.load('c:\\Users\\Dell\\Desktop\\Sentinal 2\\BANDS.npy', mmap_mode='r')

# Normalize the bands in chunks
print("Normalizing bands in chunks...")
normalized_bands = normalize_bands_chunked(bands)
print("Normalization complete.")

# Save the normalized data
np.save('c:\\Users\\Dell\\Desktop\\Sentinal 2\\NORMALIZED_BANDS.npy', normalized_bands)

# Print array info
print(f"Shape of normalized array: {normalized_bands.shape}")
print(f"Data type of normalized array: {normalized_bands.dtype}")
bands._mmap.close()


# %% [markdown]
# Pixel Range Check:code snippet loads these normalized bands and checks whether all pixel values fall within the expected range of [0, 255]

# %%
import numpy as np
import time

def check_pixel_range_chunked(bands, chunk_size=100):
    shape = bands.shape
    within_range = True
    
    for z in range(0, shape[0], chunk_size):
        print(f"Checking chunk: Z({z}-{min(z + chunk_size, shape[0])})")
        for y in range(0, shape[1], chunk_size):
            for x in range(0, shape[2], chunk_size):
                chunk = bands[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]
                if not np.all((chunk >= 0) & (chunk <= 255)):
                    within_range = False
                    out_of_range_indices = np.where((chunk < 0) | (chunk > 255))
                    print(f"Out-of-range values found in chunk: Z({z}-{z+chunk_size}), Y({y}-{y+chunk_size}), X({x}-{x+chunk_size})")
                    print("Indices of out-of-range values:", out_of_range_indices)
                    print("Out-of-range values:", chunk[out_of_range_indices])
                    break
            if not within_range:
                break
        if not within_range:
            break
    
    if within_range:
        print("All pixel values are within the expected range (0-255).")
    else:
        print("Some pixel values are out of the expected range (0-255).")

# Load the normalized data
normalized_bands = np.load('c:\\Users\\Dell\\Desktop\\Sentinal 2\\NORMALIZED_BANDS.npy')

# Measure the time taken to check the range
start_time = time.time()

# Check pixel values in chunks
check_pixel_range_chunked(normalized_bands)

end_time = time.time()
print(f"Time taken to check all pixel values: {end_time - start_time} seconds")


# %% [markdown]
# Preprocessing: applies a cloud mask to the data to remove or mask out cloud-covered pixels
# The mask is applied to the data by setting cloud pixels to `0` and clear pixels to `1`. 

# %%
import numpy as np
from tqdm import tqdm

def create_cloud_mask(chunk):
    # Assuming RGB bands are band 2, 3, 4 (indices 1, 2, 3)
    cloud_mask = (chunk[..., 1] > 240) & (chunk[..., 2] > 230) & (chunk[..., 3] > 220)
    return cloud_mask

def apply_cloud_mask(chunk, cloud_mask):
    # Create a mask where 0 represents clouds and 1 represents clear pixels
    clear_pixels = (~cloud_mask).astype(np.uint8)
    # Multiply each band by the clear_pixels mask
    masked_chunk = chunk * clear_pixels[..., np.newaxis]
    return masked_chunk

def process_in_chunks(input_file, output_file, chunk_size=1000):
    # Load the full array to get its shape
    full_array = np.load(input_file, mmap_mode='r')
    
    # Create an empty output array of the same shape and dtype
    output_array = np.empty(full_array.shape, dtype=full_array.dtype)
    
    # Calculate the number of chunks
    num_chunks = int(np.ceil(full_array.shape[0] / chunk_size))
    
    # Process the data in chunks
    for i in tqdm(range(num_chunks), desc="Processing chunks"):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, full_array.shape[0])
        
        # Load a chunk of data
        chunk = full_array[start:end]
        
        # Create and apply cloud mask for the chunk
        cloud_mask = create_cloud_mask(chunk)
        masked_chunk = apply_cloud_mask(chunk, cloud_mask)
        
        # Store the processed chunk in the output array
        output_array[start:end] = masked_chunk
    
    # Save the processed array
    np.save(output_file, output_array)
    print("Processing complete. Output saved.")

# Usage
input_file = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\NORMALIZED_BANDS.npy'
output_file = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\MASKED_NORMALIZED_BANDS.npy'
process_in_chunks(input_file, output_file, chunk_size=1000)

# %% [markdown]
# Calculates and compares the sizes of the original and masked data files.

# %%
import os

original_file = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\NORMALIZED_BANDS.npy'
masked_file = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\MASKED_NORMALIZED_BANDS.npy'

original_size = os.path.getsize(original_file) / (1024 * 1024 * 1024)  # Size in GB
masked_size = os.path.getsize(masked_file) / (1024 * 1024 * 1024)  # Size in GB

print(f"Original file size: {original_size:.2f} GB")
print(f"Masked file size: {masked_size:.2f} GB")

# %% [markdown]
# Masked Normalized Band Visualization

# %%
import numpy as np
import matplotlib.pyplot as plt

#File path to bands data
file_path = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\MASKED_NORMALIZED_BANDS.npy'

# Load the data using memory mapping
bands = np.load(file_path, mmap_mode='r')

# Function to normalize and extract RGB channels for a batch of images
def process_batch(bands, start, end):
    # Extract Red, Green, Blue bands (indices might need adjustment based on your data)
    red = bands[start:end, ..., 3]  # index 3 is Red
    green = bands[start:end, ..., 2]  # index 2 is Green
    blue = bands[start:end, ..., 1]  # index 1 is Blue

    # Stack and normalize
    rgb_batch = np.stack([red, green, blue], axis=-1)
    max_val = np.max(rgb_batch, axis=(1, 2, 3), keepdims=True)
    rgb_normalized = (rgb_batch / max_val).astype(float)
    
    return rgb_normalized

# Display function for RGB images
def display_images(rgb_images, start):
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(rgb_images):
        plt.subplot(1, len(rgb_images), i+1)
        plt.imshow(img)
        plt.title(f'Time Step: {start + i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Process and display in batches
batch_size = 10  # Process 10 time steps at a time
num_steps = bands.shape[0]

for start in range(0, num_steps, batch_size):
    end = min(start + batch_size, num_steps)
    rgb_images = process_batch(bands, start, end)
    display_images(rgb_images, start)


# %% [markdown]
# This code loads and inspects the normalized bands data. It prints the shape of the data and displays the values for specific bands, including Blue, Green, and Red. Additionally, it shows a sample of the Blue band to visually inspect the content.

# %% [markdown]
# ***Note: Visulaize this final data of s2

# %%
import numpy as np

# File path to bands data (replace with your actual path)
file_path = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\NORMALIZED_BANDS.npy'

# Load the bands data
bands_data = np.load(file_path)

# Print shape of the bands data
print("Shape of bands data:", bands_data.shape)

# RGB bands are band 2, 3, 4 (indices 1, 2, 3)
print("Band indices:")
print("Band 1:", bands_data[..., 0])  # Replace ... with appropriate slicing for spatial dimensions
print("Band 2 (Blue):", bands_data[..., 1])
print("Band 3 (Green):", bands_data[..., 2])
print("Band 4 (Red):", bands_data[..., 3])

# Optionally, print a sample of the bands to visually inspect their content
print("\nSample of Band 2 (Blue):")
print(bands_data[..., 1][:10, :10])  # Print a 10x10 sample; adjust as needed




# %% [markdown]
# Plots histograms of pixel values for each band in the normalized Sentinel-2 data, subplot for each band to show its frequency distribution.
# 

# %%
import numpy as np

# Assume `cloud_mask` is a Boolean array where True indicates a cloud
data_loss_percent = np.mean(cloud_mask) * 100  
data_remaining_percent = 100 - data_loss_percent  

print(f"Data lost to clouds: {data_loss_percent:.2f}%")
print(f"Data remaining after masking: {data_remaining_percent:.2f}%")


# %% [markdown]
# Distribution

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_band_histograms(data, num_bands, sample_size=1000):
    plt.figure(figsize=(15, 10))
    
    
    indices = np.random.choice(data.shape[0] * data.shape[1], size=sample_size, replace=False)
    
    for i in range(num_bands):
        # Flatten the band data to one dimension and sample
        band_data = data[..., i].flatten()[indices]
        
        plt.subplot(3, 4, i + 1)  
        plt.hist(band_data, bins=50, color='gray', alpha=0.7)
        plt.title(f'Band {i+1} Histogram')
        plt.xlabel('Pixel Values')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Load normalized bands data
file_path = 'c:\\Users\\Dell\\Desktop\\Sentinal 2\\NORMALIZED_BANDS.npy'
normalized_bands = np.load(file_path)

# Plot histograms for each band
plot_band_histograms(normalized_bands, num_bands=12)  


# %% [markdown]
# visualizes and compares normalized and cloud-masked Sentinel-2 data, processing it in chunks for multiple time steps. It overlays cloud masks on RGB composites and calculates the percentage of data lost due to cloud masking. Additionally, it provides overall and per-band statistics on data loss and remaining data.

# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize

# Load normalized and masked data
normalized_data = np.load('c:\\Users\\Dell\\Desktop\\Sentinal 2\\NORMALIZED_BANDS.npy', mmap_mode='r')
masked_data = np.load('c:\\Users\\Dell\\Desktop\\Sentinal 2\\MASKED_NORMALIZED_BANDS.npy', mmap_mode='r')

def downsample_image(image, max_size=1000):
    """Downsample an image to a maximum size while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    
    if h > w:
        new_h, new_w = max_size, int(max_size * w / h)
    else:
        new_h, new_w = int(max_size * h / w), max_size
    
    return resize(image, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(image.dtype)

def visualize_comparison_with_mask(norm_chunk, masked_chunk, start_step):
    num_steps = norm_chunk.shape[0]
    fig, axes = plt.subplots(3, num_steps, figsize=(5*num_steps, 15))
    
    for i in range(num_steps):
        # Normalized data
        rgb_norm = np.transpose(norm_chunk[i, :, :, [3,2,1]], (1, 2, 0))
        rgb_norm_small = downsample_image(rgb_norm)
        axes[0, i].imshow(rgb_norm_small / 255.0)
        axes[0, i].set_title(f'Normalized - Step {start_step + i}')
        axes[0, i].axis('off')
        
        # Masked data
        rgb_masked = np.transpose(masked_chunk[i, :, :, [3,2,1]], (1, 2, 0))
        rgb_masked_small = downsample_image(rgb_masked)
        axes[1, i].imshow(rgb_masked_small / 255.0)
        axes[1, i].set_title(f'Masked - Step {start_step + i}')
        axes[1, i].axis('off')
        
        # Cloud mask overlay
        cloud_mask = np.any(masked_chunk[i] == 0, axis=-1)
        cloud_mask_small = resize(cloud_mask, rgb_norm_small.shape[:2], order=0, anti_aliasing=False, preserve_range=True).astype(bool)
        rgb_with_mask = rgb_norm_small.copy() / 255.0
        rgb_with_mask[cloud_mask_small] = [1, 0, 1]  # Pink color for masked areas
        axes[2, i].imshow(rgb_with_mask)
        axes[2, i].set_title(f'Cloud Mask - Step {start_step + i}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize in chunks
chunk_size = 7  # Process 7 time steps at once
total_steps = normalized_data.shape[0]

for start in tqdm(range(0, total_steps, chunk_size), desc="Processing chunks"):
    end = min(start + chunk_size, total_steps)
    norm_chunk = normalized_data[start:end]
    masked_chunk = masked_data[start:end]
    visualize_comparison_with_mask(norm_chunk, masked_chunk, start)
    
    # Optional: Add a prompt to continue to the next chunk
    if end < total_steps:
        input("Press Enter to continue to the next chunk...")

# Calculate overall statistics
def calculate_data_loss(norm_data, masked_data):
    total_pixels = norm_data.size
    masked_pixels = np.sum(masked_data == 0)
    loss_percentage = (masked_pixels / total_pixels) * 100
    remaining_percentage = 100 - loss_percentage
    return loss_percentage, remaining_percentage

loss_percentage, remaining_percentage = calculate_data_loss(normalized_data, masked_data)
print(f"Percentage of data lost due to cloud masking: {loss_percentage:.2f}%")


print(f"Percentage of data remaining after cloud masking: {remaining_percentage:.2f}%")

# Calculate per-band statistics
for band in range(12):
    band_loss = calculate_data_loss(normalized_data[..., band], masked_data[..., band])
    print(f"Band {band} - Data lost: {band_loss[0]:.2f}%, Data remaining: {band_loss[1]:.2f}%")

# %% [markdown]
# Load S1 Data

# %%
import os  # Import the os module
# Define the base directory for the data
base_dir = r'c:\\Users\\Dell\\Desktop\\Project downlaods'  # Update this to your actual directory path

# Paths to specific data files

mask_dir = os.path.join(base_dir, 'mask')
bbox_file = os.path.join(base_dir, 'bbox.pkl')
meta_info_file = os.path.join(base_dir, 'meta_info.pkl')
timestamp_file = os.path.join(base_dir, 'timestamp.pkl')

# Load polarization data
vv_path = os.path.join(base_dir, 'VV.npy')
vh_path = os.path.join(base_dir, 'VH.npy')
vv_data = np.load(vv_path)
vh_data = np.load(vh_path)

# %% [markdown]
# Shape of S1 data

# %%
import pickle
# Load mask data

mask_path = os.path.join(base_dir, 'MASK.npy')  
mask_data = np.load(mask_path)

# Load bounding box information
with open(bbox_file, 'rb') as f:
    bbox = pickle.load(f)


# Print basic information
print("VV Data Shape:", vv_data.shape)
print("VH Data Shape:", vh_data.shape)
print("Mask Data Shape:", mask_data.shape)
print("Bounding Box:", bbox)



# %%
# Example code to replace NaNs and infinities
vv_data[np.isnan(vv_data)] = 0
vv_data[np.isinf(vv_data)] = 0


# %%
def chunked_processing(data_path, chunk_size, operation):
    """
    Process large data in chunks.
    """
    data = np.load(data_path, mmap_mode='r')
    shape = data.shape
    processed_data = np.zeros(shape, dtype=np.float32)
    
    for i in tqdm(range(0, shape[0], chunk_size)):
        chunk = data[i:i+chunk_size].astype(np.float32)
        processed_chunk = operation(chunk)
        processed_data[i:i+chunk_size] = processed_chunk
    
    return processed_data

# %% [markdown]
# Preprocessing: Lee filter to reduce speckle noise. processes data in chunks to handle large datasets efficiently. 

# %%
import numpy as np
from scipy import ndimage
from tqdm import tqdm

def lee_filter(img, size=5):
    def filter_3d(img_3d):
        img_3d = np.nan_to_num(img_3d, nan=0, posinf=0, neginf=0)  # Replace NaN and inf values
        img_mean = ndimage.uniform_filter(img_3d, (size, size, 1))
        img_sqr_mean = ndimage.uniform_filter(img_3d**2, (size, size, 1))
        img_variance = np.maximum(img_sqr_mean - img_mean**2, 0)  # Ensure non-negative variance
        overall_variance = np.maximum(np.var(img_3d), 1e-10)  # Avoid division by zero
        
        denominator = img_variance + overall_variance
        weight = np.divide(img_variance, denominator, out=np.zeros_like(img_variance), where=denominator!=0)
        img_filtered = img_mean + weight * (img_3d - img_mean)
        
        return img_filtered

    filtered_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        filtered_img[i] = filter_3d(img[i])
        
        # Diagnostic information
        if i == 0:  # Print info for the first slice
            print(f"First slice stats:")
            print(f"  Min: {np.min(filtered_img[i])}")
            print(f"  Max: {np.max(filtered_img[i])}")
            print(f"  Mean: {np.mean(filtered_img[i])}")
            print(f"  NaN count: {np.isnan(filtered_img[i]).sum()}")
            print(f"  Inf count: {np.isinf(filtered_img[i]).sum()}")
    
    return filtered_img

def chunked_processing(data_path, chunk_size, operation):
    data = np.load(data_path, mmap_mode='r')
    shape = data.shape
    processed_data = np.zeros(shape, dtype=np.float32)
    
    total_chunks = (shape[0] + chunk_size - 1) // chunk_size
    for i in tqdm(range(0, shape[0], chunk_size), total=total_chunks):
        chunk = data[i:i+chunk_size].astype(np.float32)
        processed_chunk = operation(chunk)
        processed_data[i:i+chunk_size] = processed_chunk
    
    return processed_data


# Apply Lee filter to VV and VH data
print("Processing VV data...")
vv_filtered = chunked_processing(vv_path, chunk_size=10, operation=lee_filter)
print("Processing VH data...")
vh_filtered = chunked_processing(vh_path, chunk_size=10, operation=lee_filter)

print("Filtered VV shape:", vv_filtered.shape)
print("Filtered VH shape:", vh_filtered.shape)

# Save filtered data
np.save(os.path.join(base_dir, 'filtered_VV.npy'), vv_filtered)
np.save(os.path.join(base_dir, 'filtered_VH.npy'), vh_filtered)
print("Filtered data saved.")

# %% [markdown]
# S1 VV & VH Visualization (Raw & Filtered)

# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Load data
vv_band = np.load("C:\\Users\\Dell\\Desktop\\Project downlaods\\VV.npy")
vh_band = np.load("C:\\Users\\Dell\\Desktop\\Project downlaods\\VH.npy")
filtered_vv = np.load("C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VV.npy")
filtered_vh = np.load("C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VH.npy")

# Load timestamps
with open("C:\\Users\\Dell\\Desktop\\Project downlaods\\timestamp.pkl", 'rb') as f:
    timestamps = pickle.load(f)

# Create a custom colormap
colors = ['darkblue', 'blue', 'lightblue', 'white', 'lightgreen', 'green', 'darkgreen']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

def enhance_contrast(image):
    # Replace inf and -inf with nan
    image = np.nan_to_num(image, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    # Compute percentiles ignoring nan values
    p2, p98 = np.nanpercentile(image, (2, 98))
    
    # Clip and scale the image
    image_clipped = np.clip(image, p2, p98)
    image_normalized = (image_clipped - p2) / (p98 - p2)
    
    return image_normalized

def plot_band(ax, data, title):
    enhanced_data = enhance_contrast(data)
    im = ax.imshow(enhanced_data, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    return im

def plot_image_grid(vv_raw, vv_filtered, vh_raw, vh_filtered, timestamps, start_idx, end_idx):
    num_timestamps = end_idx - start_idx
    fig, axs = plt.subplots(num_timestamps, 4, figsize=(20, 5 * num_timestamps))
    fig.suptitle(f'Sentinel-1 SAR Data - Raw and Filtered VV/VH Bands (Timestamps {start_idx+1}-{end_idx})', fontsize=16)

    for i in range(num_timestamps):
        idx = start_idx + i
        plot_band(axs[i, 0], vv_raw[idx], f'VV Raw - {timestamps[idx].date()}')
        plot_band(axs[i, 1], vv_filtered[idx], f'VV Filtered - {timestamps[idx].date()}')
        plot_band(axs[i, 2], vh_raw[idx], f'VH Raw - {timestamps[idx].date()}')
        plot_band(axs[i, 3], vh_filtered[idx], f'VH Filtered - {timestamps[idx].date()}')

    # Add legend
    legend_elements = [
        Patch(facecolor='darkblue', edgecolor='none', label='Water/Very low backscatter'),
        Patch(facecolor='lightblue', edgecolor='none', label='Low backscatter'),
        Patch(facecolor='white', edgecolor='none', label='Moderate backscatter'),
        Patch(facecolor='green', edgecolor='none', label='High backscatter'),
        Patch(facecolor='darkgreen', edgecolor='none', label='Very high backscatter')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  
    plt.show()

# Plot all timestamps in batches of 10
total_timestamps = len(timestamps)
batch_size = 10

for start in range(0, total_timestamps, batch_size):
    end = min(start + batch_size, total_timestamps)
    plot_image_grid(vv_band, filtered_vv, vh_band, filtered_vh, timestamps, start, end)

# %% [markdown]
# Loads and inspects Sentinel-1 and Sentinel-2 datasets, including their data and timestamp information, shapes of the datasets, the range and count of timestamps, temporal alignment check to identify any common dates between the two datasets. 
# 

# %%
import numpy as np
import pickle
import os

# Define base directories
s1_base_dir = r'c:\Users\Dell\Desktop\Project downlaods'
s2_base_dir = r'c:\Users\Dell\Desktop\Sentinal 2'

# Sentinel-1 data
s1_vv_path = os.path.join(s1_base_dir, 'filtered_VV.npy')
s1_vh_path = os.path.join(s1_base_dir, 'filtered_VH.npy')
s1_timestamp_path = os.path.join(s1_base_dir, 'timestamp.pkl')

# Sentinel-2 data
s2_data_path = os.path.join(s2_base_dir, 'MASKED_NORMALIZED_BANDS.npy')
s2_timestamp_path = os.path.join(s2_base_dir, 'timestamp.pkl')

# Load Sentinel-1 data and timestamps
s1_vv = np.load(s1_vv_path)
s1_vh = np.load(s1_vh_path)
with open(s1_timestamp_path, 'rb') as f:
    s1_timestamps = pickle.load(f)

# Load Sentinel-2 data and timestamps
s2_data = np.load(s2_data_path)
with open(s2_timestamp_path, 'rb') as f:
    s2_timestamps = pickle.load(f)

# Print shapes and timestamp information
print("Sentinel-1 VV shape:", s1_vv.shape)
print("Sentinel-1 VH shape:", s1_vh.shape)
print("Number of Sentinel-1 timestamps:", len(s1_timestamps))
print("Sentinel-1 date range:", min(s1_timestamps), "to", max(s1_timestamps))

print("\nSentinel-2 data shape:", s2_data.shape)
print("Number of Sentinel-2 timestamps:", len(s2_timestamps))
print("Sentinel-2 date range:", min(s2_timestamps), "to", max(s2_timestamps))

# Print first few timestamps for both datasets
print("\nFirst 5 Sentinel-1 timestamps:", s1_timestamps[:5])
print("First 5 Sentinel-2 timestamps:", s2_timestamps[:5])

# Check for temporal alignment
s1_set = set(s1_timestamps)
s2_set = set(s2_timestamps)
common_dates = s1_set.intersection(s2_set)
print("\nNumber of common dates:", len(common_dates))
if common_dates:
    print("Common date range:", min(common_dates), "to", max(common_dates))
else:
    print("No common dates found.")

# Print information about the Sentinel-2 data
print("\nSentinel-2 data statistics:")
print("Min value:", np.min(s2_data))
print("Max value:", np.max(s2_data))
print("Mean value:", np.mean(s2_data))
print("Number of zero values:", np.sum(s2_data == 0))

# %% [markdown]
# Loads timestamp data for Sentinel-1 and Sentinel-2 from pickle files. It defines a function to find the closest matching timestamps between the two datasets by calculating the absolute differences. The `find_closest` function returns index pairs of the closest timestamps, allowing for alignment of data from both sensors. prints the aligned index pairs to facilitate temporal correlation between the datasets.
# 

# %%
import pickle
import numpy as np

# Load timestamp data
def load_timestamps(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Sentinel-1 Timestamps
s1_timestamp_file = r'c:\\Users\\Dell\\Desktop\\Project downlaods\\timestamp.pkl'
s1_timestamps = load_timestamps(s1_timestamp_file)

# Sentinel-2 Timestamps
s2_timestamp_file = r'c:\\Users\\Dell\\Desktop\\Sentinal 2\\timestamp.pkl'
s2_timestamps = load_timestamps(s2_timestamp_file)

# Function to find closest timestamps
def find_closest(s1_times, s2_times):
    index_pairs = []
    for idx1, time1 in enumerate(s1_times):
        time_differences = [abs(time1 - time2) for time2 in s2_times]
        closest_idx = time_differences.index(min(time_differences))
        index_pairs.append((idx1, closest_idx))
    return index_pairs

# Align timestamps
aligned_indices = find_closest(s1_timestamps, s2_timestamps)
print("Aligned index pairs:", aligned_indices)


# %% [markdown]
# loads timestamp data for Sentinel-1 and Sentinel-2 checks and prints the type and examples of the first few timestamps from both datasets. Additionally, it verifies whether the timestamps are timezone-aware by inspecting their timezone information (`tzinfo`). 

# %%
import numpy as np

# Load the timestamp data
s2_timestamps = np.load(r'c:\\Users\\Dell\\Desktop\\Sentinal 2\\timestamp.pkl', allow_pickle=True)
s1_timestamps = np.load(r'c:\\Users\\Dell\\Desktop\\Project downlaods\\timestamp.pkl', allow_pickle=True)

# Check the types of the first few timestamps
print("Sentinel-2 timestamps type and example:", type(s2_timestamps[0]), s2_timestamps[:5])
print("Sentinel-1 timestamps type and example:", type(s1_timestamps[0]), s1_timestamps[:5])

# Check if they are timezone-aware
for ts in s2_timestamps[:5]:
    print("Sentinel-2 timestamp:", ts, "Timezone aware:", ts.tzinfo is not None)

for ts in s1_timestamps[:5]:
    print("Sentinel-1 timestamp:", ts, "Timezone aware:", ts.tzinfo is not None)


# %% [markdown]
# This script performs interpolation of Sentinel-2 data to align with Sentinel-1 timestamps. It first converts timestamps to numeric values (seconds since 1970-01-01) and sets up an output directory for the processed data. The script then processes the Sentinel-2 data in chunks, using linear interpolation to match the Sentinel-1 timestamps, and saves the interpolated data into an HDF5 file with gzip compression. Each chunk is processed and stored sequentially.
# 

# %%
import numpy as np
from scipy.interpolate import interp1d
import datetime
import pytz
import os
import h5py

# Load timestamps
s2_timestamps = np.load(r'c:\Users\Dell\Desktop\Sentinal 2\timestamp.pkl', allow_pickle=True)
s1_timestamps = np.load(r'c:\Users\Dell\Desktop\Project downlaods\timestamp.pkl', allow_pickle=True)

# Convert timestamps to numeric
def convert_timestamps_to_numeric(timestamps):
    reference_time = datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)
    return np.array([(ts - reference_time).total_seconds() for ts in timestamps])

s2_numeric_timestamps = convert_timestamps_to_numeric(s2_timestamps)
s1_numeric_timestamps = convert_timestamps_to_numeric(s1_timestamps)

# Output directory
output_dir = r"c:\Users\Dell\Desktop\Sentinal 2\interpolated_data"
os.makedirs(output_dir, exist_ok=True)

# Memory-mapped input file
input_file = r'c:\Users\Dell\Desktop\Sentinal 2\MASKED_NORMALIZED_BANDS.npy'
s2_data = np.load(input_file, mmap_mode='r')

# Get shape information
num_timestamps, num_rows, num_cols, num_bands = s2_data.shape

# Interpolation function
def interpolate_chunk(chunk, s2_times, s1_times):
    chunk_float32 = chunk.astype(np.float32)
    f = interp1d(s2_times, chunk_float32, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(s1_times)

# Process data in chunks
def process_and_save_chunks(s2_data, s2_times, s1_times, chunk_size=100):
    total_chunks = (num_rows // chunk_size + 1) * (num_cols // chunk_size + 1)
    
    with h5py.File(os.path.join(output_dir, 'interpolated_data.h5'), 'w') as hf:
        for i, row_start in enumerate(range(0, num_rows, chunk_size)):
            row_end = min(row_start + chunk_size, num_rows)
            for j, col_start in enumerate(range(0, num_cols, chunk_size)):
                col_end = min(col_start + chunk_size, num_cols)
                
                chunk = s2_data[:, row_start:row_end, col_start:col_end, :]
                interpolated_chunk = interpolate_chunk(chunk, s2_times, s1_times)
                
                chunk_name = f'chunk_{i}_{j}'
                hf.create_dataset(chunk_name, data=interpolated_chunk, compression='gzip', compression_opts=9)
                
                chunk_number = i * (num_cols // chunk_size + 1) + j + 1
                print(f"Processed chunk {chunk_number}/{total_chunks}")

# Run the processing
process_and_save_chunks(s2_data, s2_numeric_timestamps, s1_numeric_timestamps, chunk_size=100)
print("All data processed and saved.")

# %% [markdown]
# Reads an HDF5 file, lists datasets, loads a specific chunk, and prints its shape, data type, and value range.

# %%
import h5py

with h5py.File("C:\\Users\\Dell\\Desktop\\Sentinal 2\\interpolated_data\\interpolated_data.h5", 'r') as f:
    # Get a list of all chunks
    chunk_names = list(f.keys())
    
    # Load a specific chunk
    chunk_data = f['chunk_0_0'][:]

    # Print info about the first chunk
    print(f"Shape of first chunk: {chunk_data.shape}")
    print(f"Data type: {chunk_data.dtype}")
    print(f"Min value: {chunk_data.min()}, Max value: {chunk_data.max()}")

# %% [markdown]
# The `interpolate_chunk` function converts the chunk to `float32`, interpolates it along the time axis using `interp1d`, and ensures the output is also in `float32` format.
# 

# %%
def interpolate_chunk(chunk, s2_times, s1_times):
    chunk_float32 = chunk.astype(np.float32)
    f = interp1d(s2_times, chunk_float32, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(s1_times).astype(np.float32)  # Ensure float32 output

# %% [markdown]
# Plots the values of the first pixel in the first band over time, and prints statistics such as mean, median, and standard deviation.
# 

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("C:\\Users\\Dell\\Desktop\\Sentinal 2\\interpolated_data\\interpolated_data.h5", 'r') as f:
    chunk = f['chunk_0_0'][:]
    
    # Plot values for the first pixel of the first band over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(chunk.shape[0]), chunk[:, 0, 0, 0])
    plt.title('Values for First Pixel of First Band Over Time')
    plt.xlabel('Interpolated Timestamp Index')
    plt.ylabel('Pixel Value')
    plt.show()

    # Print some statistics
    print(f"Mean: {np.mean(chunk)}")
    print(f"Median: {np.median(chunk)}")
    print(f"Standard Deviation: {np.std(chunk)}")

# %% [markdown]
# Retrieves the name and shape of the first dataset (chunk) to check its dimensions.
# 

# %%
import h5py

# Open the Sentinel-2 HDF5 file
with h5py.File("C:\\Users\\Dell\\Desktop\\Sentinal 2\\interpolated_data\\interpolated_data.h5", 'r') as in_file:
    # Get the name of the first dataset
    first_chunk_name = list(in_file.keys())[0]
    
    # Access the first chunk and print its shape
    first_chunk = in_file[first_chunk_name]
    print(f"S2 first chunk shape: {first_chunk.shape}")


# %% [markdown]
# The code combines Sentinel-1 (S1) and Sentinel-2 (S2) data by selecting relevant S2 bands and fusing them with S1 data in chunks. It processes data from HDF5 files, ensuring the final dataset includes 10 S2 bands and 1 S1 band. ensures that data is processed in `float32` format by converting the S2 data in `process_chunk` to `float32` before saving it.
# 
# 

# %%
import numpy as np
import h5py
from tqdm import tqdm

def process_chunk(chunk):
    # If the chunk is 4D (time, height, width, bands), select only the 10 relevant bands
    relevant_bands = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # Adjusted for 0-based indexing
    return chunk[:, :, :, relevant_bands].astype(np.float32)

# Load S1 data
s1_data = np.load("C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VH.npy", mmap_mode='r')


if len(s1_data.shape) == 3:
    s1_data = s1_data[:, :, :, np.newaxis]

# Open the input and output files
with h5py.File("C:\\Users\\Dell\\Desktop\\Sentinal 2\\interpolated_data\\interpolated_data.h5", 'r') as in_file, \
     h5py.File('C:\\Users\\Dell\\Desktop\\fused_data.h5', 'w') as out_file:
    
    # Get the shape of the first chunk to initialize the output dataset
    first_chunk_name = list(in_file.keys())[0]
    first_chunk = in_file[first_chunk_name]
    chunk_shape = first_chunk.shape
    
    # Calculate total shape (10 S2 bands + 1 S1 band)
    total_shape = (s1_data.shape[0], s1_data.shape[1], s1_data.shape[2], 11)
    
    # Create the output dataset
    out_dataset = out_file.create_dataset('fused_data', shape=total_shape, dtype=np.float32, 
                                          chunks=(1, 100, 100, 11), compression='gzip')
    
    # Calculate total number of chunks
    total_chunks = total_shape[0] * (total_shape[1] // 100) * (total_shape[2] // 100)
    
    # Process data in small chunks
    with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as pbar:
        for t in range(total_shape[0]):  # For each timestamp
            for i in range(0, total_shape[1], 100):  # For each row chunk
                for j in range(0, total_shape[2], 100):  # For each column chunk
                    # Get S1 data for this chunk
                    s1_chunk = s1_data[t, i:i+100, j:j+100, :]
                    
                    # Get corresponding S2 data
                    chunk_index = (i // 100) * (total_shape[2] // 100) + (j // 100)
                    chunk_name = list(in_file.keys())[chunk_index]
                    s2_chunk = process_chunk(in_file[chunk_name][t:t+1, :, :, :])
                    
                    # Combine S1 and S2 data
                    fused_chunk = np.concatenate([s1_chunk, s2_chunk[0]], axis=-1)
                    
                    # Write to output file
                    out_dataset[t, i:i+100, j:j+100, :] = fused_chunk
                    
                    # Update progress bar
                    pbar.update(1)

print("Fusion complete!")

# %% [markdown]
#  Prints dataset details, and analyzes a subset of the data from the first time step. It prints statistics for each band, checks for NaN and infinite values, and plots a histogram for the first band in the subset. 

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open the fused data file
with h5py.File('C:\\Users\\Dell\\Desktop\\fused_data.h5', 'r') as f:
    # Get the dataset
    dataset = f['fused_data']
    
    # Print basic information
    print(f"Dataset shape: {dataset.shape}")
    print(f"Dataset dtype: {dataset.dtype}")
    
    # Analyze a subset of the data (e.g., first time step, central 500x500 pixels)
    subset = dataset[0, 950:1450, 950:1450, :]
    
    # Print information about each band in the subset
    for i in range(subset.shape[-1]):
        band_data = subset[:,:,i]
        print(f"\nBand {i+1}:")
        print(f"  Min: {np.min(band_data)}")
        print(f"  Max: {np.max(band_data)}")
        print(f"  Mean: {np.mean(band_data)}")
        print(f"  Std Dev: {np.std(band_data)}")
    
    # Check for any NaN or infinite values in the subset
    print(f"\nNumber of NaN values in subset: {np.isnan(subset).sum()}")
    print(f"Number of infinite values in subset: {np.isinf(subset).sum()}")
    
    # Plot histogram of a sample band (e.g., first band) from the subset
    plt.figure(figsize=(10, 6))
    plt.hist(subset[:,:,0].flatten(), bins=100)
    plt.title("Histogram of Band 1 (Subset)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()

print("Analysis complete!")

# %% [markdown]
# Shape of Fused Data

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open the fused data file
with h5py.File('C:\\Users\\Dell\\Desktop\\fused_data.h5', 'r') as f:
    # Get the dataset
    dataset = f['fused_data']
    
    # Print basic information about the entire dataset
    print(f"Full dataset shape: {dataset.shape}")
    print(f"Dataset dtype: {dataset.dtype}")
    
    # Select a single day (e.g., the first day)
    single_day_data = dataset[0]
    
    # Print information about the single day data
    print(f"\nSingle day data shape: {single_day_data.shape}")
    print(f"Number of dimensions: {single_day_data.ndim}")
    
    if single_day_data.ndim == 3:
        height, width, bands = single_day_data.shape
        print(f"Height: {height}")
        print(f"Width: {width}")
        print(f"Number of bands: {bands}")
    elif single_day_data.ndim == 2:
        height, width = single_day_data.shape
        print(f"Height: {height}")
        print(f"Width: {width}")
        print("This appears to be a 2D image (single band)")
    
    # Basic statistics for the single day data
    print(f"\nMin value: {np.min(single_day_data)}")
    print(f"Max value: {np.max(single_day_data)}")
    print(f"Mean value: {np.mean(single_day_data)}")
    print(f"Standard deviation: {np.std(single_day_data)}")
    
    # Check for any NaN or infinite values
    print(f"\nNumber of NaN values: {np.isnan(single_day_data).sum()}")
    print(f"Number of infinite values: {np.isinf(single_day_data).sum()}")

print("Analysis complete!")

# %%
import h5py

# Open the fused data file
with h5py.File('C:\\Users\\Dell\\Desktop\\fused_data.h5', 'r') as f:
    # Get the dataset
    dataset = f['fused_data']
    
    # Extract data for a single day (e.g., the first day)
    single_day_data = dataset[0, :, :, :]
    
    # Print the shape of the single day's data
    print(f"Shape of single day's data: {single_day_data.shape}")
    
    # Print the height (spatial dimension)
    print(f"Height (number of rows) of single day's data: {single_day_data.shape[0]}")


# %%
# Reshape the single day's data from (2400, 2400, 11) to (5760000, 11)
vectorized_data = single_day_data.reshape(-1, single_day_data.shape[-1])

# Print the shape to confirm
print(f"Shape of vectorized data: {vectorized_data.shape}")


# %% [markdown]
# Reading Labels

# %%
import geopandas as gpd

# Read the GeoJSON file
train_labels_path = "C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson"
train_labels = gpd.read_file(train_labels_path)

# Display basic information
print(f"Number of features: {len(train_labels)}")
print(f"\nColumns in the dataset: {train_labels.columns.tolist()}")
print(f"\nGeometry type: {train_labels.geometry.type.value_counts()}")
print(f"\nCoordinate Reference System (CRS): {train_labels.crs}")

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(train_labels.head())

# Display some statistics if there are numerical columns
print("\nBasic statistics of numerical columns:")
print(train_labels.describe())

# If there's a 'crop_type' column, show its unique values and counts
if 'crop_type' in train_labels.columns:
    print("\nUnique crop types and their counts:")
    print(train_labels['crop_type'].value_counts())

# Display the bounding box of the entire dataset
print("\nBounding box of the dataset:")
print(train_labels.total_bounds)

# %% [markdown]
# Version 2

# %%
import numpy as np
import h5py

# Open the fused data file
with h5py.File('C:\\Users\\Dell\\Desktop\\fused_data.h5', 'r') as f:
    # Get the dataset
    dataset = f['fused_data']
    
    # Initialize list to hold day indices with clear data
    clear_day_indices = []

    # Loop over all days in the dataset
    for day in range(dataset.shape[0]):
        # Extract data for the current day
        day_data = dataset[day, :, :, :]
        
        # Compute mean brightness of Sentinel-2 bands (first 10 bands)
        mean_brightness = np.mean(day_data[:, :, :10], axis=-1)
        
        # Define a threshold for brightness (this is a basic approach)
        brightness_threshold = 0.2  # Adjust based on your data
        
        # Calculate the percentage of low-brightness pixels
        clear_pixel_percentage = np.mean(mean_brightness < brightness_threshold)
        
        # If a significant portion of the image is clear, keep this day
        if clear_pixel_percentage > 0.8:  # 80% of pixels are clear
            clear_day_indices.append(day)
    
    # Output the indices of the clear days
    print(f"Clear days identified: {clear_day_indices}")


# %%
import geopandas as gpd
from rasterio.features import rasterize
import rasterio
import numpy as np

def rasterize_labels(labels_gdf, template_raster_path):
    """
    Rasterize polygon labels to match the spatial resolution and extent of a template raster.

    Args:
    - labels_gdf (GeoDataFrame): The GeoDataFrame containing the labels.
    - template_raster_path (str): Path to the raster file whose spatial characteristics should be matched.

    Returns:
    - np.array: A numpy array of rasterized labels matching the template raster.
    """
    with rasterio.open(template_raster_path) as src:
        # Prepare an empty raster to match the input raster's shape
        out_shape = src.shape
        transform = src.transform
        
        # Rasterize the polygons
        raster_labels = rasterize(
            [(geom, crop_id) for geom, crop_id in zip(labels_gdf.geometry, labels_gdf['crop_id'])],
            out_shape=out_shape,
            transform=transform,
            fill=0,  # Background value, use 0 for no label
            all_touched=False,  # Optional: only label pixels whose centers are within the polygon
            dtype='int32'
        )
    
    return raster_labels


train_labels_raster = rasterize_labels(train_labels, 'C:\\Users\\Dell\\Desktop\\fused_data.h5')




# %%
import numpy as np
import h5py

# Open the fused data file
with h5py.File('C:\\Users\\Dell\\Desktop\\fused_data.h5', 'r') as f:
    # Get the dataset
    dataset = f['fused_data']
    
    # Initialize list to hold day indices with clear data
    clear_day_indices = []
    
    # Define batch size
    batch_size = 20
    total_days = dataset.shape[0]
    
    # Process data in batches
    for start_day in range(0, total_days, batch_size):
        end_day = min(start_day + batch_size, total_days)
        print(f"Processing days {start_day} to {end_day-1}")
        
        for day in range(start_day, end_day):
            # Extract data for the current day
            day_data = dataset[day, :, :, :]
            
            # Compute mean brightness of Sentinel-2 bands (first 10 bands)
            mean_brightness = np.mean(day_data[:, :, :10], axis=-1)
            
            # Define a threshold for brightness (this is a basic approach)
            brightness_threshold = 0.2  # Adjust based on your data
            
            # Calculate the percentage of low-brightness pixels
            clear_pixel_percentage = np.mean(mean_brightness < brightness_threshold)
            
            # If a significant portion of the image is clear, keep this day
            if clear_pixel_percentage > 0.8:  # 80% of pixels are clear
                clear_day_indices.append(day)
    
    # Output the indices of the clear days
    print(f"Clear days identified: {clear_day_indices}")


# %% [markdown]
# SOM Model....

# %% [markdown]
# Distribution of 11 bands of fused data

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_data_info(file_path):
    with h5py.File(file_path, 'r') as f:
        shape = f['fused_data'].shape
        dtype = f['fused_data'].dtype
    return shape, dtype

def process_chunk(chunk):
    # Calculate statistics for the chunk
    chunk_min = np.min(chunk, axis=(0, 1, 2))
    chunk_max = np.max(chunk, axis=(0, 1, 2))
    chunk_mean = np.mean(chunk, axis=(0, 1, 2))
    chunk_std = np.std(chunk, axis=(0, 1, 2))
    chunk_hist = [np.histogram(chunk[:,:,:,i].flatten(), bins=50)[0] for i in range(chunk.shape[3])]
    return chunk_min, chunk_max, chunk_mean, chunk_std, chunk_hist

def check_data_distribution(file_path, chunk_size=10):
    shape, dtype = load_data_info(file_path)
    
    # Initialize arrays to store results
    all_mins = []
    all_maxs = []
    all_means = []
    all_stds = []
    all_hists = [np.zeros(50) for _ in range(shape[3])]
    
    with h5py.File(file_path, 'r') as f:
        dataset = f['fused_data']
        for i in range(0, shape[0], chunk_size):
            chunk = dataset[i:i+chunk_size]
            chunk_min, chunk_max, chunk_mean, chunk_std, chunk_hist = process_chunk(chunk)
            all_mins.append(chunk_min)
            all_maxs.append(chunk_max)
            all_means.append(chunk_mean)
            all_stds.append(chunk_std)
            for j in range(shape[3]):
                all_hists[j] += chunk_hist[j]
    
    # Combine results
    min_values = np.min(all_mins, axis=0)
    max_values = np.max(all_maxs, axis=0)
    mean_values = np.mean(all_means, axis=0)
    std_values = np.mean(all_stds, axis=0)
    
    # Plot histograms
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for i in range(shape[3]):
        ax = axes[i // 4, i % 4]
        ax.bar(range(50), all_hists[i], width=1)
        ax.set_title(f"Band {i+1}")
    axes[2, 3].axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Min values: {min_values}")
    print(f"Max values: {max_values}")
    print(f"Mean values: {mean_values}")
    print(f"Std dev values: {std_values}")

# Load data and check distribution
file_path = 'C:\\Users\\Dell\\Desktop\\fused_data.h5'  # Adjust this path as necessary
check_data_distribution(file_path)

# %%
import numpy as np
from sklearn.preprocessing import RobustScaler

def preprocess_chunk(chunk):
    # Log transform Band 1
    chunk[:,:,:,0] = np.log1p(chunk[:,:,:,0])
    
    # Square root transform Bands 2-11
    chunk[:,:,:,1:] = np.sqrt(chunk[:,:,:,1:])
    
    # Initialize RobustScaler for each band
    scalers = [RobustScaler() for _ in range(chunk.shape[3])]
    
    # Reshape, scale, and reshape back
    for i in range(chunk.shape[3]):
        flat = chunk[:,:,:,i].reshape(-1, 1)
        chunk[:,:,:,i] = scalers[i].fit_transform(flat).reshape(chunk.shape[0:3])
    
    return chunk



# %% [markdown]
# Load and preprocess data from an HDF5 file, applying transformations and scaling to the fused bands data.
# 
# Process chunks of data to compute statistics and visualize histograms for each band, ensuring efficient handling of large datasets.

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def load_data_info(file_path):
    with h5py.File(file_path, 'r') as f:
        shape = f['fused_data'].shape
        dtype = f['fused_data'].dtype
    return shape, dtype

def preprocess_chunk(chunk):
    # Log transform Band 1
    chunk[:,:,:,0] = np.log1p(chunk[:,:,:,0])
    
    # Square root transform Bands 2-11
    chunk[:,:,:,1:] = np.sqrt(chunk[:,:,:,1:])
    
    # Initialize RobustScaler for each band
    scalers = [RobustScaler() for _ in range(chunk.shape[3])]
    
    # Reshape, scale, and reshape back
    for i in range(chunk.shape[3]):
        flat = chunk[:,:,:,i].reshape(-1, 1)
        chunk[:,:,:,i] = scalers[i].fit_transform(flat).reshape(chunk.shape[0:3])
    
    return chunk

def process_chunk(chunk):
    preprocessed_chunk = preprocess_chunk(chunk)
    
    # Calculate statistics for the preprocessed chunk
    chunk_min = np.min(preprocessed_chunk, axis=(0, 1, 2))
    chunk_max = np.max(preprocessed_chunk, axis=(0, 1, 2))
    chunk_mean = np.mean(preprocessed_chunk, axis=(0, 1, 2))
    chunk_std = np.std(preprocessed_chunk, axis=(0, 1, 2))
    chunk_hist = [np.histogram(preprocessed_chunk[:,:,:,i].flatten(), bins=50, range=(-5, 5))[0] for i in range(preprocessed_chunk.shape[3])]
    
    return chunk_min, chunk_max, chunk_mean, chunk_std, chunk_hist

def check_data_distribution(file_path, chunk_size=10):
    shape, dtype = load_data_info(file_path)
    
    # Initialize arrays to store results
    all_mins = []
    all_maxs = []
    all_means = []
    all_stds = []
    all_hists = [np.zeros(50) for _ in range(shape[3])]
    
    with h5py.File(file_path, 'r') as f:
        dataset = f['fused_data']
        for i in range(0, shape[0], chunk_size):
            chunk = dataset[i:i+chunk_size]
            chunk_min, chunk_max, chunk_mean, chunk_std, chunk_hist = process_chunk(chunk)
            all_mins.append(chunk_min)
            all_maxs.append(chunk_max)
            all_means.append(chunk_mean)
            all_stds.append(chunk_std)
            for j in range(shape[3]):
                all_hists[j] += chunk_hist[j]
    
    # Combine results
    min_values = np.min(all_mins, axis=0)
    max_values = np.max(all_maxs, axis=0)
    mean_values = np.mean(all_means, axis=0)
    std_values = np.mean(all_stds, axis=0)
    
    # Plot histograms
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for i in range(shape[3]):
        ax = axes[i // 4, i % 4]
        ax.bar(range(50), all_hists[i], width=1)
        ax.set_title(f"Band {i+1}")
        ax.set_xlim(0, 50)
    axes[2, 3].axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Min values: {min_values}")
    print(f"Max values: {max_values}")
    print(f"Mean values: {mean_values}")
    print(f"Std dev values: {std_values}")

# Load data and check distribution
file_path = 'C:\\Users\\Dell\\Desktop\\fused_data.h5'  
check_data_distribution(file_path)

# %%
import numpy as np
import h5py
from tqdm import tqdm

def update_mean_var(count, mean, var, new_count, new_mean, new_var):
    total_count = count + new_count
    delta = new_mean - mean
    m_a = var * count
    m_b = new_var * new_count
    M2 = m_a + m_b + delta**2 * count * new_count / total_count
    updated_mean = mean + delta * new_count / total_count
    updated_var = M2 / total_count
    return total_count, updated_mean, updated_var

def select_clear_day(data_file, row_chunk=100, col_chunk=100):
    with h5py.File(data_file, 'r') as f:
        dataset = f['fused_data']
        total_timestamps, total_rows, total_cols, total_bands = dataset.shape
        
        counts = np.zeros((total_timestamps, total_bands))
        means = np.zeros((total_timestamps, total_bands))
        vars = np.zeros((total_timestamps, total_bands))
        
        for row in tqdm(range(0, total_rows, row_chunk), desc="Processing rows"):
            for col in range(0, total_cols, col_chunk):
                end_row = min(row + row_chunk, total_rows)
                end_col = min(col + col_chunk, total_cols)
                
                chunk_data = dataset[:, row:end_row, col:end_col, :]
                chunk_count = (end_row - row) * (end_col - col)
                chunk_mean = np.mean(chunk_data, axis=(1, 2))
                chunk_var = np.var(chunk_data, axis=(1, 2))
                
                counts, means, vars = update_mean_var(
                    counts, means, vars, chunk_count, chunk_mean, chunk_var)
        
        stds = np.sqrt(vars)
        
        # Find the day with highest mean and lowest std
        score = np.mean(means, axis=1) / np.mean(stds, axis=1)
        best_day = np.argmax(score)
        
        return best_day

# Select the best day
best_day = select_clear_day('C:\\Users\\Dell\\Desktop\\fused_data.h5')
print(f"Best day for analysis: {best_day}")

# %% [markdown]
# Load and check properties of the HDF5 fused data, processing in chunks for memory efficiency. Calculate and print statistics such as min, max, mean, and standard deviation for each band.

# %%
import h5py
import numpy as np
from tqdm import tqdm

def check_data_properties(file_path, chunk_size=100):
    with h5py.File(file_path, 'r') as f:
        dataset = f['fused_data']
        
        print(f"Dataset shape: {dataset.shape}")
        print(f"Data type: {dataset.dtype}")
        
        # Initialize variables for statistics
        min_vals = np.full(dataset.shape[-1], np.inf)
        max_vals = np.full(dataset.shape[-1], -np.inf)
        sum_vals = np.zeros(dataset.shape[-1])
        sum_sq_vals = np.zeros(dataset.shape[-1])
        count = 0
        
        has_nan = False
        has_inf = False
        
        # Process data in chunks
        for i in tqdm(range(0, dataset.shape[1], chunk_size)):
            for j in range(0, dataset.shape[2], chunk_size):
                chunk = dataset[:, i:i+chunk_size, j:j+chunk_size, :]
                
                # Check for NaN and Inf
                if not has_nan:
                    has_nan = np.isnan(chunk).any()
                if not has_inf:
                    has_inf = np.isinf(chunk).any()
                
                # Update statistics
                min_vals = np.minimum(min_vals, np.min(chunk, axis=(0,1,2)))
                max_vals = np.maximum(max_vals, np.max(chunk, axis=(0,1,2)))
                sum_vals += np.sum(chunk, axis=(0,1,2))
                sum_sq_vals += np.sum(chunk**2, axis=(0,1,2))
                count += chunk.size // dataset.shape[-1]
        
        # Calculate final statistics
        mean_vals = sum_vals / count
        std_vals = np.sqrt((sum_sq_vals / count) - (mean_vals**2))
        
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")
        
        # Print summary statistics for each band
        for i in range(dataset.shape[-1]):
            print(f"\nBand {i+1} statistics:")
            print(f"  Min: {min_vals[i]}")
            print(f"  Max: {max_vals[i]}")
            print(f"  Mean: {mean_vals[i]}")
            print(f"  Std Dev: {std_vals[i]}")

# Check the properties of the dataset
input_file = 'C:\\Users\\Dell\\Desktop\\fused_data.h5'
check_data_properties(input_file)

# %%
def normalize_data(input_file, output_file, chunk_size=100):
    with h5py.File(input_file, 'r') as in_f, h5py.File(output_file, 'w') as out_f:
        in_dataset = in_f['fused_data']
        out_dataset = out_f.create_dataset('normalized_data', shape=in_dataset.shape, dtype='float32')
        
        # Normalize Band 1 separately (log transform)
        band1_max = np.log1p(35608.84765625)
        
        for i in tqdm(range(0, in_dataset.shape[1], chunk_size)):
            for j in range(0, in_dataset.shape[2], chunk_size):
                chunk = in_dataset[:, i:i+chunk_size, j:j+chunk_size, :]
                
                # Log transform and normalize Band 1
                chunk[:,:,:,0] = np.log1p(chunk[:,:,:,0]) / band1_max
                
                # Min-max normalization for Bands 2-11
                for b in range(1, 11):
                    min_val = -98.73687744140625  # Minimum value across all bands
                    max_val = 253.55316162109375  # Maximum value across all bands
                    chunk[:,:,:,b] = (chunk[:,:,:,b] - min_val) / (max_val - min_val)
                
                out_dataset[:, i:i+chunk_size, j:j+chunk_size, :] = chunk

normalize_data('C:\\Users\\Dell\\Desktop\\fused_data.h5', 'C:\\Users\\Dell\\Desktop\\normalized_data.h5')

# %%
import h5py
import numpy as np
from tqdm import tqdm

def check_normalized_data(file_path, chunk_size=100):
    with h5py.File(file_path, 'r') as f:
        dataset = f['normalized_data']
        
        print(f"Dataset shape: {dataset.shape}")
        print(f"Data type: {dataset.dtype}")
        
        min_vals = np.ones(dataset.shape[-1])
        max_vals = np.zeros(dataset.shape[-1])
        sum_vals = np.zeros(dataset.shape[-1])
        sum_sq_vals = np.zeros(dataset.shape[-1])
        count = 0
        
        for i in tqdm(range(0, dataset.shape[1], chunk_size)):
            for j in range(0, dataset.shape[2], chunk_size):
                chunk = dataset[:, i:i+chunk_size, j:j+chunk_size, :]
                
                min_vals = np.minimum(min_vals, np.min(chunk, axis=(0,1,2)))
                max_vals = np.maximum(max_vals, np.max(chunk, axis=(0,1,2)))
                sum_vals += np.sum(chunk, axis=(0,1,2))
                sum_sq_vals += np.sum(chunk**2, axis=(0,1,2))
                count += chunk.size // dataset.shape[-1]
        
        mean_vals = sum_vals / count
        std_vals = np.sqrt((sum_sq_vals / count) - (mean_vals**2))
        
        for i in range(dataset.shape[-1]):
            print(f"\nBand {i+1} statistics:")
            print(f"  Min: {min_vals[i]}")
            print(f"  Max: {max_vals[i]}")
            print(f"  Mean: {mean_vals[i]}")
            print(f"  Std Dev: {std_vals[i]}")

# Check the properties of the normalized dataset
normalized_file = 'C:\\Users\\Dell\\Desktop\\normalized_data.h5'
check_normalized_data(normalized_file)

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_normalized_data(file_path, output_prefix, time_steps=[0, 59, 119], chunk_size=100):
    with h5py.File(file_path, 'r') as f:
        dataset = f['normalized_data']
        
        for t in time_steps:
            rgb_data = np.zeros((dataset.shape[1], dataset.shape[2], 3))
            
            for i in tqdm(range(0, dataset.shape[1], chunk_size), desc=f"Processing time step {t}"):
                for j in range(0, dataset.shape[2], chunk_size):
                    chunk = dataset[t, i:i+chunk_size, j:j+chunk_size, :]
                    rgb_data[i:i+chunk_size, j:j+chunk_size, 0] = chunk[:,:,3]  # Red: Band 4
                    rgb_data[i:i+chunk_size, j:j+chunk_size, 1] = chunk[:,:,2]  # Green: Band 3
                    rgb_data[i:i+chunk_size, j:j+chunk_size, 2] = chunk[:,:,1]  # Blue: Band 2
            
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_data)
            plt.title(f'Normalized RGB Composite - Time Step {t}')
            plt.axis('off')
            plt.savefig(f'{output_prefix}_timestep_{t}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Visualize temporal profile for a sample pixel
        sample_pixel = dataset[:, 1200, 1200, :]
        plt.figure(figsize=(15, 10))
        for band in range(11):
            plt.plot(sample_pixel[:, band], label=f'Band {band+1}')
        plt.title('Temporal Profile of a Sample Pixel')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.savefig(f'{output_prefix}_temporal_profile.png', dpi=300, bbox_inches='tight')
        plt.close()

# Visualize the normalized data
normalized_file = 'C:\\Users\\Dell\\Desktop\\normalized_data.h5'
output_prefix = 'C:\\Users\\Dell\\Desktop\\normalized_visualization'
visualize_normalized_data(normalized_file, output_prefix)

# %% [markdown]
# Load and analyze normalized data from an HDF5 file in randomly selected chunks. Calculate and plot NDVI values along with temporal features like mean, max, and min NDVI for each chunk.**
# 
# Visualize NDVI time series for a central pixel and histogram distributions, while printing key statistics for each chunk analyzed.**

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

def analyze_data_chunks(input_file, chunk_size=100, num_chunks=5):
    with h5py.File(input_file, 'r') as f:
        dataset = f['normalized_data']
        time_steps, height, width, bands = dataset.shape
        
        # Randomly select chunks to analyze
        chunk_starts = [(np.random.randint(0, height - chunk_size),
                         np.random.randint(0, width - chunk_size))
                        for _ in range(num_chunks)]
        
        for chunk_num, (start_h, start_w) in enumerate(chunk_starts):
            chunk = dataset[:, start_h:start_h+chunk_size, start_w:start_w+chunk_size, :]
            
            # Calculate NDVI
            ndvi = calculate_ndvi(chunk[:,:,:,7], chunk[:,:,:,3])
            
            # Calculate temporal features
            mean_ndvi = np.mean(ndvi, axis=0)
            max_ndvi = np.max(ndvi, axis=0)
            min_ndvi = np.min(ndvi, axis=0)
            
            # Plot results for this chunk
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            
            # Plot mean NDVI
            im0 = axs[0, 0].imshow(mean_ndvi, cmap='RdYlGn')
            axs[0, 0].set_title(f'Mean NDVI - Chunk {chunk_num+1}')
            plt.colorbar(im0, ax=axs[0, 0])
            
            # Plot max NDVI
            im1 = axs[0, 1].imshow(max_ndvi, cmap='RdYlGn')
            axs[0, 1].set_title(f'Max NDVI - Chunk {chunk_num+1}')
            plt.colorbar(im1, ax=axs[0, 1])
            
            # Plot NDVI time series for center pixel
            center = chunk_size // 2
            axs[1, 0].plot(ndvi[:, center, center])
            axs[1, 0].set_title(f'NDVI Time Series - Center Pixel of Chunk {chunk_num+1}')
            axs[1, 0].set_xlabel('Time Step')
            axs[1, 0].set_ylabel('NDVI')
            
            # Plot histogram of mean NDVI
            axs[1, 1].hist(mean_ndvi.flatten(), bins=50)
            axs[1, 1].set_title(f'Histogram of Mean NDVI - Chunk {chunk_num+1}')
            axs[1, 1].set_xlabel('NDVI')
            axs[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nStatistics for Chunk {chunk_num+1}:")
            print(f"Mean NDVI: {np.mean(mean_ndvi):.4f}")
            print(f"Max NDVI: {np.max(max_ndvi):.4f}")
            print(f"Min NDVI: {np.min(min_ndvi):.4f}")
            print(f"NDVI Range: {np.max(max_ndvi) - np.min(min_ndvi):.4f}")
            print("\n" + "="*50 + "\n")

# Analyze the data
input_file = 'C:\\Users\\Dell\\Desktop\\normalized_data.h5'
analyze_data_chunks(input_file)

# %% [markdown]
# Load and visualize RGB composites for all time steps from the normalized dataset in an HDF5 file. Create a grid of images to display the RGB composites for each time step efficiently.**

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_all_timestamps(input_file, chunk_size=100):
    with h5py.File(input_file, 'r') as f:
        dataset = f['normalized_data']
        time_steps, height, width, bands = dataset.shape

        # Calculate number of rows needed
        rows = (time_steps + 3) // 4

        # Create a large figure
        fig, axs = plt.subplots(rows, 4, figsize=(20, 5*rows))
        fig.suptitle('RGB Composites for All Time Steps', fontsize=16)

        for t in tqdm(range(time_steps), desc="Processing time steps"):
            rgb_data = np.zeros((height, width, 3))
            
            for i in range(0, height, chunk_size):
                for j in range(0, width, chunk_size):
                    chunk = dataset[t, i:i+chunk_size, j:j+chunk_size, :]
                    rgb_data[i:i+chunk_size, j:j+chunk_size, 0] = chunk[:,:,3]  # Red: Band 4
                    rgb_data[i:i+chunk_size, j:j+chunk_size, 1] = chunk[:,:,2]  # Green: Band 3
                    rgb_data[i:i+chunk_size, j:j+chunk_size, 2] = chunk[:,:,1]  # Blue: Band 2

            rgb_data = np.clip(rgb_data, 0, 1)
            
            row = t // 4
            col = t % 4
            axs[row, col].imshow(rgb_data)
            axs[row, col].set_title(f'Time Step {t}')
            axs[row, col].axis('off')

        # Remove any unused subplots
        for t in range(time_steps, rows*4):
            row = t // 4
            col = t % 4
            fig.delaxes(axs[row, col])

        plt.tight_layout()
        plt.show()

# Visualize all time steps
input_file = 'C:\\Users\\Dell\\Desktop\\normalized_data.h5'
visualize_all_timestamps(input_file)

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_s1_enhanced(input_file, start_time=40, end_time=70, chunk_size=100):
    with h5py.File(input_file, 'r') as f:
        dataset = f['normalized_data']
        time_steps = end_time - start_time
        height, width = dataset.shape[1:3]

        rows = (time_steps + 3) // 4  # 4 images per row
        fig, axs = plt.subplots(rows, 4, figsize=(20, 5*rows))
        fig.suptitle('Sentinel-1 Enhanced Composites', fontsize=16)
        axs = axs.flatten()

        for t in tqdm(range(start_time, end_time), desc="Processing time steps"):
            s1_enhanced = np.zeros((height, width, 3))
            
            for i in range(0, height, chunk_size):
                for j in range(0, width, chunk_size):
                    chunk = dataset[t, i:i+chunk_size, j:j+chunk_size, :]
                    
                    # Sentinel-1 enhanced composite
                    s1_enhanced[i:i+chunk_size, j:j+chunk_size, 0] = chunk[:,:,0]  # Sentinel-1 band
                    s1_enhanced[i:i+chunk_size, j:j+chunk_size, 1] = chunk[:,:,3]  # Red band from Sentinel-2
                    s1_enhanced[i:i+chunk_size, j:j+chunk_size, 2] = chunk[:,:,1]  # Blue band from Sentinel-2

            s1_enhanced = np.clip(s1_enhanced, 0, 1)
            
            ax_index = t - start_time
            axs[ax_index].imshow(s1_enhanced)
            axs[ax_index].set_title(f'Time Step {t}')
            axs[ax_index].axis('off')

        # Remove any unused subplots
        for i in range(time_steps, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.show()

# Visualize Sentinel-1 enhanced composites
input_file = 'C:\\Users\\Dell\\Desktop\\normalized_data.h5'
visualize_s1_enhanced(input_file)

# %% [markdown]
# Load and extract data for a specific time step from the HDF5 dataset. Process the data in chunks to handle large dimensions efficiently.

# %%
import h5py
import numpy as np
from tqdm import tqdm

def extract_time_step(input_file, time_step, chunk_size=100):
    with h5py.File(input_file, 'r') as f:
        dataset = f['normalized_data']
        _, height, width, bands = dataset.shape
        
        extracted_data = np.zeros((height, width, bands), dtype=np.float32)
        
        for i in tqdm(range(0, height, chunk_size), desc="Extracting data"):
            for j in range(0, width, chunk_size):
                chunk = dataset[time_step, i:i+chunk_size, j:j+chunk_size, :]
                extracted_data[i:i+chunk_size, j:j+chunk_size, :] = chunk
        
    print(f"Extracted data shape: {extracted_data.shape}")
    return extracted_data

# Extract data for Time Step 49
input_file = 'C:\\Users\\Dell\\Desktop\\normalized_data.h5'
time_step_49_data = extract_time_step(input_file, 49)

# Visualize a false color composite of the extracted data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(time_step_49_data[:,:,[3,2,1]])  
plt.title("False Color Composite of Time Step 49")
plt.axis('off')
plt.show()

# %% [markdown]
# All 3 Visualiztion [Sentinal 1, Sentinal 2, Fused Data]

# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def visualize_imagery_enhanced(data, title, band=None):
    plt.figure(figsize=(15, 5))
    
    if band is not None:
        img = data[:,:,band]
    else:
        img = data
    
    # Create three subplots
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title(f"{title} - Original")
    plt.colorbar()
    
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    plt.subplot(132)
    plt.imshow(img_rescale, cmap='gray')
    plt.title(f"{title} - Contrast Stretched")
    plt.colorbar()
    
    # Histogram equalization
    img_eq = exposure.equalize_hist(img)
    plt.subplot(133)
    plt.imshow(img_eq, cmap='gray')
    plt.title(f"{title} - Histogram Equalized")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Data statistics for {title}:")
    print(f"Min: {np.min(img)}, Max: {np.max(img)}, Mean: {np.mean(img)}")


visualize_imagery_enhanced(time_step_49_data, "Panchromatic Image", band=0)

# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_three_image_types(multispectral_data, panchromatic_data, fused_data):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Satellite Imagery Comparison", fontsize=16)
    
    # Function to add colorbar
    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Multispectral image (using bands 3, 2, 1 for RGB)
    rgb_bands = [3, 2, 1]  # Adjust these if needed
    rgb_image = multispectral_data[:, :, rgb_bands]
    rgb_image = exposure.rescale_intensity(rgb_image, in_range='image', out_range=(0, 1))
    im1 = axs[0].imshow(rgb_image)
    axs[0].set_title("Multispectral (RGB)")
    add_colorbar(im1, axs[0])
    
    # Panchromatic image
    p2, p98 = np.percentile(panchromatic_data, (2, 98))
    pan_image = exposure.rescale_intensity(panchromatic_data, in_range=(p2, p98))
    im2 = axs[1].imshow(pan_image, cmap='gray')
    axs[1].set_title("Panchromatic")
    add_colorbar(im2, axs[1])
    
    # Fused image (all 13 bands)
  
    from sklearn.decomposition import PCA
    
    fused_reshaped = fused_data.reshape(-1, fused_data.shape[2])
    pca = PCA(n_components=3)
    fused_pca = pca.fit_transform(fused_reshaped)
    fused_pca = fused_pca.reshape(fused_data.shape[0], fused_data.shape[1], 3)
    fused_pca = exposure.rescale_intensity(fused_pca, in_range='image', out_range=(0, 1))
    
    im3 = axs[2].imshow(fused_pca)
    axs[2].set_title("Fused (PCA of 13 bands)")
    add_colorbar(im3, axs[2])
    
    # Remove ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("Multispectral data statistics:")
    print(f"Shape: {multispectral_data.shape}, Min: {np.min(multispectral_data)}, Max: {np.max(multispectral_data)}, Mean: {np.mean(multispectral_data)}")
    print("\nPanchromatic data statistics:")
    print(f"Shape: {panchromatic_data.shape}, Min: {np.min(panchromatic_data)}, Max: {np.max(panchromatic_data)}, Mean: {np.mean(panchromatic_data)}")
    print("\nFused data statistics:")
    print(f"Shape: {fused_data.shape}, Min: {np.min(fused_data)}, Max: {np.max(fused_data)}, Mean: {np.mean(fused_data)}")


multispectral_data = time_step_49_data[:, :, 1:11]  # 1-10 are multispectral
panchromatic_data = time_step_49_data[:, :, 0]  #  0 is panchromatic
fused_data = time_step_49_data  # All 13 bands

visualize_three_image_types(multispectral_data, panchromatic_data, fused_data)

# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_imagery_enhanced_row(data, title, band=None):
    if band is not None:
        img = data[:,:,band]
    else:
        img = data
    
    # Create figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16)
    
    # Original Image
    im1 = axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Original")
    
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    im2 = axs[1].imshow(img_rescale, cmap='gray')
    axs[1].set_title("Contrast Stretched")
    
    # Histogram equalization
    img_eq = exposure.equalize_hist(img)
    im3 = axs[2].imshow(img_eq, cmap='gray')
    axs[2].set_title("Histogram Equalized")
    
    # Remove ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Data statistics for {title}:")
    print(f"Min: {np.min(img)}, Max: {np.max(img)}, Mean: {np.mean(img)}")

# Assuming you have already extracted the data
visualize_imagery_enhanced_row(time_step_49_data, "Panchromatic Image", band=0)

# %% [markdown]
# Standardize the data and perform Principal Component Analysis (PCA) in chunks to reduce dimensionality efficiently. Reshape the PCA result and visualize the first three principal components.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(data, n_components=0.95, chunk_size=10000):
    height, width, bands = data.shape
    flattened_data = data.reshape(-1, bands)
    
    # Standardize the data
    scaler = StandardScaler()
    
    # Fit the scaler and perform PCA on chunks
    pca = PCA(n_components=n_components)
    
    for i in tqdm(range(0, flattened_data.shape[0], chunk_size), desc="Processing PCA"):
        chunk = flattened_data[i:i+chunk_size]
        if i == 0:
            scaled_chunk = scaler.fit_transform(chunk)
            pca.fit(scaled_chunk)
        else:
            scaled_chunk = scaler.transform(chunk)
        
        if i == 0:
            pca_result = pca.transform(scaled_chunk)
        else:
            pca_result = np.vstack((pca_result, pca.transform(scaled_chunk)))
    
    print(f"PCA result shape: {pca_result.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    return pca_result.reshape(height, width, -1), pca

# Perform PCA on the extracted data
pca_data, pca_model = perform_pca(time_step_49_data)

# Visualize the first three PCA components
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axs[i].imshow(pca_data[:,:,i], cmap='viridis')
    axs[i].set_title(f"PCA Component {i+1}")
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# Train a Self-Organizing Map (SOM) on PCA-transformed data in chunks, creating a 2D cluster map based on Best Matching Units (BMUs). Visualize the resulting cluster map and the SOM component planes.

# %% [markdown]
#  [Very Initial Stage of working with SOM]

# %%
from minisom import MiniSom
import numpy as np
from tqdm import tqdm

def train_som(data, som_shape=(20, 20), iterations=10000, chunk_size=10000):
    height, width, features = data.shape
    flattened_data = data.reshape(-1, features)
    
    # Initialize SOM
    som = MiniSom(som_shape[0], som_shape[1], features, sigma=1.0, learning_rate=0.5)
    
    # Train SOM in chunks
    for _ in tqdm(range(iterations), desc="Training SOM"):
        chunk_index = np.random.randint(0, flattened_data.shape[0], chunk_size)
        chunk = flattened_data[chunk_index]
        som.train_random(chunk, 1)  # Train on one sample at a time
    
    # Get BMU (Best Matching Unit) for each data point
    bmu_indices = np.array([som.winner(x) for x in tqdm(flattened_data, desc="Finding BMUs")])
    
    # Create a 2D grid of cluster labels
    cluster_map = np.array([som_shape[1] * x[0] + x[1] for x in bmu_indices]).reshape(height, width)
    
    return som, cluster_map

# Train SOM on PCA-transformed data
som_model, cluster_map = train_som(pca_data)

# Visualize the cluster map
plt.figure(figsize=(10, 10))
plt.imshow(cluster_map, cmap='tab20')
plt.title("SOM Cluster Map")
plt.colorbar(label="Cluster")
plt.axis('off')
plt.show()

# Visualize the SOM component planes
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
for i in range(min(6, pca_data.shape[2])):
    axs[i].imshow(som_model.get_weights()[:,:,i], cmap='viridis')
    axs[i].set_title(f"SOM Component {i+1}")
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# Load label data using GeoPandas and create a GeoTIFF raster file from the SOM cluster map. Define the appropriate raster transform and coordinate reference system (CRS) to align with the label data.
# 
# Save the cluster map as a GeoTIFF for further geospatial analysis and visualization, ensuring proper alignment and metadata.

# %%
import numpy as np
import geopandas as gpd
from rasterio.transform import from_origin
import rasterio
from shapely.geometry import box

# Load label data
train_labels = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")

# Create a raster transform 
transform = from_origin(408000, 5860000, 10, 10)  

# Create a raster dataset from the cluster map
with rasterio.open('cluster_map.tif', 'w', driver='GTiff', height=cluster_map.shape[0], 
                   width=cluster_map.shape[1], count=1, dtype=cluster_map.dtype, 
                   crs=train_labels.crs, transform=transform) as dst:
    dst.write(cluster_map, 1)

print("Cluster map saved as GeoTIFF.")

# %%
import rasterio
from rasterio.mask import mask

def get_cluster_values(geom, src):
    try:
        out_image, out_transform = mask(src, [geom], crop=True)
        return out_image[0]
    except ValueError:
        return np.array([])

with rasterio.open('cluster_map.tif') as src:
    train_labels['cluster_values'] = train_labels.geometry.apply(lambda geom: get_cluster_values(geom, src))

print("Cluster values extracted for each polygon.")

# %% [markdown]
# Create a final mapping that assigns each cluster to the most frequent crop type, allowing for interpretation of the SOM clusters based on the crop label data.

# %%
from collections import Counter
import numpy as np

def map_cluster_to_crop(cluster_values, crop_id):
    if len(cluster_values) == 0:
        return None
    # Flatten the array and convert to tuple
    flattened = tuple(cluster_values.flatten())
    return (crop_id, Counter(flattened).most_common(1)[0][0])

cluster_crop_mapping = {}

for _, row in train_labels.iterrows():
    mapping = map_cluster_to_crop(row['cluster_values'], row['crop_id'])
    if mapping:
        cluster_id = mapping[1]
        if cluster_id in cluster_crop_mapping:
            cluster_crop_mapping[cluster_id].append(mapping[0])
        else:
            cluster_crop_mapping[cluster_id] = [mapping[0]]


final_mapping = {cluster: Counter(crops).most_common(1)[0][0] 
                 for cluster, crops in cluster_crop_mapping.items()}

print("Cluster to crop type mapping complete.")
print(final_mapping)

# %% [markdown]
# Mapping Statistics

# %%
# Print statistics about the mapping
print("\nMapping Statistics:")
print(f"Total number of clusters mapped: {len(final_mapping)}")
print("Number of polygons mapped to each crop type:")
crop_counts = Counter([crop for crops in cluster_crop_mapping.values() for crop in crops])
for crop_id, count in crop_counts.items():
    crop_name = train_labels[train_labels['crop_id'] == crop_id]['crop_name'].iloc[0]
    print(f"Crop ID {crop_id} ({crop_name}): {count} polygons")

# %% [markdown]
# Populate the confusion matrix with true and predicted crop labels and display it for further analysis of model accuracy and errors.

# %%
from sklearn.metrics import confusion_matrix
import pandas as pd

def get_predicted_crop(cluster_values, mapping):
    if len(cluster_values) == 0:
        return None
    cluster = Counter(cluster_values.flatten()).most_common(1)[0][0]
    return mapping.get(cluster, None)

# Get true and predicted labels
y_true = []
y_pred = []

for _, row in train_labels.iterrows():
    true_crop = row['crop_id']
    pred_crop = get_predicted_crop(row['cluster_values'], final_mapping)
    if pred_crop is not None:
        y_true.append(true_crop)
        y_pred.append(pred_crop)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
crop_names = train_labels['crop_name'].unique()
cm_df = pd.DataFrame(cm, index=crop_names, columns=crop_names)

print("Confusion Matrix:")
print(cm_df)

# %% [markdown]
# Calculate overall accuracy metrics (accuracy, precision, recall, F1-score) and per-class metrics to evaluate the performance of SOM-based crop classification. Use weighted averages for overall scores and provide a detailed class-wise report.
# 
# Print out the results, including overall metrics and a DataFrame with precision, recall, and F1-score for each crop type to facilitate model evaluation and performance insights.

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\nAccuracy Metrics:")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1-score: {f1:.4f}")

# Calculate per-class metrics
class_report = pd.DataFrame({
    'Precision': precision_score(y_true, y_pred, average=None),
    'Recall': recall_score(y_true, y_pred, average=None),
    'F1-score': f1_score(y_true, y_pred, average=None)
}, index=crop_names)

print("\nPer-class Metrics:")
print(class_report)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Visualize per-class metrics
class_report.plot(kind='bar', figsize=(12, 6))
plt.title('Per-class Metrics')
plt.xlabel('Crop Type')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# Reshape PCA-transformed data and cluster map for SMOTE application to address class imbalance. Use SMOTE to generate synthetic samples for the minority classes in the dataset.
# 
# Print the dataset class distribution before and after resampling to confirm the effectiveness of SMOTE in balancing the dataset.

# %%
from imblearn.over_sampling import SMOTE
from collections import Counter

# Flatten the data for SMOTE
X = pca_data.reshape(pca_data.shape[0] * pca_data.shape[1], -1)
y = cluster_map.flatten()

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Original dataset shape:", Counter(y))
print("Resampled dataset shape:", Counter(y_resampled))

# %% [markdown]
# Apply PCA to the enhanced dataset to reduce its dimensionality while retaining 95% of the variance, and reshape the PCA result back to the original spatial dimensions. Finally, print the shapes of the enhanced and PCA-transformed data for verification.

# %%
import numpy as np
import h5py

# Load the original fused data
with h5py.File('C:\\Users\\Dell\\Desktop\\normalized_data.h5', 'r') as f:
    original_data = f['normalized_data'][49]  # Assuming we're using time step 49

print("Shape of original data:", original_data.shape)

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

def calculate_evi(nir, red, blue):
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

# Calculate indices

nir = original_data[:,:,8]
red = original_data[:,:,4]
blue = original_data[:,:,2]

ndvi = calculate_ndvi(nir, red)
evi = calculate_evi(nir, red, blue)

# Combine original data with new indices
enhanced_data = np.dstack((original_data, ndvi[:,:,np.newaxis], evi[:,:,np.newaxis]))

print("Shape of enhanced data:", enhanced_data.shape)

# If we still want to use PCA, we can apply it to the enhanced data
from sklearn.decomposition import PCA

# Reshape the data for PCA
reshaped_data = enhanced_data.reshape(-1, enhanced_data.shape[-1])

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
pca_result = pca.fit_transform(reshaped_data)

# Reshape back to original spatial dimensions
pca_data = pca_result.reshape(enhanced_data.shape[0], enhanced_data.shape[1], -1)

print("Shape of PCA-transformed data:", pca_data.shape)
print("Number of PCA components:", pca_data.shape[-1])

# %%
print("Shape of train_labels:", train_labels.shape)
print("Number of unique crop_id values:", train_labels['crop_id'].nunique())
print("Unique crop_id values:", train_labels['crop_id'].unique())

# %%
import geopandas as gpd

# Assuming 'train_labels' is a GeoDataFrame
print("CRS of vector data:", train_labels.crs)

# Open the raster and check its CRS
with rasterio.open('enhanced_data.tif') as src:
    print("CRS of raster data:", src.crs)

# If they differ, align the CRS of the vector data to that of the raster
if train_labels.crs != src.crs:
    train_labels = train_labels.to_crs(src.crs)


# %% [markdown]
# Visualize both the raster dataset and vector label data by overlaying the raster (with transparency) onto the polygons from the GeoDataFrame.

# %%
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import geopandas as gpd


fig, ax = plt.subplots(figsize=(10, 10))
train_labels.plot(ax=ax, color='red', edgecolor='k')  # Plot polygons

# Open your raster file
with rasterio.open('enhanced_data.tif') as src:
    show(src, ax=ax, alpha=0.5)  # Overlay raster

plt.show()


# %%
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, box

def extract_features_for_polygon(geom, raster_path, geom_crs):
    with rasterio.open(raster_path) as src:
        # The geometry and raster are in the same CRS
        if str(geom_crs) != str(src.crs):
            geom = geom.to_crs(src.crs)

        # Mask for the geometry if it overlaps the raster
        try:
            out_image, out_transform = mask(src, [geom], crop=True)
            return out_image.mean(axis=(1, 2))  # Average values for each band
        except ValueError:
            return None  # Return None if the geometry does not overlap

# Open the raster and write it (Ensure enhanced_data is correctly defined)
with rasterio.open('enhanced_data.tif', 'w', driver='GTiff', height=enhanced_data.shape[0], 
                   width=enhanced_data.shape[1], count=enhanced_data.shape[2], dtype=enhanced_data.dtype, 
                   crs=train_labels.crs, transform=transform) as dst:
    for i in range(enhanced_data.shape[2]):
        dst.write(enhanced_data[:, :, i], i+1)

# Extract features for each polygon
features = []
for geom in train_labels.geometry:
    feature = extract_features_for_polygon(geom, 'enhanced_data.tif', train_labels.crs)
    if feature is not None:
        features.append(feature)

features = np.array(features)  # Convert to numpy array, ignoring None values
labels = train_labels['crop_id'].values

print("Shape of features:", features.shape)
print("Shape of labels:", labels.shape)


# %%
print("Shape of features:", features.shape)  # Expected: (1749, 13)
print("Shape of labels:", labels.shape)      # Expected: (2064,)


# %% [markdown]
# Self-Organizing Map (SOM) using engineered features and labels. The features are first standardized, and the SOM is trained to map clusters to labels by finding the most common label in each cluster. The function then predicts labels, calculates accuracy, and generates a classification report and confusion matrix. 
# 
# 

# %%
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter

def train_and_evaluate_som(features, labels, som_shape=(10, 10), learning_rate=0.5, sigma=1.0):
    # Ensure labels match features in length
    labels = labels[:len(features)]
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Initialize and train SOM
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1], 
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    som.train_random(scaled_features, 10000)  # Train for 10000 iterations
    
    # Get best matching unit for each data point
    bmu_indices = np.array([som.winner(x) for x in scaled_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    # Map clusters to labels
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    # Assign the most common label to each cluster
    cluster_to_class = {}
    for cluster, cluster_labels in cluster_labels.items():
        cluster_to_class[cluster] = Counter(cluster_labels).most_common(1)[0][0]
    
    # Predict labels
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    # Calculate accuracy and generate report
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som

# Example usage assuming 'features' and 'labels' are already loaded
accuracy, report, cm, som = train_and_evaluate_som(features, labels)

print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(pd.DataFrame(cm))


# %%
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter

def train_and_evaluate_som(features, labels, som_shape=(30, 30), learning_rate=0.5, sigma=1.5, iterations=30000):
    labels = labels[:len(features)]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    som.train_random(scaled_features, iterations)
    
    bmu_indices = np.array([som.winner(x) for x in scaled_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(lbs).most_common(1)[0][0] for cluster, lbs in cluster_labels.items()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som

# Assuming features and labels are correctly loaded and prepared
accuracy, report, cm, som = train_and_evaluate_som(features, labels)
print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", pd.DataFrame(cm))


# %% [markdown]
# tweeked shape, learningrate and iterations

# %%
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter

def train_and_evaluate_som(features, labels, som_shape=(70, 70), learning_rate=0.5, sigma=1.5, iterations=30000):
    labels = labels[:len(features)]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    som.train_random(scaled_features, iterations)
    
    bmu_indices = np.array([som.winner(x) for x in scaled_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(lbs).most_common(1)[0][0] for cluster, lbs in cluster_labels.items()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som

# Assuming features and labels are correctly loaded and prepared
accuracy, report, cm, som = train_and_evaluate_som(features, labels)
print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", pd.DataFrame(cm))


# %%
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter
import json

def train_and_evaluate_som(features, labels, som_shape=(90, 90), learning_rate=0.5, sigma=1.5, iterations=30000):
    labels = labels[:len(features)]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    som.train_random(scaled_features, iterations)
    
    bmu_indices = np.array([som.winner(x) for x in scaled_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(lbs).most_common(1)[0][0] for cluster, lbs in cluster_labels.items()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels

# Assuming features and labels are correctly loaded and prepared
accuracy, report, cm, som, predicted_labels = train_and_evaluate_som(features, labels)

# Print results
print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", pd.DataFrame(report).transpose())
print("Confusion Matrix:\n", pd.DataFrame(cm))

# Save predictions
np.save('som_predictions.npy', predicted_labels)

# Save accuracy and report
results = {
    'accuracy': accuracy,
    'classification_report': report
}

with open('som_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nPredictions and results saved.")

# Check and print the distribution of predictions
prediction_distribution = Counter(predicted_labels)
print("\nPrediction Distribution:")
for label, count in prediction_distribution.items():
    print(f"Class {label}: {count/len(predicted_labels):.4f}")



# %% [markdown]
# Visualizing [27] 

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Load label file (which contains the geometries)
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictions.npy')

print(f"Number of geometries: {len(gdf)}")
print(f"Number of predictions: {len(predicted_labels)}")

# Create a new column for predictions, initially filled with a 'No Data' value
gdf['predicted_crop_id'] = -1  # or any value to represent 'No Data'

# Add predictions to GeoDataFrame, but only for the available predictions
gdf.loc[:len(predicted_labels)-1, 'predicted_crop_id'] = predicted_labels

# Define colors for each crop label, including a color for 'No Data'
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']
cmap = ListedColormap(colors)

crop_labels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']

# Create the plot
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax, legend=True,
         legend_kwds={'label': 'Predicted Crop Types', 'orientation': 'horizontal'})

# Add title and remove axes
plt.title('SOM Predicted Crop Types (Including No Data)')
plt.axis('off')

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(handles, crop_labels, title="Crop Types", loc="lower left", bbox_to_anchor=(0, -0.1), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('som_predicted_crops_map.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# Visualizing Ground Truth

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load GeoDataFrame
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")

# Define colors for each crop label
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
# Create a colormap
cmap = ListedColormap(colors)

# Assuming 'cropLabels' is a list of labels corresponding to 'crop_id' sorted numerically
cropLabels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

# Plot the GeoDataFrame with colors based on crop_id
ax = gdf.plot(column='crop_id', cmap=cmap, figsize=(10, 10), legend=True)

# Create custom legend handles
handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color, label=label) 
           for color, label in zip(colors, cropLabels)]
ax.legend(handles=handles, title='Crop Labels', loc='upper left')

# Save the plot as PNG
plt.savefig('output_image.png', dpi=300, bbox_inches='tight')


# %% [markdown]
# SOM on temporal window of fused data

# %%
import numpy as np
import h5py
from tqdm import tqdm

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

def calculate_evi(nir, red, blue):
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

def process_chunk(chunk):
    nir = chunk[:,:,8]
    red = chunk[:,:,4]
    blue = chunk[:,:,2]
    
    ndvi = calculate_ndvi(nir, red)
    evi = calculate_evi(nir, red, blue)
    
    return np.dstack((chunk, ndvi[:,:,np.newaxis], evi[:,:,np.newaxis]))

# Load and process the data for timestamps 40 to 80
chunk_size = 500  
with h5py.File('C:\\Users\\Dell\\Desktop\\normalized_data.h5', 'r') as f:
    dataset = f['normalized_data']
    shape = (41, dataset.shape[1], dataset.shape[2], dataset.shape[3] + 2)  # +2 for NDVI and EVI
    
    # Create a memory-mapped array for the enhanced data
    enhanced_data = np.memmap('enhanced_data.mmap', dtype='float32', mode='w+', shape=shape)
    
    for t in tqdm(range(40, 81), desc="Processing timestamps"):
        for i in range(0, shape[1], chunk_size):
            for j in range(0, shape[2], chunk_size):
                chunk = dataset[t, i:i+chunk_size, j:j+chunk_size, :]
                enhanced_chunk = process_chunk(chunk)
                enhanced_data[t-40, i:i+chunk_size, j:j+chunk_size, :] = enhanced_chunk

print("Shape of enhanced data:", enhanced_data.shape)



# %%
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from tqdm import tqdm
from affine import Affine

def extract_features_for_polygon(geom, enhanced_data, geom_crs, transform):
    # Create a mask for the geometry
    mask = geometry_mask([geom], out_shape=enhanced_data.shape[1:3], 
                         transform=transform, invert=True)
    
    # Apply the mask to each band and timestamp
    masked_data = enhanced_data[:, mask]
    
    # Calculate mean values for each band and timestamp
    return masked_data.mean(axis=1)



# Extract features for each polygon
features = []
for geom in tqdm(train_labels.geometry, desc="Extracting features"):
    feature = extract_features_for_polygon(geom, enhanced_data, train_labels.crs, transform)
    if feature is not None:
        features.append(feature)

features = np.array(features)
labels = train_labels['crop_id'].values

print("Shape of features:", features.shape)
print("Shape of labels:", labels.shape)

# %%
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter

def train_and_evaluate_som(features, labels, som_shape=(90, 90), learning_rate=0.5, sigma=1.5, iterations=30000):
    # Reshape features to 2D array (samples, features)
    features_2d = features.reshape(features.shape[0], -1)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_2d)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    
    # Initialize the SOM
    som.random_weights_init(scaled_features)
    
    # Train the SOM
    som.train_random(scaled_features, iterations, verbose=True)
    
    # Get the best matching unit for each input
    bmu_indices = np.array([som.winner(x) for x in scaled_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    # Map clusters to labels
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(lbs).most_common(1)[0][0] for cluster, lbs in cluster_labels.items()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels

# Train and evaluate SOM
accuracy, report, cm, som, predicted_labels = train_and_evaluate_som(features, labels)

# Print results
print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", pd.DataFrame(report).transpose())
print("Confusion Matrix:\n", pd.DataFrame(cm))

# Check and print the distribution of predictions
prediction_distribution = Counter(predicted_labels)
print("\nPrediction Distribution:")
for label, count in prediction_distribution.items():
    print(f"Class {label}: {count/len(predicted_labels):.4f}")

# %%
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter

def train_and_evaluate_som(features, labels, som_shape=(90, 90), learning_rate=1, sigma=1.5, iterations=40000):
    # Reshape features to 2D array (samples, features)
    features_2d = features.reshape(features.shape[0], -1)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_2d)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    
    # Initialize the SOM
    som.random_weights_init(scaled_features)
    
    # Train the SOM
    som.train_random(scaled_features, iterations, verbose=True)
    
    # Get the best matching unit for each input
    bmu_indices = np.array([som.winner(x) for x in scaled_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    # Map clusters to labels
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(lbs).most_common(1)[0][0] for cluster, lbs in cluster_labels.items()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels

# Train and evaluate SOM
accuracy, report, cm, som, predicted_labels = train_and_evaluate_som(features, labels)

# Print results
print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", pd.DataFrame(report).transpose())
print("Confusion Matrix:\n", pd.DataFrame(cm))

# Check and print the distribution of predictions
prediction_distribution = Counter(predicted_labels)
print("\nPrediction Distribution:")
for label, count in prediction_distribution.items():
    print(f"Class {label}: {count/len(predicted_labels):.4f}")

# %%
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter

def train_and_evaluate_som(features, labels, som_shape=(100, 100), learning_rate=0.5, sigma=10, iterations=40000):
    # Reshape features to 2D array (samples, features)
    features_2d = features.reshape(features.shape[0], -1)
    
    # Check for NaN or infinite values
    if np.isnan(features_2d).any() or np.isinf(features_2d).any():
        print("Warning: Data contains NaN or infinite values. Cleaning data...")
        features_2d = np.nan_to_num(features_2d, nan=0, posinf=0, neginf=0)
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_2d)
    
    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    pca_features = pca.fit_transform(scaled_features)
    print(f"Reduced features from {scaled_features.shape[1]} to {pca_features.shape[1]} dimensions")
    
    # Initialize and train SOM
    som = MiniSom(som_shape[0], som_shape[1], pca_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    
    som.random_weights_init(pca_features)
    som.train_random(pca_features, iterations, verbose=True)
    
    # Get best matching unit for each input
    bmu_indices = np.array([som.winner(x) for x in pca_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    # Map clusters to labels
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(lbs).most_common(1)[0][0] for cluster, lbs in cluster_labels.items()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels

# Train and evaluate SOM
accuracy, report, cm, som, predicted_labels = train_and_evaluate_som(features, labels)

# Print results
print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", pd.DataFrame(report).transpose())
print("Confusion Matrix:\n", pd.DataFrame(cm))

# Check and print the distribution of predictions
prediction_distribution = Counter(predicted_labels)
print("\nPrediction Distribution:")
for label, count in prediction_distribution.items():
    print(f"Class {label}: {count/len(predicted_labels):.4f}")

# %%
# Add these lines to your current session
np.save('som_predictions_multitemporal.npy', predicted_labels)
print("Predictions saved to 'som_predictions_multitemporal.npy'")

# %% [markdown]
# Visualizing [54] som temporal window result

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Load label file (which contains the geometries)
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictions_multitemporal.npy')

print(f"Number of geometries: {len(gdf)}")
print(f"Number of predictions: {len(predicted_labels)}")

# Create a new column for predictions
gdf['predicted_crop_id'] = predicted_labels

# Define colors for each crop label
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)

crop_labels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

# Create the plot
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax, legend=True,
         legend_kwds={'label': 'Predicted Crop Types', 'orientation': 'horizontal'})

# Add title and remove axes
plt.title('SOM Predicted Crop Types (Multi-temporal)')
plt.axis('off')

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(handles, crop_labels, title="Crop Types", loc="lower left", bbox_to_anchor=(0, -0.1), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('som_predicted_crops_map_multitemporal.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
import h5py

file_path = 'C:\\Users\\Dell\\Desktop\\normalized_data.h5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as file:
    print("Datasets and groups in HDF5 file:")
    file.visit(print)  # This will print all datasets and groups in the file


# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load masked data
masked_data = np.load('c:\\Users\\Dell\\Desktop\\Sentinal 2\\MASKED_NORMALIZED_BANDS.npy', mmap_mode='r')

def check_distribution(data, num_samples=1000):
    # Randomly sample pixels from the dataset
    total_pixels = data.shape[0] * data.shape[1] * data.shape[2]
    indices = np.random.choice(total_pixels, num_samples, replace=False)
    sampled_data = data.reshape(-1, data.shape[-1])[indices]

    # Plot histograms for each band
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()

    for i in range(data.shape[-1]):
        axes[i].hist(sampled_data[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Band {i+1}')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Print basic statistics
    print("Basic statistics for each band:")
    for i in range(data.shape[-1]):
        band_data = sampled_data[:, i]
        print(f"Band {i+1}:")
        print(f"  Min: {np.min(band_data):.2f}")
        print(f"  Max: {np.max(band_data):.2f}")
        print(f"  Mean: {np.mean(band_data):.2f}")
        print(f"  Std Dev: {np.std(band_data):.2f}")
        print()

# Check distribution
print("Checking data distribution...")
check_distribution(masked_data)

# %% [markdown]
# SOM for Sentinal 2 

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

print("Shape of masked_data:", masked_data.shape)

# Define chunk size (number of time steps per chunk)
chunk_size = 1  # Process 1 time step at a time
n_chunks = masked_data.shape[0]

print(f"Processing {n_chunks} time steps...")

# Open a file to save results incrementally
with open('pca_results_s2.npy', 'wb') as f:
    for i in tqdm(range(n_chunks), desc="Processing time steps"):
        chunk = masked_data[i]
        
        # Process chunk in smaller portions
        portion_size = 100000 
        n_portions = (chunk.shape[0] * chunk.shape[1] + portion_size - 1) // portion_size
        
        # Initialize PCA with a fixed number of components
        pca = PCA(n_components=5)  # Adjust this number as needed
        
        pca_results = []
        for j in range(n_portions):
            start = j * portion_size
            end = min((j + 1) * portion_size, chunk.shape[0] * chunk.shape[1])
            portion = chunk.reshape(-1, chunk.shape[-1])[start:end]
            
            # Apply StandardScaler
            scaler = StandardScaler()
            scaled_portion = scaler.fit_transform(portion)
            
            # Apply PCA
            pca_result = pca.fit_transform(scaled_portion)
            
            pca_results.append(pca_result)
        
        # Combine PCA results for this time step
        pca_result_combined = np.vstack(pca_results)
        pca_result_3d = pca_result_combined.reshape(chunk.shape[0], chunk.shape[1], -1)
        
        # Save this time step's results immediately
        np.save(f, pca_result_3d)
        
        print(f"Time step {i+1}/{n_chunks} processed and saved. Shape: {pca_result_3d.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

print("PCA completed and results saved incrementally.")

# %%
import numpy as np

# Load the PCA results
pca_results = np.load('pca_results_s2.npy', mmap_mode='r')

print("Shape of PCA results:", pca_results.shape)
print("Type of PCA results:", type(pca_results))


first_element = pca_results[0]
print("Shape of first element:", first_element.shape)
print("Type of first element:", type(first_element))

# Print the first few values
print("First few values:")
print(first_element[:5])

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# Print out some basic information about the masked_data
print("Shape of masked_data:", masked_data.shape)
print("Type of masked_data:", type(masked_data))
print("First few values of masked_data (raw data):")
print(masked_data[0, 0, :5]) 

# %%
import geopandas as gpd

# Path to your label file
label_file_path = "C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson"

# Load the GeoJSON file into a GeoDataFrame
train_labels_gdf = gpd.read_file(label_file_path)

# Check the loaded data
print("Loaded GeoDataFrame:")
print(train_labels_gdf.head())


# %%
# Load the GeoJSON file into a GeoDataFrame
label_file_path = "C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson"
labels_gdf = gpd.read_file(label_file_path)

# Print column names to identify the label column
print("Column names in GeoDataFrame:", labels_gdf.columns)


# %%
import pandas as pd

# Load the timestamp file
timestamps = pd.read_pickle("C:/Users/Dell/Desktop/Sentinal 2/timestamp.pkl")

# Print the timestamps
print(timestamps)


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load PCA-processed data

pca_data = np.load("C:\\Users\\Dell\\Desktop\\pca_results_s2.npy")

# Load timestamps
timestamps = pd.read_pickle("C:\\Users\\Dell\\Desktop\\Sentinal 2\\timestamp.pkl")

# Timestamps is a pandas DatetimeIndex to filter by month
timestamps = pd.DatetimeIndex(timestamps)


if pca_data.shape[0] == len(timestamps):
    # Filter data for May to July
    may_to_july_mask = (timestamps.month >= 5) & (timestamps.month <= 7)
    may_to_july_data = pca_data[may_to_july_mask, :, :]

    # Compute the composite image by averaging
    composite_image = np.mean(may_to_july_data, axis=0)  

    # Visualize the composite
    plt.figure(figsize=(10, 10))
    plt.imshow(composite_image, cmap='viridis') 
    plt.colorbar()
    plt.title('Pre-SOM Composite Image (May to July)')
    plt.show()
else:
    print("Dimension mismatch, please check the data dimensions and timestamp alignment.")


# %%
print(pca_data.shape)
print(len(timestamps))


# %%
print(masked_data.shape)
print(len(timestamps))

# %%
import numpy as np


# Load all PCA results
all_pca_results = np.load("C:\\Users\\Dell\\Desktop\\pca_results_s2.npy", allow_pickle=True)

# Check the shape to ensure all days are included
print(all_pca_results.shape)


composite = np.mean(all_pca_results, axis=0)  # Average over the days



# %%
import numpy as np

# Path to PCA file
file_path = "C:\\Users\\Dell\\Desktop\\pca_results_s2.npy"

# Create a memory map to the file
pca_data = np.memmap(file_path, dtype='float32', mode='r', shape=(144, 2400, 2400, 5))

# Example: Access PCA data for the first timestamp
first_timestamp_pca = pca_data[0]


# Process or visualize the PCA data

print(first_timestamp_pca.shape)


# %%
import numpy as np


file_path = "C:\\Users\\Dell\\Desktop\\pca_results_s2.npy"
pca_data = np.memmap(file_path, dtype='float32', mode='r', shape=(144, 2400, 2400, 5))

# Iterate over each timestamp and print a summary
for i in range(pca_data.shape[0]):
    current_timestamp_pca = pca_data[i]
    print(f"Timestamp {i+1}: shape {current_timestamp_pca.shape}")
    # Example of additional operation: print the mean of the PCA components
    print(f"Mean values for each component: {np.mean(current_timestamp_pca, axis=(0, 1))}")


# %%
import numpy as np

# Check each timestamp's PCA data for NaNs or Infs
for i in range(pca_data.shape[0]):
    current_timestamp_pca = pca_data[i]
    if np.isnan(current_timestamp_pca).any() or np.isinf(current_timestamp_pca).any():
        print(f"Timestamp {i+1} contains NaN or Inf values.")
    else:
        print(f"Timestamp {i+1} is clean.")


# %%
# Check the first few entries
print("First 5 features:", features[:5])
print("First 5 labels:", labels[:5])


# %%
import matplotlib.pyplot as plt

def debug_feature_extraction(pca_data, labels_gdf, meta):
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(pca_data)

            # For debugging, let's check a few masks
            for idx, row in labels_gdf.iterrows():
                if idx >= 5:  # Limit to checking the first 5 geometries
                    break

                geom = row['geometry']
                label = row['crop_id']
                geom_mask = geometry_mask([geom], out_shape=(dataset.height, dataset.width),
                                          transform=dataset.transform, invert=True)
                
                print(f"Geometry {idx} label {label}, mask any: {geom_mask.any()}")
                if geom_mask.any():
                    plt.figure(figsize=(10, 5))
                    plt.subplot(121)
                    plt.title(f'Geometry {idx} Mask')
                    plt.imshow(geom_mask, cmap='gray')
                    plt.subplot(122)
                    plt.title(f'PCA Data Band 1')
                    plt.imshow(dataset.read(1), cmap='viridis')
                    plt.show()

                else:
                    print(f"No data within the mask for geometry {idx}")


# %%
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load PCA results
pca_results = np.load('pca_results_s2.npy', mmap_mode='r')

# Reshape the data for SOM
som_data = pca_results.reshape(-1, 5)

# Initialize and train SOM
som_shape = (20, 20)  # You can adjust this
som = MiniSom(som_shape[0], som_shape[1], 5, sigma=1.0, learning_rate=0.5)

print("Training SOM...")
som.train_random(som_data, 10000)  # You can adjust the number of iterations

# Get the winner neurons for each data point
winner_coordinates = np.array([som.winner(x) for x in som_data]).T

# Reshape winner_coordinates back to the original image shape
som_result = winner_coordinates.reshape(2400, 2400, 2)

# Visualize SOM result
plt.figure(figsize=(10, 10))
plt.imshow(som_result[:,:,0], cmap='viridis')
plt.title('SOM Classification (X coordinate)')
plt.colorbar()
plt.show()

# You can also visualize the Y coordinate or combine them
plt.figure(figsize=(10, 10))
plt.imshow(som_result[:,:,1], cmap='viridis')
plt.title('SOM Classification (Y coordinate)')
plt.colorbar()
plt.show()

print("SOM analysis completed.")

# %%
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from collections import Counter

def train_and_evaluate_som(features, labels, som_shape=(90, 90), learning_rate=0.5, sigma=1.5, iterations=30000):
    labels = labels[:len(features)]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    som.train_random(scaled_features, iterations)
    
    bmu_indices = np.array([som.winner(x) for x in scaled_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(lbs).most_common(1)[0][0] for cluster, lbs in cluster_labels.items()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load PCA results

pca_results = np.load('pca_results_s2.npy')


# Option 2: Use a simple temporal summary (mean across all time steps)
pca_mean = np.mean(pca_results, axis=0).reshape(-1, 5)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(pca_mean)

print("Data is loaded and preprocessed. Shape:", scaled_features.shape)


# %%
from minisom import MiniSom
import numpy as np

# Load PCA results
pca_results = np.load('pca_results_s2.npy')


pca_mean = np.mean(pca_results, axis=0).reshape(-1, 5)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(pca_mean)

# Configure the SOM
som_shape = (50, 50)  
learning_rate = 0.5
sigma = 1.5
iterations = 5000  # May need adjustment based on convergence and computational capacity

# Initialize and train SOM
som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
              learning_rate=learning_rate, sigma=sigma, random_seed=42)
som.train_random(scaled_features, iterations)

print("SOM training completed.")


# %%
import matplotlib.pyplot as plt

# Function to plot the distance map
def plot_distance_map(som):
    plt.figure(figsize=(10, 10))
    plt.title('Distance Map')
    distance_map = som.distance_map()
    plt.imshow(distance_map, cmap='bone_r', interpolation='nearest')
    plt.colorbar()
    plt.show()

plot_distance_map(som)


som_weights = som.get_weights()
np.save('som_weights.npy', som_weights)


def plot_weight_planes(som):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(50, 10))
    for i, ax in enumerate(axes):
        ax.set_title(f'Component {i+1}')
        weight_plane = som.get_weights()[:,:,i]
        img = ax.imshow(weight_plane, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(img, ax=ax)
    plt.show()

plot_weight_planes(som)


# %%
# Mapping data points to their Best Matching Units (BMUs)
winner_coordinates = np.array([som.winner(x) for x in scaled_features]).T
# Organize winners by their coordinates
clusters = np.zeros(som_shape, dtype=int)
for x, y in zip(winner_coordinates[0], winner_coordinates[1]):
    clusters[x, y] += 1

plt.figure(figsize=(10, 10))
plt.title('Hits Map')
plt.imshow(clusters, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.show()


# %%
import numpy as np

# Load masked data 
file_path = r'C:\Users\Dell\Desktop\Sentinal 2\MASKED_NORMALIZED_BANDS.npy'
masked_data = np.load(file_path)

# Check the shape of your masked data
print(f"Shape of masked_data: {masked_data.shape}")

# Assign variables for dimensions
n_samples, n_rows, n_cols, n_features = masked_data.shape

# Define number of PCA components
n_components = 5

# Estimate PCA file size (assuming float32 data)
pca_data_size = (n_samples * n_rows * n_cols * n_components * 4) / (1024 ** 3)  # in GB
print(f"Estimated PCA file size: {pca_data_size:.2f} GB")


# %%
import numpy as np
import matplotlib.pyplot as plt

def load_and_visualize_sample(file_path, n_timestamps=3):
    try:
        # Memory-map the file
        data = np.load(file_path, mmap_mode='r')
        
        # Create a figure with subplots for each timestamp and PCA component
        fig, axs = plt.subplots(n_timestamps, 6, figsize=(20, 5*n_timestamps))
        
        for i in range(n_timestamps):
            for j in range(6):
                # Load only the required slice of data
                component = data[i, :, :, j]
                
                # Plot the component
                im = axs[i, j].imshow(component, cmap='viridis')
                axs[i, j].set_title(f"Timestamp {i+1}, Component {j+1}")
                plt.colorbar(im, ax=axs[i, j])
        
        plt.tight_layout()
        plt.savefig(r"C:\Users\Dell\Desktop\Sentinal 2\pcasamplevisualization.png")
        plt.close()
        
        print("Sample visualization saved successfully.")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")

# Path to PCA results file
pca_result_path = r"C:\\Users\\Dell\\Desktop\\Sentinal 2\\mayjunepcaresult.npy"

# Visualize 3 sample timestamps
load_and_visualize_sample(pca_result_path, n_timestamps=3)

print("Script execution completed.")

# %%
import numpy as np

try:
    data = np.load("C:\\Users\\Dell\\Desktop\\Sentinal 2\\mayjunepcaresult.npy")
    print("File loaded successfully:", data.shape)
except Exception as e:
    print("Failed to load file:", str(e))


# %%
import numpy as np

try:
    # Replace 'path_to_your_file.npy' with the actual path to your .npy file
    data = np.load("C:\\Users\\Dell\\Desktop\\Sentinal 2\\mayjunepcaresult.npy")
    print("File loaded successfully:", data.shape)
except Exception as e:
    print("Failed to load file:", str(e))


# %%
import numpy as np

try:
    data = np.load("C:\\Users\\Dell\\Desktop\\Sentinal 2\\mayjunepcaresult.npy")
    print("File loaded successfully", data.shape)
except Exception as e:
    print("Failed to load file:", str(e))


# %%
import numpy as np

def test_load_npy(file_path):
    try:
        data = np.load(file_path, mmap_mode='r')
        print("Shape of data:", data.shape)
        # Optionally, try accessing some part of the data
        print("Sample data:", data[0])
    except Exception as e:
        print("Error loading npy file:", str(e))

file_path = "C:\\Users\\Dell\\Desktop\\Sentinal 2\\mayjunepcaresult.npy"  # Replace with your file path
test_load_npy(file_path)


# %%
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# File paths
masked_data_path = r"C:\Users\Dell\Desktop\Sentinal 2\MASKED_NORMALIZED_BANDS.npy"
output_path = r"C:\Users\Dell\Desktop\Sentinal 2\may_june_pca_result.npy"

# Load data
print("Loading data...")
masked_data = np.load(masked_data_path, mmap_mode='r')

# Extract May-June data (indices 40 to 80)
may_june_data = masked_data[40:81]  # Check your indices are correct

n_timesteps, height, width, n_bands = may_june_data.shape
print(f"May-June data shape: {may_june_data.shape}")

# Initialize PCA
n_components = 6  # Adjust as needed
ipca = IncrementalPCA(n_components=n_components)

# Perform PCA
print("Performing PCA...")
chunk_size = 1000  # Adjust based on your available memory
n_samples = n_timesteps * height * width

for i in tqdm(range(0, n_samples, chunk_size)):
    chunk = may_june_data.reshape(-1, n_bands)[i:i+chunk_size]
    ipca.partial_fit(chunk)

# Transform and save results using np.save
print("Transforming data and saving results...")
transformed_data = np.zeros((n_timesteps, height, width, n_components), dtype=np.float32)

for i in tqdm(range(n_timesteps)):
    transformed = ipca.transform(may_june_data[i].reshape(-1, n_bands))
    transformed_data[i] = transformed.reshape(height, width, n_components)

np.save(output_path, transformed_data)
print(f"PCA results saved to {output_path}")

# Print explained variance ratio
print("\nExplained variance ratio:")
print(ipca.explained_variance_ratio_)
print(f"Total explained variance: {sum(ipca.explained_variance_ratio_):.4f}")

# Calculate and print file sizes
original_size = may_june_data.nbytes / (1024**3)
pca_size = transformed_data.nbytes / (1024**3)

print(f"\nOriginal May-June data size: {original_size:.2f} GB")
print(f"PCA results size: {pca_size:.2f} GB")
print(f"Size reduction: {(1 - pca_size/original_size)*100:.2f}%")


# %%
import numpy as np
import matplotlib.pyplot as plt

# Path to the PCA results file
pca_result_path = r"C:\Users\Dell\Desktop\Sentinal 2\may_june_pca_result.npy"

# Load the PCA results
print("Loading PCA results...")
pca_data = np.load(pca_result_path)

# Select the timestamp to visualize
timestamp_index = 0  # Adjust this to view different timestamps

# Create a figure to visualize the first 6 PCA components at the selected timestamp
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Visualization of PCA Components for Timestamp {timestamp_index + 1}')

for i in range(6):
    ax = axs[i // 3, i % 3]
    component = pca_data[timestamp_index, :, :, i]
    im = ax.imshow(component, cmap='viridis')
    ax.set_title(f'Component {i + 1}')
    plt.colorbar(im, ax=ax)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()


# %%
import numpy as np

def prepare_som_input(pca_file_path, output_file_path):
    # Load PCA results using memory mapping
    pca_data = np.load(pca_file_path, mmap_mode='r')
    
    # Get the shape of the data
    n_timestamps, height, width, n_components = pca_data.shape
    
    # Prepare the output file
    with open(output_file_path, 'wb') as f:
        # Write header
        np.array([height * width, n_timestamps * n_components]).astype(np.int32).tofile(f)
        
        # Process and write data in chunks
        chunk_size = 1000  # Adjust based on available memory
        for i in range(0, height * width, chunk_size):
            chunk = pca_data[:, i//width:(i+chunk_size)//width, i%width:(i+chunk_size)%width, :].reshape(-1, n_timestamps * n_components)
            chunk.astype(np.float32).tofile(f)
    
    print(f"SOM input prepared and saved to {output_file_path}")
    print(f"Shape of SOM input: ({height * width}, {n_timestamps * n_components})")

# File paths
pca_file_path = r"C:\Users\Dell\Desktop\Sentinal 2\may_june_pca_result.npy"
som_input_path = r"C:\Users\Dell\Desktop\Sentinal 2\som_input.dat"

# Prepare SOM input
prepare_som_input(pca_file_path, som_input_path)

# %%
import numpy as np

som_input_path = r"C:\Users\Dell\Desktop\Sentinal 2\som_input.dat"

try:
    with open(som_input_path, 'rb') as f:
        # Read the header (2 integers)
        header = np.fromfile(f, dtype=np.int32, count=2)
        print("Header:", header)
        
        # Try to read some data
        data = np.fromfile(f, dtype=np.float32, count=1000)
        print("Number of data elements read:", len(data))
        if len(data) > 0:
            print("First few data elements:", data[:10])
        else:
            print("No data found after header.")

except Exception as e:
    print(f"Error reading file: {str(e)}")

print("File check completed.")

# %%
import numpy as np
from minisom import MiniSom

# Load your PCA-reduced data
data = np.load("C:\\Users\\Dell\\Desktop\\Sentinal 2\\may_june_pca_result.npy")

# Check the initial shape
print("Initial data shape:", data.shape)

# Reshape data from (2400, 2400, 6) to (5760000, 6)
data = data.reshape(-1, data.shape[-1])  # Flatten spatial dimensions, keep PCA components
print("Reshaped data shape:", data.shape)

# Normalize data (important for training SOM)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# SOM initialization (adjust grid size as needed)
som = MiniSom(x=10, y=10, input_len=data.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 5000)  # train with 5000 iterations

# Accessing the SOM's weights
weights = som.get_weights()


# %%
import matplotlib.pyplot as plt

# Plotting the weight planes of the SOM
plt.figure(figsize=(14, 8))
for i in range(data.shape[1]):  # Number of components
    plt.subplot(2, 3, i + 1)
    plt.title(f'Component {i+1}')
    plt.pcolor(som.get_weights()[:, :, i], cmap='coolwarm')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np


label_map = np.random.randint(0, 5, size=(10, 10))  # Example data

plt.figure(figsize=(10, 10))
plt.pcolormesh(label_map, cmap='tab20')  # Using a colormap with enough colors
plt.colorbar()
plt.title("SOM Crop Classification Map")
plt.show()


# %%
import geopandas as gpd

# Load the GeoJSON file containing crop labels
label_path = r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson"
crop_labels = gpd.read_file(label_path)

# Check the data and its CRS
print("Data preview:")
print(crop_labels.head())
print("\nCRS of crop labels:", crop_labels.crs)


# %%

print("Original PCA data shape:", pca_data.shape)


if pca_data.shape[0] == 1:
    pca_data = np.squeeze(pca_data, axis=0)

print("Adjusted PCA data shape:", pca_data.shape)




# %%
import numpy as np
import rasterio
from rasterio.transform import from_origin


mean_pca_data = np.mean(pca_data, axis=0)  


transform = from_origin(408500, 5846500, 10, 10) 
metadata = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': None,
    'width': mean_pca_data.shape[1],
    'height': mean_pca_data.shape[0],
    'count': mean_pca_data.shape[2],  
    'crs': '+proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
    'transform': transform
}

# Output raster file path
raster_output = r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif"


with rasterio.open(raster_output, 'w', **metadata) as dst:
    for i in range(mean_pca_data.shape[2]):  # Loop through each PCA component
        
        dst.write(mean_pca_data[:, :, i], i + 1)


# %% [markdown]
# Crop Label Raster

# %%
import geopandas as gpd
import rasterio

# Load the crop label vector data
crop_labels = gpd.read_file(r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson")

# Load the raster
with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    print("Raster CRS:", src.crs)
    crop_labels = crop_labels.to_crs(src.crs)
    print("Vector CRS after transformation:", crop_labels.crs)


# %%
# Check raster bounds
with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    print("Raster bounds:", src.bounds)

# Check the bounds of the vector data
print("Vector bounds:", crop_labels.total_bounds)


# %%
import matplotlib.pyplot as plt
from rasterio.plot import show

with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax, title="Raster and Vector Overlay")
    crop_labels.plot(ax=ax, color='none', edgecolor='red', alpha=0.5)
    plt.show()


# %%
from shapely.geometry import box

# Define the bounding box of the raster
raster_bbox = box(minx=408500, miny=5822500, maxx=432500, maxy=5846500)

# Clip the vector data to the raster bounding box
crop_labels_clipped = crop_labels[crop_labels.geometry.intersects(raster_bbox)]

print("Vector data clipped. New bounds:", crop_labels_clipped.total_bounds)


# %%
features = []
with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    for geom in crop_labels_clipped.geometry:
        try:
            out_image, out_transform = mask(src, [geom], crop=True, all_touched=True)
            if out_image.size > 0:
                mean_values = np.mean(out_image, axis=(1, 2))
                features.append(mean_values)
            else:
                features.append(np.array([np.nan] * src.count))  
        except Exception as e:
            print("Error processing geometry:", e)
            features.append(np.array([np.nan] * src.count))

features = np.array(features)


# %%
fig, ax = plt.subplots(figsize=(10, 10))
with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    show(src, ax=ax, title="Raster and Clipped Vector Overlay")
    crop_labels_clipped.plot(ax=ax, color='none', edgecolor='blue', alpha=0.5)
    plt.show()


# %%
from rasterio.plot import show
import matplotlib.pyplot as plt

# Open the raster file
with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax, title="Raster with Vector Overlay")
    crop_labels.plot(ax=ax, marker='o', color='red', markersize=5)  
    plt.show()


# %%
from shapely.geometry import box

# Raster bounds
with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    raster_bounds = src.bounds

raster_box = box(*raster_bounds)

# Check each polygon
outside_geoms = []
for index, geom in enumerate(crop_labels.geometry):
    if not raster_box.intersects(geom):
        outside_geoms.append(index)

print("Indices of geometries outside raster bounds:", outside_geoms)


# %%
from shapely.geometry import box

# Define the bounding box of the raster
raster_bounds = box(408500, 5822500, 432500, 5846500)

# Clip the vector data to the raster bounding box
crop_labels_clipped = crop_labels[crop_labels.geometry.intersects(raster_bounds)]

# Re-check the bounds to confirm clipping
print("Re-checked Vector bounds after clipping:", crop_labels_clipped.total_bounds)


# %%
# Open the raster file
with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax, title="Raster with Clipped Vector Overlay")
    crop_labels_clipped.plot(ax=ax, marker='o', color='red', markersize=5)  # Plot as points for visibility
    plt.show()


# %%
features = []
with rasterio.open(r"C:\Users\Dell\Desktop\Sentinal 2\pca_output.tif") as src:
    for geom in crop_labels_clipped.geometry:
        try:
            out_image, out_transform = mask(src, [geom], crop=True, all_touched=True)
            if out_image.size > 0:  # Check if there is any data within the polygon
                mean_values = np.mean(out_image, axis=(1, 2))  # Mean across each component
                features.append(mean_values)
            else:
                features.append(np.array([np.nan]*src.count))  # Handle no data scenario
        except Exception as e:
            print("Error processing geometry:", e)
            features.append(np.array([np.nan]*src.count))

features = np.array(features)


# %% [markdown]
# SOM S2 Data

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


labels = crop_labels_clipped['crop_id']  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# %%
import numpy as np
from sklearn.preprocessing import StandardScaler

# Assuming 'features' and 'labels' are loaded as numpy arrays
print("Initial data check:")
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# Check for any missing values in the dataset
if np.isnan(features).any() or np.isnan(labels).any():
    print("Missing values found!")
    # Handling missing values; here we choose to fill them, but other methods like dropping them might be more appropriate depending on the case.
    features = np.nan_to_num(features)
    labels = np.nan_to_num(labels)

# Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print("Features scaled.")


# %%
from minisom import MiniSom

def train_som(scaled_features, som_shape=(90, 90), learning_rate=0.5, sigma=1.5, iterations=30000):
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    som.train_random(scaled_features, iterations)
    print("SOM training completed.")
    return som

# Train the SOM
som = train_som(scaled_features)


# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from minisom import MiniSom
from collections import Counter
import json

def train_and_evaluate_som(features, labels, som_shape=(90, 90), learning_rate=1.5, sigma=1.5, iterations=30000):
    labels = labels[:len(features)]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    som.train_random(scaled_features, iterations)
    
    bmu_indices = np.array([som.winner(x) for x in scaled_features])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(cluster_labels[cluster]).most_common(1)[0][0] for cluster in cluster_labels.keys()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels

# Assuming features and labels are correctly loaded and prepared
accuracy, report, cm, som, predicted_labels = train_and_evaluate_som(features, labels)

# Print results
print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", pd.DataFrame(report).transpose())
print("Confusion Matrix:\n", pd.DataFrame(cm))

# Save predictions and results
np.save('som_predictionss1.npy', predicted_labels)
results = {'accuracy': accuracy, 'classification_report': report}
with open('som_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nPredictions and results saved.")


# %%
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# Load PCA results for May to June
pca_data = np.load("C:\\Users\\Dell\\Desktop\\Sentinal 2\\may_june_pca_result.npy")

# Load the labels (adjust the path if needed)
labels_gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson")

# Assuming 'crop_name' field in your labels GeoDataFrame is used for labels
crop_labels = labels_gdf['crop_name'].values
label_geometry = labels_gdf['geometry']


# %%
# Load PCA-transformed features from .npy file
features = np.load(r"C:\Users\Dell\Desktop\Sentinal 2\may_june_pca_result.npy")

# Load labels appropriately
# For example, if you have a labels array saved as a numpy file:
labels = np.load(r"C:\Users\Dell\Desktop\path_to_your_labels.npy")

# Run the SOM training and evaluation
accuracy, report, cm, som, predicted_labels = train_and_evaluate_som(features, labels)

# Print results
print("Results:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", pd.DataFrame(report).transpose())
print("Confusion Matrix:\n", pd.DataFrame(cm))

# Save predictions and results
np.save('som_predictionss1.npy', predicted_labels)
results = {'accuracy': accuracy, 'classification_report': report}
with open('som_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nPredictions and results saved.")


# %%
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

# Load GeoDataFrame from GeoJSON
labels_gdf = gpd.read_file(r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson")

# Define the transformation and raster dimensions, ensure this matches raster data
transform = from_origin(408000, 5856000, 10, 10)  # Modify parameters to match  data
raster_shape = (2400, 2400)  

# Rasterize the labels
labels_raster = rasterize(
    ((geom, value) for geom, value in zip(labels_gdf.geometry, labels_gdf['crop_id'])),
    out_shape=raster_shape,
    transform=transform,
    fill=0,  # Assume 0 is the background/no-data value
    dtype='int32'
)

# Plotting the rasterized labels
plt.figure(figsize=(10, 10))
plt.imshow(labels_raster, cmap='tab20')  # Using 'tab20' colormap for better distinction of labels
plt.colorbar()
plt.title('Rasterized Crop Labels')
plt.show()


# %%
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt

# Load the GeoJSON file containing the crop labels
labels_gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson")

# Specify the spatial extent and resolution of your Sentinel-2 data
transform = from_origin(408000, 5856000, 10, 10)  # Modify these parameters to data
raster_shape = (2400, 2400)  # Dimensions should match those of the Sentinel-2 data

# Rasterize the labels, assuming 'crop_id' is the column with the label identifiers
rasterized_labels = rasterize(
    ((geom, value) for geom, value in zip(labels_gdf.geometry, labels_gdf['crop_id'])),
    out_shape=raster_shape,
    transform=transform,
    fill=0,  # Use '0' for areas not covered by any GeoJSON geometry
    dtype='int32'
)

# Save the rasterized labels to a NumPy file
np.save('rasterized_labels_s2.npy', rasterized_labels)

# Print the rasterized data to confirm
print("Rasterized Labels:", rasterized_labels)

# Optionally, visualize the rasterized labels
plt.imshow(rasterized_labels, cmap='viridis')
plt.colorbar()
plt.title('Rasterized Crop Labels for Sentinel-2')
plt.show()


# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Load your rasterized labels for Sentinel-2
rasterized_labels = np.load('rasterized_labels_s2.npy')  # Ensure the filename matches your saved rasterized labels for Sentinel-2

# Define a colormap with greater contrast
# Here we use distinct colors for each crop type or label category

# Define colors for each crop label, adding a color for 'No Data'
colors = ['#FFFFFF', '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)

# Define labels for your crops, including 'No Data'
crop_labels = ['No Data', 'Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

#cropLabels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data'] 

# Plotting
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # Increase DPI for higher resolution
im = ax.imshow(rasterized_labels, cmap=cmap)

ax.set_title('High-Resolution Rasterized Ground Truth Crop Labels for Sentinel-2')
ax.set_xticks([])
ax.set_yticks([])

# Create custom legend handles, ensuring they align with the colors and crop labels accurately
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=15) for color, label in zip(colors, crop_labels)]
legend = ax.legend(handles=handles, title='Crop Labels', loc='upper left', bbox_to_anchor=(1, 1))

# Save the plot with high resolution
plt.savefig('high_res_ground_truth_crops_map_s2.png', dpi=600)  # Save the figure with increased DPI for better quality
plt.show()


# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Load rasterized labels for Sentinel-2
rasterized_labels = np.load('rasterized_labels_s2.npy')  # Ensure the filename matches your saved rasterized labels for Sentinel-2

# Colormap with greater contrast
# Here we use distinct colors for each crop type or label category


colors = ['#FFFFFF','#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']  # Ensure you have enough colors for all labels
cmap = ListedColormap(colors)


cropLabels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']  # Make sure the labels match those used during rasterization



#cropLabels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data'] 

# Plotting
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # Increase DPI for higher resolution
im = ax.imshow(rasterized_labels, cmap=cmap)

ax.set_title('High-Resolution Rasterized Ground Truth Crop Labels for Sentinel-2')
ax.set_xticks([])
ax.set_yticks([])

# Create custom legend handles, ensuring they align with the colors and crop labels accurately
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=15) for color, label in zip(colors, crop_labels)]
legend = ax.legend(handles=handles, title='Crop Labels', loc='upper left', bbox_to_anchor=(1, 1))

# Save the plot with high resolution
plt.savefig('high_res_ground_truth_crops_map_s2.png', dpi=600)  # Save the figure with increased DPI for better quality
plt.show()


# %%
import numpy as np

# Load the rasterized labels from the saved .npy file
rasterized_labels = np.load('rasterized_labels_s2.npy')

# Find unique crop IDs and their counts in the rasterized data
unique_ids, counts = np.unique(rasterized_labels, return_counts=True)

# Print each crop ID and its count
print("Unique Crop IDs and their counts in the rasterized data:")
for crop_id, count in zip(unique_ids, counts):
    print(f"Crop ID {crop_id}: {count} pixels")


# %%
import geopandas as gpd

# Load the GeoJSON file containing the crop labels
geojson_path = "C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson"

crop_labels_gdf = gpd.read_file(geojson_path)

# Assuming 'crop_id' is the column with the label identifiers
if 'crop_id' in crop_labels_gdf.columns:
    unique_ids = crop_labels_gdf['crop_id'].unique()
    print("Unique Crop Label IDs:", unique_ids)
else:
    print("The 'crop_id' column does not exist in the GeoJSON data.")


# %%
import numpy as np

# Assuming rasterized_labels is your 2D array of labels
unique_labels, counts = np.unique(rasterized_labels, return_counts=True)
label_distribution = dict(zip(unique_labels, counts))
print("Distribution of Labels:", label_distribution)


# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Load label file (which contains the geometries)
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictionss1.npy')

print(f"Number of geometries: {len(gdf)}")
print(f"Number of predictions: {len(predicted_labels)}")

# Check if there are more geometries than predictions
if len(gdf) > len(predicted_labels):
    print("There are more geometries than predictions, adjusting...")
    gdf = gdf.iloc[:len(predicted_labels)]

# Create a new column for predictions
gdf['predicted_crop_id'] = predicted_labels

# Define colors for each crop label, you need to customize this part
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']
cmap = ListedColormap(colors)

# Define labels for your crops, customize this part based on your classes
crop_labels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']

# Create the plot
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax, legend=True,
         legend_kwds={'label': 'Predicted Crop Types', 'orientation': 'horizontal'})

# Add title and remove axes
plt.title('SOM Predicted Crop Types (Including No Data)')
plt.axis('off')

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(handles, crop_labels, title="Crop Types", loc="lower left", bbox_to_anchor=(0, -0.1), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('som_predicted_crops_map.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Load label file 
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictionss1.npy')

print(f"Number of geometries: {len(gdf)}")
print(f"Number of predictions: {len(predicted_labels)}")

# Check if there are more geometries than predictions
if len(gdf) > len(predicted_labels):
    print("There are more geometries than predictions, adjusting...")
    gdf = gdf.iloc[:len(predicted_labels)]
elif len(gdf) < len(predicted_labels):
    print("There are more predictions than geometries, truncating predictions...")
    predicted_labels = predicted_labels[:len(gdf)]

# Create a new column for predictions
gdf['predicted_crop_id'] = predicted_labels

# Print diagnostic information
print(f"Unique values in predicted_labels: {np.unique(predicted_labels)}")
print(f"Unique values in gdf['predicted_crop_id']: {gdf['predicted_crop_id'].unique()}")

# Define colors for each crop label
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']
cmap = ListedColormap(colors)

# Define labels for your crops
crop_labels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']

# Create the plot
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax, legend=True,
         legend_kwds={'label': 'Predicted Crop Types', 'orientation': 'horizontal'})

# Add title and remove axes
plt.title('SOM Predicted Crop Types (Including No Data)')
plt.axis('off')

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(handles, crop_labels, title="Crop Types", loc="lower left", bbox_to_anchor=(0, -0.1), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('som_predicted_crops_map.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Load label file (which contains the geometries)
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictionss1.npy')

print(f"Number of geometries: {len(gdf)}")
print(f"Number of predictions: {len(predicted_labels)}")

# Create a new column for predictions, initializing all to 'No Data' (value 0)
gdf['predicted_crop_id'] = 0

# Assign predictions to the first len(predicted_labels) geometries
gdf.loc[:len(predicted_labels)-1, 'predicted_crop_id'] = predicted_labels

# Print diagnostic information
print(f"Unique values in predicted_labels: {np.unique(predicted_labels)}")
print(f"Unique values in gdf['predicted_crop_id']: {gdf['predicted_crop_id'].unique()}")

# Define colors for each crop label, adding a color for 'No Data'
colors = ['#CCCCCC', '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)

# Define labels for your crops, including 'No Data'
crop_labels = ['No Data', 'Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

# Create the plot
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax, legend=True,
         legend_kwds={'label': 'Predicted Crop Types', 'orientation': 'horizontal'})

# Add title and remove axes
plt.title('SOM Predicted Crop Types (Including No Data)')
plt.axis('off')

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(handles, crop_labels, title="Crop Types", loc="lower left", bbox_to_anchor=(0, -0.1), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('som_predicted_crops_map_with_no_data.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load label file (which contains the geometries)
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictions.npy')

# Ensure the length of the GeoDataFrame matches the length of predictions
if len(predicted_labels) < len(gdf):
    gdf = gdf.iloc[:len(predicted_labels)]

# Assign predictions to the GeoDataFrame
gdf['predicted_crop_id'] = predicted_labels

# Define colors and crop labels as per your classification categories
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']  # Update as needed
cmap = ListedColormap(colors)

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
gdf.plot(column='predicted_crop_id', cmap=cmap, legend=True, ax=ax, cax=cax)

# Configure the colorbar
cbar = fig.colorbar(ax.collections[0], cax=cax)
cbar.set_label('Predicted Crop Types')

plt.title('Predicted Crop Types')
plt.axis('off')
plt.tight_layout()
plt.show()


# %%
import pandas as pd

# Load the timestamp data"C:\\Users\\Dell\\Desktop\\Project downlaods\\timestamp.pkl"
timestamps = pd.read_pickle()

print("Timestamps:", timestamps)


# %% [markdown]
# Sentinal 1 SOM

# %%
import pandas as pd
import pickle

# Path to pickle file
file_path = "C:\\Users\\Dell\\Desktop\\Project downlaods\\timestamp.pkl"

# Load the pickle file
with open(file_path, 'rb') as file:
    timestamps = pickle.load(file)

# Ensure timestamps are recognized as datetime objects
timestamps = pd.to_datetime(timestamps)

# Create a pandas Series from the datetime list
timestamps_series = pd.Series(timestamps)

# Filter for May to July
may_to_july = timestamps_series[(timestamps_series.dt.month >= 5) & (timestamps_series.dt.month <= 7)]

# Get indices of the filtered timestamps
indices_may_to_july = may_to_july.index.tolist()

# Print results
print("Filtered Timestamps:", may_to_july)
print("Indices for May to July:", indices_may_to_july)


# %%
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_and_preprocess(file_path, chunk_size=1000000):
    data = np.load(file_path, mmap_mode='r')
    total_size = data.size
    processed_data = []
    
    for i in range(0, total_size, chunk_size):
        chunk = data.flat[i:i+chunk_size]
        # Log transformation (adding a small constant to avoid log(0))
        chunk_log = np.log1p(chunk)
        processed_data.append(chunk_log)
    
    return np.concatenate(processed_data)

# Load and preprocess VV and VH data
print("Processing VV data...")
vv_processed = load_and_preprocess("C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VV.npy")
print("Processing VH data...")
vh_processed = load_and_preprocess("C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VH.npy")

# Calculate VH/VV ratio
print("Calculating VH/VV ratio...")
vh_vv_ratio = np.divide(vh_processed, vv_processed, out=np.zeros_like(vh_processed), where=vv_processed!=0)

# Combine features
print("Combining features...")
features = np.column_stack((vv_processed, vh_processed, vh_vv_ratio))

# Normalize features
print("Normalizing features...")
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

print("Preprocessing complete.")

# %%
print("Shape of normalized_features:", normalized_features.shape)
print("First few rows of normalized_features:")
print(normalized_features[:5])

# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump
from tqdm import tqdm

def load_data_in_chunks(file_path, chunk_size=1000000):
    """Generator to load data in chunks."""
    with np.load(file_path, mmap_mode='r') as data:
        for i in range(0, data.shape[0], chunk_size):
            yield data[i:i+chunk_size]

def train_random_forest(features, labels, n_estimators=100, max_depth=10):
    """Train a Random Forest classifier."""
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    rf.fit(features, labels)
    return rf

def predict_in_chunks(rf, features, chunk_size=1000000):
    """Make predictions in chunks to avoid memory issues."""
    predictions = []
    for chunk in tqdm(load_data_in_chunks(features, chunk_size), desc="Predicting"):
        chunk_predictions = rf.predict(chunk)
        predictions.extend(chunk_predictions)
    return np.array(predictions)

# Paths to your data
features_path = "path/to/your/normalized_features.npy"
labels_path = "path/to/your/labels.npy"  # If you have labels

# Load a sample of data for training (adjust sample_size as needed)
sample_size = 1000000  
features_sample = next(load_data_in_chunks(features_path, sample_size))
labels_sample = np.load(labels_path)[:sample_size]  # Assuming labels are available

# Split the sample into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_sample, labels_sample, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest...")
rf_model = train_random_forest(X_train_scaled, y_train)

print("Evaluating model on test set...")
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
dump(rf_model, 'random_forest_model.joblib')
dump(scaler, 'scaler.joblib')

print("Model and scaler saved. Now predicting on full dataset...")

# Predict on the full dataset
full_predictions = predict_in_chunks(rf_model, features_path)

# Save predictions
np.save('rf_predictions.npy', full_predictions)

print("Predictions saved. Process complete.")

# %% [markdown]
# SOM On Sentinal 1

# %%
import numpy as np
import geopandas as gpd
import rasterio

# Load VV and VH data
vv_data_path = r"C:\Users\Dell\Desktop\Project downlaods\filtered_VV.npy"
vh_data_path = r"C:\Users\Dell\Desktop\Project downlaods\filtered_VH.npy"

vv_data = np.load(vv_data_path)
vh_data = np.load(vh_data_path)

print("VV data shape:", vv_data.shape)
print("VH data shape:", vh_data.shape)
print("VV data type:", vv_data.dtype)
print("VH data type:", vh_data.dtype)

# Load label data
label_path = r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson"
labels = gpd.read_file(label_path)

print("\nLabel data info:")
print(labels.info())
print("\nLabel data CRS:", labels.crs)

# Check if VV and VH data have associated geospatial information
try:
    with rasterio.open(vv_data_path) as src:
        print("\nVV data CRS:", src.crs)
        print("VV data transform:", src.transform)
except rasterio.errors.RasterioIOError:
    print("\nVV data does not have associated geospatial information in the .npy file.")

try:
    with rasterio.open(vh_data_path) as src:
        print("\nVH data CRS:", src.crs)
        print("VH data transform:", src.transform)
except rasterio.errors.RasterioIOError:
    print("\nVH data does not have associated geospatial information in the .npy file.")

# %%
import pickle
import pprint

def safe_print(obj, depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}...")
        return
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{indent}{key}:")
            safe_print(value, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            safe_print(item, depth + 1, max_depth)
    else:
        print(f"{indent}{obj}")

metadata_path = r"C:\Users\Dell\Desktop\Project downlaods\meta_info.pkl"

try:
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print("Successfully loaded pickle file")
    print("\nMetadata structure:")
    safe_print(metadata)
    
    if isinstance(metadata, dict):
        print("\nTop-level keys:")
        for key in metadata.keys():
            print(f"- {key}")
        
        if 'bbox' in metadata:
            print("\nBounding Box information:")
            print(metadata['bbox'])
        
        if 'meta_info' in metadata:
            print("\nMeta info:")
            print(metadata['meta_info'])
        
        # Check for specific geospatial information
        keys_to_check = ['crs', 'transform', 'resolution', 'bounds', 'shape']
        print("\nChecking for specific geospatial information:")
        for key in keys_to_check:
            if key in metadata:
                print(f"{key}: {metadata[key]}")
            else:
                print(f"{key} not found in top-level metadata")

except Exception as e:
    print(f"Error loading pickle file: {e}")
    print("\nAttempting to read file as bytes:")
    try:
        with open(metadata_path, 'rb') as f:
            raw_data = f.read()
        print(f"File size: {len(raw_data)} bytes")
        print("First 100 bytes:")
        print(raw_data[:100])
    except Exception as inner_e:
        print(f"Error reading file: {inner_e}")

print("\nChecking file properties:")
import os
if os.path.exists(metadata_path):
    file_stats = os.stat(metadata_path)
    print(f"File size: {file_stats.st_size} bytes")
    print(f"Last modified: {os.path.getmtime(metadata_path)}")
else:
    print(f"File not found: {metadata_path}")

# %%
import numpy as np
import pickle
from pprint import pprint

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def inspect_npy(file_path):
    try:
        data = np.load(file_path, mmap_mode='r')
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Min value: {data.min()}")
        print(f"Max value: {data.max()}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Base path
base_path = r"C:\\Users\\Dell\\Desktop\\Project downlaods"

# Inspect NPY files
print("Inspecting filtered_VH.npy:")
inspect_npy(f"{base_path}\\filtered_VH.npy")

print("\nInspecting filtered_VV.npy:")
inspect_npy(f"{base_path}\\filtered_VV.npy")

# Load and inspect pickle files
bbox_data = load_pickle(f"{base_path}\\bbox.pkl")
print("\nContents of bbox.pkl:")
pprint(bbox_data)

meta_info = load_pickle(f"{base_path}\\meta_info.pkl")
print("\nContents of meta_info.pkl:")
pprint(meta_info)

timestamp_data = load_pickle(f"{base_path}\\timestamp.pkl")
print("\nContents of timestamp.pkl:")
pprint(timestamp_data)

# %%
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
vv_data = np.load(r"C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VV.npy")
vh_data = np.load(r"C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VH.npy")

# Reshape data
vv_reshaped = vv_data.reshape(120, -1).T
vh_reshaped = vh_data.reshape(120, -1).T

# Combine VV and VH data
combined_data = np.hstack((vv_reshaped, vh_reshaped))

# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(combined_data)

print("Shape of normalized data:", normalized_data.shape)
print("Min value:", normalized_data.min())
print("Max value:", normalized_data.max())

# %%
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt


input_len = normalized_data.shape[1]
som_shape = (10, 10)  

# Initialize and train SOM
som = MiniSom(som_shape[0], som_shape[1], input_len, sigma=1.0, learning_rate=0.5)
som.train_random(normalized_data, 10000)  

# Get SOM output
som_output = np.array([som.winner(x) for x in normalized_data])

# Visualize SOM
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()
plt.title('SOM U-matrix')
plt.show()

print("Shape of SOM output:", som_output.shape)

# %%
import geopandas as gpd
from sklearn.cluster import KMeans

# Load crop labels
labels_path = r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson"
crop_labels = gpd.read_file(labels_path)

# Perform K-means clustering on SOM output
n_clusters = len(crop_labels['crop_name'].unique())  # Use the number of unique crop types
kmeans = KMeans(n_clusters=n_clusters)
cluster_labels = kmeans.fit_predict(som_output)

# Reshape cluster labels to match the original image shape
cluster_map = cluster_labels.reshape(2400, 2400)

# Visualize cluster map
plt.figure(figsize=(10, 10))
plt.imshow(cluster_map, cmap='tab20')
plt.colorbar(label='Cluster')
plt.title('Crop Type Clusters')
plt.show()

print("Number of clusters:", n_clusters)
print("Unique cluster labels:", np.unique(cluster_labels))

# %% [markdown]
# Trail

# %%
import numpy as np
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



# Perform K-means clustering on SOM output
labels_path = r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson"
crop_labels = gpd.read_file(labels_path)
n_clusters = len(crop_labels['crop_name'].unique())
kmeans = KMeans(n_clusters=n_clusters)
cluster_labels = kmeans.fit_predict(som_output)

# Reshape cluster labels to match the original image shape
cluster_map = cluster_labels.reshape(2400, 2400)

# Create transform based on the bounding box we got earlier
bbox = ((408000.1106542662, 5831999.778170622), (432000.083625369, 5856000.355934586))
transform = from_bounds(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], 2400, 2400)

# Rasterize the crop labels
shapes = ((geom, value) for geom, value in zip(crop_labels.geometry, crop_labels.crop_id))
rasterized_labels = rasterize(shapes, out_shape=(2400, 2400), transform=transform, fill=0, dtype=np.int32)

# Compare cluster map with rasterized labels
valid_pixels = rasterized_labels != 0  # Exclude areas without label data
accuracy = np.sum((cluster_map == rasterized_labels) & valid_pixels) / np.sum(valid_pixels)

print(f"Overall accuracy: {accuracy:.2%}")

# Create a confusion matrix
cm = confusion_matrix(rasterized_labels[valid_pixels], cluster_map[valid_pixels])
print("Confusion Matrix:")
print(cm)

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

im1 = ax1.imshow(cluster_map, cmap='tab20')
ax1.set_title('Cluster Map')
plt.colorbar(im1, ax=ax1, label='Cluster')

im2 = ax2.imshow(rasterized_labels, cmap='tab20')
ax2.set_title('Rasterized Crop Labels')
plt.colorbar(im2, ax=ax2, label='Crop ID')

plt.tight_layout()
plt.show()

# Save the results for future use
np.save("cluster_map.npy", cluster_map)
np.save("rasterized_labels.npy", rasterized_labels)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from minisom import MiniSom
from collections import Counter
import json
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import pickle
from tqdm import tqdm

def train_and_evaluate_som(features, labels, som_shape=(30, 30), learning_rate=0.5, sigma=1.0, iterations=10000):
    labels = labels[:len(features)]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    
    print("Training SOM...")
    for i in tqdm(range(iterations)):
        som.train_random(scaled_features, num_iteration=1, verbose=False)
    
    print("Calculating BMUs...")
    bmu_indices = np.array([som.winner(x) for x in tqdm(scaled_features)])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(cluster_labels[cluster]).most_common(1)[0][0] for cluster in cluster_labels.keys()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels, cluster_map

print("Loading data...")
# Load Sentinel-1 data (VV and VH)
vv_data = np.load(r"C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VV.npy")
vh_data = np.load(r"C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VH.npy")

# Load timestamps
with open(r"C:\\Users\\Dell\\Desktop\\Project downlaods\\timestamp.pkl", 'rb') as f:
    timestamps = pickle.load(f)

print("Selecting May to July data...")
# Select May to July data
may_july_indices = [i for i, ts in enumerate(timestamps) if 5 <= ts.month <= 7]
vv_data_may_july = vv_data[may_july_indices]
vh_data_may_july = vh_data[may_july_indices]

# Reshape and combine VV and VH data
vv_reshaped = vv_data_may_july.reshape(len(may_july_indices), -1).T
vh_reshaped = vh_data_may_july.reshape(len(may_july_indices), -1).T
features = np.hstack((vv_reshaped, vh_reshaped))

print("Preparing labels...")
# Load and prepare labels
labels_path = r"C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson"
crop_labels = gpd.read_file(labels_path)
bbox = ((408000.1106542662, 5831999.778170622), (432000.083625369, 5856000.355934586))
transform = from_bounds(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], 2400, 2400)
shapes = ((geom, value) for geom, value in zip(crop_labels.geometry, crop_labels.crop_id))
rasterized_labels = rasterize(shapes, out_shape=(2400, 2400), transform=transform, fill=0, dtype=np.int32)
labels = rasterized_labels.flatten()

# Train and evaluate SOM
accuracy, report, cm, som, predicted_labels, cluster_map = train_and_evaluate_som(features, labels)

# Print results
print("\nResults:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", pd.DataFrame(report).transpose())
print("Confusion Matrix:\n", pd.DataFrame(cm))

print("Generating visualizations...")
# Visualize SOM U-matrix
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()
plt.title('SOM U-matrix (May-July)')
plt.show()

# Visualize Cluster Map and Rasterized Labels
cluster_map_2d = cluster_map.reshape(2400, 2400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

im1 = ax1.imshow(cluster_map_2d, cmap='tab20')
ax1.set_title('Cluster Map (May-July)')
plt.colorbar(im1, ax=ax1, label='Cluster')

im2 = ax2.imshow(rasterized_labels, cmap='tab20')
ax2.set_title('Rasterized Crop Labels')
plt.colorbar(im2, ax=ax2, label='Crop ID')

plt.tight_layout()
plt.show()

print("Saving results...")
# Save predictions and results
np.save('som_predictions_s1_may_july.npy', predicted_labels)
results = {'accuracy': accuracy, 'classification_report': report}
with open('som_results_s1_may_july.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nPredictions and results saved.")

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from scipy import stats

# Load label file (which contains the geometries)
gdf = gpd.read_file(r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictions_s1_may_july.npy')

print(f"Number of geometries: {len(gdf)}")
print(f"Shape of predictions: {predicted_labels.shape}")

# Reshape predictions to match the original image shape
predicted_labels_2d = predicted_labels.reshape(2400, 2400)

# Create a raster of geometries
bbox = ((408000.1106542662, 5831999.778170622), (432000.083625369, 5856000.355934586))
transform = from_bounds(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], 2400, 2400)
shapes = ((geom, value) for geom, value in zip(gdf.geometry, range(len(gdf))))
rasterized_geometries = rasterize(shapes, out_shape=(2400, 2400), transform=transform, fill=-1, dtype=np.int32)

# Assign predictions to geometries
predicted_raster = np.zeros_like(rasterized_geometries)
for i in range(len(gdf)):
    mask = rasterized_geometries == i
    if mask.sum() > 0:
        predicted_raster[mask] = stats.mode(predicted_labels_2d[mask], keepdims=False).mode

# Create a new GeoDataFrame with predictions
gdf['predicted_crop_id'] = [stats.mode(predicted_raster[rasterized_geometries == i], keepdims=False).mode if (rasterized_geometries == i).sum() > 0 else -1 for i in range(len(gdf))]

# Define colors for each crop label, you need to customize this part
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']
cmap = ListedColormap(colors)

# Define labels for your crops, customize this part based on your classes
crop_labels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']

# Create the plot
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax, legend=True,
         legend_kwds={'label': 'Predicted Crop Types', 'orientation': 'horizontal'})

# Add title and remove axes
plt.title('SOM Predicted Crop Types (Sentinel-1, May-July)')
plt.axis('off')

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(handles, crop_labels, title="Crop Types", loc="lower left", bbox_to_anchor=(0, -0.1), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('som_predicted_crops_map_s1_may_july.png', dpi=300, bbox_inches='tight')
plt.show()

print("Map has been generated and saved as 'som_predicted_crops_map_s1_may_july.png'")

# %%
def train_and_evaluate_som(features, labels, som_shape=(60, 60), learning_rate=0.5, sigma=2.0, iterations=20000):
    labels = labels[:len(features)]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Add VV/VH ratio and difference as additional features
    vv_vh_ratio = features[:, :len(features[0])//2] / (features[:, len(features[0])//2:] + 1e-10)
    vv_vh_diff = features[:, :len(features[0])//2] - features[:, len(features[0])//2:]
    scaled_features = np.hstack((scaled_features, scaler.fit_transform(vv_vh_ratio), scaler.fit_transform(vv_vh_diff)))
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    sample_weights = class_weights[labels]
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    
    print("Training SOM...")
    for i in tqdm(range(iterations)):
        idx = np.random.randint(len(scaled_features))
        som.update(scaled_features[idx], som.winner(scaled_features[idx]), i, iterations, weighted=True, weight=sample_weights[idx])
    
    print("Calculating BMUs...")
    bmu_indices = np.array([som.winner(x) for x in tqdm(scaled_features)])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    cluster_labels = {}
    for cluster, label, weight in zip(cluster_map, labels, sample_weights):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append((label, weight))
    
    cluster_to_class = {cluster: max(cluster_labels[cluster], key=lambda x: x[1])[0] for cluster in cluster_labels.keys()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels, cluster_map

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from minisom import MiniSom
from collections import Counter
import json
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import pickle
from tqdm import tqdm

def train_and_evaluate_som(features, labels, som_shape=(90, 90), learning_rate=0.5, sigma=1.5, iterations=30000):
    labels = labels[:len(features)]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    som = MiniSom(som_shape[0], som_shape[1], scaled_features.shape[1],
                  learning_rate=learning_rate, sigma=sigma, random_seed=42)
    
    print("Training SOM...")
    for i in tqdm(range(iterations)):
        som.train_random(scaled_features, num_iteration=1, verbose=False)
    
    print("Calculating BMUs...")
    bmu_indices = np.array([som.winner(x) for x in tqdm(scaled_features)])
    cluster_map = np.ravel_multi_index(bmu_indices.T, som_shape)
    
    cluster_labels = {}
    for cluster, label in zip(cluster_map, labels):
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        cluster_labels[cluster].append(label)
    
    cluster_to_class = {cluster: Counter(cluster_labels[cluster]).most_common(1)[0][0] for cluster in cluster_labels.keys()}
    predicted_labels = [cluster_to_class.get(cluster, -1) for cluster in cluster_map]
    
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(labels, predicted_labels)
    
    return accuracy, report, cm, som, predicted_labels, cluster_map

print("Loading data...")
# Load Sentinel-1 data (VV and VH)
vv_data = np.load(r"C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VV.npy")
vh_data = np.load(r"C:\\Users\\Dell\\Desktop\\Project downlaods\\filtered_VH.npy")

# Load timestamps
with open(r"C:\\Users\\Dell\\Desktop\\Project downlaods\\timestamp.pkl", 'rb') as f:
    timestamps = pickle.load(f)

print("Selecting May to July data...")
# Select May to July data
may_july_indices = [i for i, ts in enumerate(timestamps) if 5 <= ts.month <= 7]
vv_data_may_july = vv_data[may_july_indices]
vh_data_may_july = vh_data[may_july_indices]

# Reshape and combine VV and VH data
vv_reshaped = vv_data_may_july.reshape(len(may_july_indices), -1).T
vh_reshaped = vh_data_may_july.reshape(len(may_july_indices), -1).T
features = np.hstack((vv_reshaped, vh_reshaped))

print("Preparing labels...")
# Load and prepare labels
labels_path = r"C:\\Users\\Dell\\Desktop\\label\\br-17E-243N-crop-labels-test-2019.geojson"
crop_labels = gpd.read_file(labels_path)
bbox = ((408000.1106542662, 5831999.778170622), (432000.083625369, 5856000.355934586))
transform = from_bounds(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], 2400, 2400)
shapes = ((geom, value) for geom, value in zip(crop_labels.geometry, crop_labels.crop_id))
rasterized_labels = rasterize(shapes, out_shape=(2400, 2400), transform=transform, fill=0, dtype=np.int32)
labels = rasterized_labels.flatten()

# Train and evaluate SOM
accuracy, report, cm, som, predicted_labels, cluster_map = train_and_evaluate_som(features, labels)

# Print results
print("\nResults:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", pd.DataFrame(report).transpose())
print("Confusion Matrix:\n", pd.DataFrame(cm))

print("Generating visualizations...")
# Visualize SOM U-matrix
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()
plt.title('SOM U-matrix (May-July)')
plt.show()

# Visualize Cluster Map and Rasterized Labels
cluster_map_2d = cluster_map.reshape(2400, 2400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

im1 = ax1.imshow(cluster_map_2d, cmap='tab20')
ax1.set_title('Cluster Map (May-July)')
plt.colorbar(im1, ax=ax1, label='Cluster')

im2 = ax2.imshow(rasterized_labels, cmap='tab20')
ax2.set_title('Rasterized Crop Labels')
plt.colorbar(im2, ax=ax2, label='Crop ID')

plt.tight_layout()
plt.show()

print("Saving results...")
# Save predictions and results
np.save('som_predictions_s1_may_july.npy', predicted_labels)
results = {'accuracy': accuracy, 'classification_report': report}
with open('som_results_s1_may_july.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nPredictions and results saved.")

# %%
# Save the rasterized labels array to a numpy file for future use
np.save('rasterized_labels.npy', rasterized_labels)


# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from rasterio.transform import from_bounds
from rasterio.features import rasterize

# Load label file (which contains the geometries)
gdf = gpd.read_file(r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictions_s1_may_july.npy')

print(f"Number of geometries: {len(gdf)}")
print(f"Shape of predictions: {predicted_labels.shape}")

# Reshape predictions to match the original image shape
predicted_labels_2d = predicted_labels.reshape(2400, 2400)

# Create a raster of geometries
bbox = ((408000.1106542662, 5831999.778170622), (432000.083625369, 5856000.355934586))
transform = from_bounds(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], 2400, 2400)
shapes = ((geom, value) for geom, value in zip(gdf.geometry, range(len(gdf))))
rasterized_geometries = rasterize(shapes, out_shape=(2400, 2400), transform=transform, fill=-1, dtype=np.int32)

# Assign predictions to geometries
predicted_raster = np.zeros_like(rasterized_geometries)
for i in range(len(gdf)):
    mask = rasterized_geometries == i
    if mask.sum() > 0:
        predicted_raster[mask] = predicted_labels_2d[mask].flatten()[0]

# Create a new GeoDataFrame with predictions
gdf['predicted_crop_id'] = [predicted_raster[rasterized_geometries == i][0] if (rasterized_geometries == i).sum() > 0 else -1 for i in range(len(gdf))]

# Define colors for each crop label
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']
cmap = ListedColormap(colors)

# Define labels for your crops
crop_labels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']

# Create the plot
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax, legend=True,
         legend_kwds={'label': 'Predicted Crop Types', 'orientation': 'horizontal'})

# Add title and remove axes
plt.title('SOM Predicted Crop Types (Sentinel-1, May-July)')
plt.axis('off')

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(handles, crop_labels, title="Crop Types", loc="lower left", bbox_to_anchor=(0, -0.1), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('som_predicted_crops_map_s1_may_july_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("Map has been generated and saved as 'som_predicted_crops_map_s1_may_july_final.png'")

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from rasterio.transform import from_bounds
from rasterio.features import rasterize

# Load label file (which contains the geometries and true crop IDs)
gdf = gpd.read_file(r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson")

# Define colors and labels for crops 
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']
cmap = ListedColormap(colors)
crop_labels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']



# Create a raster of ground truth geometries with true crop IDs
bbox = ((408000.1106542662, 5831999.778170622), (432000.083625369, 5856000.355934586))
transform = from_bounds(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], 2400, 2400)
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['crop_id']))  # Assuming 'true_crop_id' is the field with the crop IDs
rasterized_labels = rasterize(shapes, out_shape=(2400, 2400), transform=transform, fill=-1, dtype=np.int32)

# Plotting the rasterized ground truth labels
fig, ax = plt.subplots(figsize=(15, 15))
im = ax.imshow(rasterized_labels, cmap=cmap)
ax.set_title('Rasterized Ground Truth Crop Labels')
colorbar = plt.colorbar(im, ax=ax, label='Crop ID')
colorbar.set_ticks(np.arange(len(crop_labels)))
colorbar.set_ticklabels(crop_labels)

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(handles, crop_labels, title="Crop Types", loc="lower left", bbox_to_anchor=(0, -0.1), ncol=5)

# Remove axes and adjust layout
plt.axis('off')
plt.tight_layout()
plt.savefig('rasterized_ground_truth_crops_map.png', dpi=300, bbox_inches='tight')
plt.show()

print("Rasterized ground truth map has been generated and saved as 'rasterized_ground_truth_crops_map.png'")


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Rasterized labels from a numpy file
rasterized_labels = np.load('rasterized_labels.npy')  # Ensure you replace 'path_to_rasterized_labels.npy' with the actual file path

# Define colors for each crop label
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']
cmap = ListedColormap(colors)


cropLabels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(rasterized_labels, cmap=cmap)
ax.set_title('Rasterized Ground Truth Crop Labels')




ax.set_xticks([])
ax.set_yticks([])

# Create custom legend handles
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=15) for color, label in zip(colors, cropLabels)]
legend = ax.legend(handles=handles, title='Crop Labels', loc='upper left', bbox_to_anchor=(1, 1))

# Save the plot as a PNG file
plt.savefig('ground_truth_crops_map.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Rasterized labels from a numpy file
rasterized_labels = np.load('rasterized_labels.npy')  # Ensure you replace 'path_to_rasterized_labels.npy' with the actual file path

# Define colors for each crop label, including a specific color for 'No Data'
colors = ['#0000',  # Color for 'No Data' which corresponds to '-1' in the array
          '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)


# Including 'No Data' at the start of the list to match the color for '-1' index
cropLabels = ['No Data', 'Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(rasterized_labels, cmap=cmap)
ax.set_title('Rasterized Ground Truth Crop Labels')




ax.set_xticks([])
ax.set_yticks([])

# Custom legend handles
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=15) for color, label in zip(colors, cropLabels)]
legend = ax.legend(handles=handles, title='Crop Labels', loc='upper left', bbox_to_anchor=(1, 1))

# Save the plot as a PNG file
plt.savefig('ground_truth_crops_map.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Data
rasterized_labels = np.load('rasterized_labels.npy')

# Colormap with greater contrast
colors = ['#0000',  # Color for 'No Data' which corresponds to '-1' in the array
          '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # Increase DPI for higher resolution
im = ax.imshow(rasterized_labels, cmap=cmap)

ax.set_title('High-Resolution Rasterized Ground Truth Crop Labels')
ax.set_xticks([])
ax.set_yticks([])

# Create custom legend handles
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=15) for color, label in zip(colors, cropLabels)]
legend = ax.legend(handles=handles, title='Crop Labels', loc='upper left', bbox_to_anchor=(1, 1))

# Save with high resolution
plt.savefig('high_res_ground_truth_crops_map.png', dpi=600)  # Increase DPI during saving as well
plt.show()


# %%
import numpy as np

# Rasterized labels from a numpy file
rasterized_labels = np.load('rasterized_labels.npy')  

# Some statistics about the rasterized labels
print("Rasterized Labels Stats:")
print("Shape of rasterized labels:", rasterized_labels.shape)
print("Unique crop IDs in rasterized labels:", np.unique(rasterized_labels))


print("Top-left corner of the rasterized labels:")
print(rasterized_labels[:10, :10])


# %%
import geopandas as gpd

# Load the GeoDataFrame from the GeoJSON file
gdf = gpd.read_file(r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson")

# Check the first few rows of the DataFrame to understand its structure
print("First few rows of the GeoDataFrame:")
print(gdf.head())


if 'crop_id' in gdf.columns:
    print("Unique Crop IDs in the dataset:")
    print(gdf['crop_id'].unique())
else:
    print("The column 'crop_id' does not exist in the dataset. Available columns are:")
    print(gdf.columns)


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


rasterized_labels = np.load('rasterized_labels.npy')  

# Colors for each crop label
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1', '#CCCCCC']  

# Assuming cropLabels match the indices used in your rasterized_labels
cropLabels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops', 'No Data']  

# Plot the GeoDataFrame with colors based on crop_id
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust size as needed
im = gdf.plot(column='crop_id', cmap=cmap, ax=ax, legend=True, legend_kwds={'label': "Crop Types", 'orientation': "horizontal"})

# Create custom legend handles
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=15) for color, label in zip(colors, cropLabels)]
ax.legend(handles=handles, title='Crop Labels', loc='upper left', bbox_to_anchor=(1, 1)) 


# Save the plot as PNG
plt.savefig('ground_truth_crops_map_temporal_window.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# All comparision visualization

# %% [markdown]
# Comparison: Predicted Sentinal 2 with Ground Truth

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# label file (which contains the geometries)
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")

# Saved predictions
predicted_labels = np.load('som_predictions_multitemporal.npy')

# Rasterized labels for Sentinel-2
rasterized_labels = np.load('rasterized_labels_s2.npy')  

# Number of geometries matches predictions
print(f"Number of geometries: {len(gdf)}")
print(f"Number of predictions: {len(predicted_labels)}")

# Column for predictions in the GeoDataFrame
gdf['predicted_crop_id'] = predicted_labels

# Colors for each crop label
colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)

s2_colors = ['#FFFFFF', '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080']
s2_cmap = ListedColormap(s2_colors)

# Define crop labels
crop_labels = ['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

# Plot with two subplots in one row
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot SOM predicted crops on the first axis (GeoDataFrame plot)
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax1)
ax1.set_title('SOM Predicted Crop Types (Multi-temporal)')
ax1.axis('off')

# Plot rasterized Sentinel-2 ground truth labels on the second axis
im = ax2.imshow(rasterized_labels, cmap=s2_cmap)
ax2.set_title('High-Resolution Ground Truth Crop Labels for Sentinel-2')
ax2.axis('off')

# Custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
fig.legend(handles, crop_labels, title="Crop Types", loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('combined_crop_maps_with_legend.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# Highlited

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def highlight_same_crop_types(image_path, color_to_highlight):
    # Load the image
    img = cv2.imread("C:\\Users\\Dell\\Desktop\\sen1\\new.png")
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lower_bound = np.array(color_to_highlight[0])
    upper_bound = np.array(color_to_highlight[1])
    
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img_rgb, contours, -1, (255, 0, 0), 2)  # Red contour lines
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title("Highlighted Crop Types")
    plt.axis('off')
    plt.show()

image_path = "C:\\Users\\Dell\\Desktop\\sen1\\new.png"

color_ranges = {
    'yellow': ([200, 180, 0], [255, 255, 100]),  # Yellow range in RGB
    
}

# Highlight one of the color ranges
highlight_same_crop_types(image_path, color_ranges['yellow'])


# %% [markdown]
# Comparison: Predicted Sentinal 1 with Ground Truth

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Load the label file which contains the geometries
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")

# Load the saved predictions
predicted_labels = np.load('som_predictions_multitemporal.npy')

# Load rasterized labels for the second map
rasterized_labels = np.load('rasterized_labels.npy')

print(f"Number of geometries: {len(gdf)}")
print(f"Number of predictions: {len(predicted_labels)}")

# Assign predictions to GeoDataFrame
gdf['predicted_crop_id'] = predicted_labels

colors = ['#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)

s1 = ['#0000',  # Color for 'No Data' which corresponds to '-1' in the array
          '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
s11_cmap = ListedColormap(s1)

# Define crop labels including 'No Data'
crop_labels = ['No Data', 'Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

# Create the plot with two subplots in one row
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot SOM predicted crops on the first axis
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax1)
ax1.set_title('SOM Predicted Crop Types (Multi-temporal)')
ax1.axis('off')

# Plot rasterized labels on the second axis
im = ax2.imshow(rasterized_labels, cmap=s11_cmap)
ax2.set_title('High-Resolution Rasterized Ground Truth Crop Labels')
ax2.set_xticks([])
ax2.set_yticks([])

# Create a common custom legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
fig.legend(handles, crop_labels, title="Crop Types", loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('combined_crops_maps_with_legend.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# Highlited

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def highlight_same_crop_types(image_path, color_to_highlight):
    # Load the image
    img = cv2.imread("C:\\Users\\Dell\\Desktop\\sen1\\part2.png")
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define the lower and upper bounds of the color you want to highlight
    # These should be in RGB format
    lower_bound = np.array(color_to_highlight[0])
    upper_bound = np.array(color_to_highlight[1])
    
    # Create a mask that isolates the areas of the image with the specified color
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    cv2.drawContours(img_rgb, contours, -1, (255, 0, 0), 2)  # Red contour lines
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title("Highlighted Crop Types")
    plt.axis('off')
    plt.show()

# Specify the path to your image
image_path = "C:\\Users\\Dell\\Desktop\\sen1\\part2.png"


color_ranges = {
    'red': ([165, 42, 42], [255, 255, 100]),  # Yellow range in RGB
    
}


highlight_same_crop_types(image_path, color_ranges['red'])


# %% [markdown]
# All 3 Sentinal 1, Sentinal 2, Fused Data Comparison

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")


predicted_labels = np.load('som_predictions_multitemporal.npy')
predicted_labels_no_data = np.load('som_predictionss1.npy')

# Load the saved predictions for the second map (Sentinel-1, May-July)
predicted_labels_s1 = np.load('som_predictions_s1_may_july.npy')
predicted_labels_s1_2d = predicted_labels_s1.reshape(2400, 2400)  # Example reshaping, adjust dimensions as necessary

# Define colors for each crop label, adding a color for 'No Data'
colors = ['#CCCCCC', '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)

s1 = ['#0000',  # Color for 'No Data' which corresponds to '-1' in the array
          '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
s11_cmap = ListedColormap(s1)

# Define labels for your crops, including 'No Data'
crop_labels = ['No Data', 'Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

# Assign predictions to GeoDataFrame
gdf['predicted_crop_id'] = predicted_labels
gdf['predicted_crop_id_s1'] = predicted_labels_s1_2d.flatten()[:len(gdf)]  # Assuming flat structure for simplicity
gdf['predicted_crop_id_no_data'] = 0
gdf.loc[:len(predicted_labels_no_data)-1, 'predicted_crop_id_no_data'] = predicted_labels_no_data

# Create the plot with three subplots in one row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

# Plot for Multi-temporal SOM predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax1)
ax1.set_title('SOM Predicted Crop Types (Multi-temporal)')
ax1.axis('off')

# Plot for Sentinel-1 predictions (May-July)
gdf.plot(column='predicted_crop_id_s1', cmap=s11_cmap, ax=ax2)
ax2.set_title('SOM Predicted Crop Types (Sentinel-1, May-July)')
ax2.axis('off')

# Plot for predictions including 'No Data'
gdf.plot(column='predicted_crop_id_no_data', cmap=cmap, ax=ax3)
ax3.set_title('SOM Predicted Crop Types (Including No Data)')
ax3.axis('off')

# Create a common custom legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
fig.legend(handles, crop_labels, title="Crop Types", loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=10)

# Adjust layout and save
plt.tight_layout()
plt.savefig('all_predicted_crops_maps_combined.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from rasterio.transform import from_bounds
from rasterio.features import rasterize

# Load the label file for all maps
gdf = gpd.read_file("C:\\Users\\Dell\\Desktop\\label\\br-18E-242N-crop-labels-train-2018.geojson")

# Load saved predictions for the first map
predicted_labels = np.load('som_predictions_multitemporal.npy')

# Load and process the second map's data
gdf2 = gpd.read_file(r"C:\Users\Dell\Desktop\label\br-17E-243N-crop-labels-test-2019.geojson")
predicted_labels_s1 = np.load('som_predictions_s1_may_july.npy')
predicted_labels_s1_2d = predicted_labels_s1.reshape(2400, 2400)  # Example reshaping, adjust dimensions as necessary

# Create a raster of geometries for the second map
bbox = ((408000.1106542662, 5831999.778170622), (432000.083625369, 5856000.355934586))
transform = from_bounds(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], 2400, 2400)
shapes = ((geom, value) for geom, value in zip(gdf2.geometry, range(len(gdf2))))
rasterized_geometries = rasterize(shapes, out_shape=(2400, 2400), transform=transform, fill=-1, dtype=np.int32)

predicted_raster = np.zeros_like(rasterized_geometries)
for i in range(len(gdf2)):
    mask = rasterized_geometries == i
    if mask.sum() > 0:
        predicted_raster[mask] = predicted_labels_s1_2d[mask].flatten()[0]
gdf2['predicted_crop_id_s1'] = [predicted_raster[rasterized_geometries == i][0] if (rasterized_geometries == i).sum() > 0 else -1 for i in range(len(gdf2))]

# Load saved predictions for the third map
predicted_labels_no_data = np.load('som_predictionss1.npy')

# Assign predictions to GeoDataFrame
gdf['predicted_crop_id'] = predicted_labels
gdf['predicted_crop_id_no_data'] = 0
gdf.loc[:len(predicted_labels_no_data)-1, 'predicted_crop_id_no_data'] = predicted_labels_no_data

# Define colors and crop labels, including 'No Data'
colors = ['#CCCCCC', '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
cmap = ListedColormap(colors)
crop_labels = ['No Data', 'Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil Seeds', 'Root Crops', 'Meadows', 'Forage Crops']

s1 = ['#0000',  # Color for 'No Data' which corresponds to '-1' in the array
          '#FFD700', '#A52A2A', '#ADD8E6', '#228B22', '#FFA500', '#8B0000', '#FFC0CB', '#800080', '#00CED1']
s11_cmap = ListedColormap(s1)

# Create the plot with three subplots in one row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

# Plot for Multi-temporal SOM predictions
gdf.plot(column='predicted_crop_id', cmap=cmap, ax=ax1)
ax1.set_title('SOM Predicted Crop Types (Multi-temporal)')
ax1.axis('off')

im = ax2.imshow(rasterized_labels, cmap=s11_cmap)
ax2.set_title('High-Resolution Rasterized Ground Truth Crop Labels')
ax2.set_xticks([])
ax2.set_yticks([])

# Plot for predictions including 'No Data'
gdf.plot(column='predicted_crop_id_no_data', cmap=cmap, ax=ax3)
ax3.set_title('SOM Predicted Crop Types (Including No Data)')
ax3.axis('off')

# Create a common custom legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
fig.legend(handles, crop_labels, title="Crop Types", loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=10)

# Adjust layout and save
plt.tight_layout()
plt.savefig('all_predicted_crops_maps_combined.png', dpi=300, bbox_inches='tight')
plt.show()



