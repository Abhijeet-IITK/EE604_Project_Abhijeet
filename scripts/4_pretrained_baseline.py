
# ============================================================
# PROJECT MODULE 4: PRETRAINED RCAN BASELINE
# ============================================================
#
# Description:
# This module evaluates a pretrained Residual Channel
# Attention Network (RCAN) for 3x satellite image
# super-resolution.
#
# Landsat-8 Low Resolution (LR) images are used as input,
# while Sentinel-2 High Resolution (HR) images are used
# as reference images for evaluation.
#
# Main functionalities:
# 1. Load the prepared testing dataset.
# 2. Load the pretrained RCAN model and checkpoint.
# 3. Convert satellite Digital Numbers (DN) into reflectance.
# 4. Prepare LR images as input tensors for RCAN.
# 5. Generate 3x super-resolution predictions.
# 6. Save predicted images as GeoTIFF files.
# 7. Evaluate predictions using PSNR, SSIM, RMSE,
#    MAE and Pearson correlation.
# 8. Generate CSV evaluation reports.
# 9. Visualize LR, HR and predicted HR images.
#
# Input:
# - Testing LR Landsat-8 images.
# - Testing HR Sentinel-2 images.
# - Pretrained RCAN checkpoint.
# - RCAN model implementation.
#
# Output:
# - Predicted super-resolution GeoTIFF images.
# - Individual image evaluation metrics.
# - Overall evaluation summary.
# - Visual comparison of satellite images.
#
# Super-resolution scale: 3x
#
# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import re
import sys
import time

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import rasterio
import matplotlib.pyplot as plt

from rasterio.transform import Affine

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from scipy.ndimage import binary_erosion


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

# Main project directory.
# Change this path according to your computer.

BASE_DIR = Path(
    r"C:\Users\ARF-G5\Desktop\Abhijeet"
)

# Testing dataset directories.
# LR contains Landsat-8 images.
# HR contains corresponding Sentinel-2 images.

TEST_LR_DIR = (
    BASE_DIR / "prepared_dataset" / "test" / "LR"
)

TEST_HR_DIR = (
    BASE_DIR / "prepared_dataset" / "test" / "HR"
)

# Directory containing the RCAN implementation.

RCAN_CODE_DIR = (
    BASE_DIR / "RCAN" / "RCAN_TestCode" / "code"
)

# Pretrained RCAN model checkpoint.

CHECKPOINT_PATH = (
    BASE_DIR / "checkpoints" / "RCAN_BIX3.pt"
)

# Main output directory for baseline results.

OUTPUT_DIR = (
    BASE_DIR / "results" / "pretrained_rcan"
)

# Directory for storing predicted GeoTIFF images.

PRED_DIR = OUTPUT_DIR / "predictions"

# Create output directories if they do not exist.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Select GPU if available; otherwise use CPU.

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("PRETRAINED RCAN BASELINE")
print("==========================================")

print("Device:", DEVICE)
print("Testing LR directory:", TEST_LR_DIR)
print("Testing HR directory:", TEST_HR_DIR)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 3. CHECK REQUIRED FILES AND DIRECTORIES
# ============================================================

# Check whether the required input directories exist.

if not TEST_LR_DIR.exists():
    raise FileNotFoundError(
        f"LR testing directory not found: {TEST_LR_DIR}"
    )

if not TEST_HR_DIR.exists():
    raise FileNotFoundError(
        f"HR testing directory not found: {TEST_HR_DIR}"
    )

# Check whether the RCAN source code exists.

if not RCAN_CODE_DIR.exists():
    raise FileNotFoundError(
        f"RCAN code directory not found: {RCAN_CODE_DIR}"
    )

# Check whether the pretrained checkpoint exists.

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"RCAN checkpoint not found: {CHECKPOINT_PATH}"
    )

# ============================================================
# 4. LOAD PRETRAINED RCAN CHECKPOINT
# ============================================================

# Add the RCAN source directory to Python's import path.
# This allows us to import the RCAN model implementation.

sys.path.insert(0, str(RCAN_CODE_DIR))

from model import rcan


# Load pretrained model weights.

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

print("\nCheckpoint loaded successfully.")

print(
    "Number of checkpoint parameters:",
    len(state_dict)
)


# ============================================================
# 5. DETECT RCAN ARCHITECTURE FROM CHECKPOINT
# ============================================================

# RCAN consists of Residual Groups and Residual Blocks.
# Their numbers can be detected from checkpoint parameter names.

groups = set()
blocks = set()

for key in state_dict.keys():

    match = re.match(
        r"body\.(\d+)\.body\.(\d+)\.body\.0\.weight",
        key
    )

    if match:

        groups.add(int(match.group(1)))
        blocks.add(int(match.group(2)))


# Determine the total number of residual groups and blocks.

if not groups or not blocks:
    raise ValueError(
        "Could not detect RCAN architecture from checkpoint."
    )

n_resgroups = max(groups) + 1
n_resblocks = max(blocks) + 1

print("\nRCAN Architecture")
print("------------------------------")
print("Residual groups:", n_resgroups)
print("Residual blocks:", n_resblocks)


# ============================================================
# 6. CONFIGURE AND INITIALIZE RCAN MODEL
# ============================================================

# Configure the model using the architecture parameters.
# These settings must match the pretrained checkpoint.

args = SimpleNamespace(

    n_resgroups=n_resgroups,

    n_resblocks=n_resblocks,

    n_feats=64,

    reduction=16,

    scale=[3],

    n_colors=3,

    rgb_range=255,

    res_scale=1,

    data_train="DIV2K"
)


# Initialize the RCAN architecture.

model = rcan.make_model(args)


# Load pretrained weights into the model.

model.load_state_dict(
    state_dict,
    strict=True
)


# Move the model to the selected device.

model = model.to(DEVICE)


# Set the model to evaluation mode.

model.eval()

print("\nRCAN loaded successfully!")
print("Scale factor: 3x")
print("Device:", DEVICE)


# ============================================================
# 7. LOAD AND MATCH TEST DATASET PAIRS
# ============================================================

# Read all LR and HR GeoTIFF filenames.

lr_files = sorted(
    TEST_LR_DIR.glob("*.tif")
)

hr_files = sorted(
    TEST_HR_DIR.glob("*.tif")
)


# Create dictionaries using filenames as keys.
# This ensures that corresponding LR and HR images
# are matched correctly.

lr_names = {
    file.name: file for file in lr_files
}

hr_names = {
    file.name: file for file in hr_files
}


# Identify matching filenames.

common_names = sorted(
    set(lr_names) & set(hr_names)
)


# Identify files that do not have matching pairs.

missing_hr = sorted(
    set(lr_names) - set(hr_names)
)

missing_lr = sorted(
    set(hr_names) - set(lr_names)
)


print("\nDataset Verification")
print("------------------------------")

print("LR images:", len(lr_files))
print("HR images:", len(hr_files))
print("Matching pairs:", len(common_names))
print("Missing HR images:", len(missing_hr))
print("Missing LR images:", len(missing_lr))


# Stop execution if any image pair is missing.

if missing_hr or missing_lr:

    raise RuntimeError(
        "LR and HR datasets do not contain identical filenames."
    )


# Create a list containing corresponding LR-HR image paths.

test_pairs = [

    (lr_names[name], hr_names[name])

    for name in common_names
]


print("\nDataset pairing verified successfully!")


# ============================================================
# 8. SATELLITE REFLECTANCE CONVERSION
# ============================================================

# Landsat Collection 2 Level-2 surface reflectance formula.
#
# DN = Digital Number

def landsat_to_reflectance(dn):

    return dn * 0.0000275 - 0.2


# Sentinel-2 reflectance conversion.
#
# Divide Digital Number values by 10000.

def sentinel_to_reflectance(dn):

    return dn / 10000.0

# ============================================================
# 9. PREPARE LANDSAT IMAGE FOR RCAN
# ============================================================

def prepare_input(lr_path):

    """
    Read a Landsat image and convert it into an RCAN
    input tensor.

    Steps:
    1. Read the first image band.
    2. Convert DN values into surface reflectance.
    3. Scale reflectance to the RCAN input range.
    4. Replicate the single band into three channels.
    5. Convert the image into a PyTorch tensor.
    """

    # Read Landsat image.

    with rasterio.open(lr_path) as src:

        lr_dn = src.read(1).astype(np.float32)

        profile = src.profile.copy()


    # Convert Digital Numbers into surface reflectance.

    lr_ref = landsat_to_reflectance(lr_dn)


    # Scale reflectance to the RCAN input range.
    # RCAN expects values in the 0-255 range.

    lr_input = np.clip(
        lr_ref, 0, 1
    ) * 255.0


    # RCAN expects three input channels.
    # Replicate the single satellite band into
    # three identical channels.

    lr_rgb = np.stack(
        [lr_input] * 3,
        axis=0
    )


    # Convert NumPy array into a PyTorch tensor.

    lr_tensor = torch.from_numpy(
        lr_rgb.astype(np.float32)
    )


    # Add batch dimension.
    # Final shape: (1, 3, height, width).

    lr_tensor = lr_tensor.unsqueeze(0)


    # Move tensor to CPU or GPU.

    lr_tensor = lr_tensor.to(DEVICE)

    return lr_tensor, profile


# ============================================================
# 10. TEST RCAN INPUT AND OUTPUT SHAPES
# ============================================================

# Select the first testing image for a quick model check.

if not test_pairs:
    raise RuntimeError("No matching test image pairs found.")

sample_lr_path, sample_hr_path = test_pairs[0]

sample_tensor, sample_profile = prepare_input(
    sample_lr_path
)

print("\nRCAN Input Information")
print("------------------------------")

print("Input image:", sample_lr_path.name)
print("Input tensor shape:", sample_tensor.shape)
print("Input minimum:", sample_tensor.min().item())
print("Input maximum:", sample_tensor.max().item())


# Perform a test prediction.
# no_grad() disables gradient calculation and reduces memory use.

with torch.no_grad():

    sample_prediction = model(sample_tensor)


print("\nRCAN Prediction Information")
print("------------------------------")

print("Prediction shape:", sample_prediction.shape)
print("Prediction minimum:", sample_prediction.min().item())
print("Prediction maximum:", sample_prediction.max().item())


# ============================================================
# 11. GENERATE SUPER-RESOLUTION PREDICTIONS
# ============================================================

# Store the processing time for every image.

inference_times = []

print("\nStarting RCAN inference...")
print("Total test pairs:", len(test_pairs))


# Process every LR-HR pair in the testing dataset.

for i, (lr_path, hr_path) in enumerate(test_pairs):

    # Record the starting time.

    start_time = time.time()


    # Prepare Landsat image as RCAN input.

    lr_tensor, profile = prepare_input(lr_path)


    # Generate the super-resolution prediction.

    with torch.no_grad():

        prediction = model(lr_tensor)


    # Remove the batch dimension.
    # Shape becomes (3, HR height, HR width).

    pred_rgb = (
        prediction.squeeze(0)
        .cpu()
        .numpy()
    )


    # Average the three predicted channels.
    # This produces a single-band output image.

    pred_single = np.mean(
        pred_rgb,
        axis=0
    )


    # Convert RCAN output back into reflectance.

    pred_ref = pred_single / 255.0


    # --------------------------------------------------------
    # UPDATE GEOTIFF METADATA
    # --------------------------------------------------------

    # Update the original Landsat profile to match
    # the 3x super-resolution output dimensions.

    profile.update(

        height=pred_ref.shape[0],

        width=pred_ref.shape[1],

        count=1,

        dtype="float32",

        transform=(
            profile["transform"]
            * Affine.scale(1 / 3, 1 / 3)
        ),

        nodata=np.nan,

        compress="deflate"
    )


    # --------------------------------------------------------
    # SAVE PREDICTED IMAGE
    # --------------------------------------------------------

    output_path = PRED_DIR / lr_path.name

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            pred_ref.astype(np.float32),
            1
        )


    # Calculate the time required for this image.

    elapsed_time = time.time() - start_time

    inference_times.append(elapsed_time)


    # Display progress after every 10 images
    # and after processing the first image.

    if (i + 1) % 10 == 0 or i == 0:

        print(

            f"Processed {i + 1}/{len(test_pairs)} | "

            f"Time: {elapsed_time:.2f}s | "

            f"Image: {lr_path.name}"

        )


# ============================================================
# 12. INFERENCE TIME SUMMARY
# ============================================================

total_time = sum(inference_times)

average_time = np.mean(inference_times)

print("\nInference completed!")
print("------------------------------")

print("Total images:", len(inference_times))

print(
    "Total processing time:",
    round(total_time, 2),
    "seconds"
)

print(
    "Average processing time:",
    round(average_time, 2),
    "seconds/image"
)

# ============================================================
# 13. EVALUATE SUPER-RESOLUTION PREDICTIONS
# ============================================================

# Store evaluation metrics for every image.

results = []

print("\nStarting evaluation...")


# Evaluate every predicted image against its
# corresponding Sentinel-2 reference image.

for i, (lr_path, hr_path) in enumerate(test_pairs):

    # --------------------------------------------------------
    # READ PREDICTED IMAGE
    # --------------------------------------------------------

    pred_path = PRED_DIR / lr_path.name

    with rasterio.open(pred_path) as src:

        pred = src.read(1).astype(np.float32)

        pred_transform = src.transform

        pred_crs = src.crs


    # --------------------------------------------------------
    # READ SENTINEL-2 REFERENCE IMAGE
    # --------------------------------------------------------

    with rasterio.open(hr_path) as src:

        hr_dn = src.read(1).astype(np.float32)

        hr_transform = src.transform

        hr_crs = src.crs

        hr_nodata = src.nodata


    # Convert Sentinel-2 DN values into reflectance.

    hr = sentinel_to_reflectance(hr_dn)


    # --------------------------------------------------------
    # CHECK IMAGE ALIGNMENT
    # --------------------------------------------------------

    # Both images must have identical dimensions.

    if pred.shape != hr.shape:

        raise ValueError(
            f"Shape mismatch: {hr_path.name}"
        )


    # Both images must also have identical spatial metadata.

    if (
        pred_transform != hr_transform
        or pred_crs != hr_crs
    ):

        raise ValueError(
            f"Georeferencing mismatch: {hr_path.name}"
        )


    # --------------------------------------------------------
    # CREATE VALID PIXEL MASK
    # --------------------------------------------------------

    # Exclude pixels containing NaN or infinite values.

    valid = (
        np.isfinite(pred)
        & np.isfinite(hr)
    )


    # Exclude NoData pixels from the HR reference image.

    if hr_nodata is not None:

        if np.isnan(hr_nodata):

            valid &= ~np.isnan(hr_dn)

        else:

            valid &= hr_dn != hr_nodata


    # Skip the image if no valid pixels are available.

    if valid.sum() == 0:

        print(
            "Skipping image with no valid pixels:",
            hr_path.name
        )

        continue


    # Extract valid predicted and reference pixel values.

    p = pred[valid].astype(np.float64)

    h = hr[valid].astype(np.float64)


    # --------------------------------------------------------
    # CALCULATE RMSE
    # --------------------------------------------------------

    # Root Mean Square Error measures the average
    # magnitude of prediction errors.

    rmse = np.sqrt(
        np.mean((p - h) ** 2)
    )


    # --------------------------------------------------------
    # CALCULATE MAE
    # --------------------------------------------------------

    # Mean Absolute Error measures the average
    # absolute difference between predicted and actual values.

    mae = np.mean(
        np.abs(p - h)
    )


    # --------------------------------------------------------
    # CALCULATE PSNR
    # --------------------------------------------------------

    # Peak Signal-to-Noise Ratio measures image quality.
    # Higher PSNR generally indicates lower reconstruction error.

    psnr = peak_signal_noise_ratio(

        h,

        p,

        data_range=1.0
    )


    # --------------------------------------------------------
    # CALCULATE SSIM
    # --------------------------------------------------------

    # Structural Similarity Index measures structural
    # similarity between predicted and reference images.

    # Replace invalid pixels with zero temporarily
    # so that SSIM can be calculated.

    pred_filled = np.where(
        valid,
        pred,
        0
    )

    hr_filled = np.where(
        valid,
        hr,
        0
    )


    # Calculate the complete SSIM map.

    _, ssim_map = structural_similarity(

        hr_filled,

        pred_filled,

        data_range=1.0,

        full=True
    )


    # Exclude invalid pixels and their immediate surroundings.

    ssim_mask = binary_erosion(

        valid,

        structure=np.ones((7, 7))
    )


    # Calculate average SSIM over valid interior pixels.

    ssim = (

        np.mean(ssim_map[ssim_mask])

        if ssim_mask.any()

        else np.nan

    )


    # --------------------------------------------------------
    # CALCULATE PEARSON CORRELATION
    # --------------------------------------------------------

    # Pearson correlation measures the relationship
    # between predicted and reference pixel values.

    if np.std(p) > 0 and np.std(h) > 0:

        correlation = np.corrcoef(
            p,
            h
        )[0, 1]

    else:

        correlation = np.nan


    # --------------------------------------------------------
    # STORE IMAGE EVALUATION RESULTS
    # --------------------------------------------------------

    results.append({

        "Image": hr_path.name,

        "Valid_Pixels": int(valid.sum()),

        "PSNR": psnr,

        "SSIM": ssim,

        "RMSE": rmse,

        "MAE": mae,

        "Correlation": correlation

    })


    # Display evaluation progress.

    if (i + 1) % 25 == 0:

        print(
            f"Evaluated {i + 1}/{len(test_pairs)} images"
        )


# ============================================================
# 14. SAVE INDIVIDUAL IMAGE METRICS
# ============================================================

# Convert the results list into a Pandas DataFrame.

df = pd.DataFrame(results)


# Save the evaluation metrics of each image as a CSV file.

individual_metrics_path = (
    OUTPUT_DIR / "pretrained_metrics_per_image.csv"
)

df.to_csv(
    individual_metrics_path,
    index=False
)


print("\nEvaluation completed!")

print("Images evaluated:", len(df))

print(
    "Individual metrics saved:",
    individual_metrics_path
)


# ============================================================
# 15. CALCULATE OVERALL EVALUATION SUMMARY
# ============================================================

# Select the evaluation metrics that will be summarized.

metric_columns = [

    "PSNR",

    "SSIM",

    "RMSE",

    "MAE",

    "Correlation"

]


# Calculate mean, standard deviation, minimum
# and maximum values for every metric.

summary = df[metric_columns].agg(

    ["mean", "std", "min", "max"]

).T


# Display the final evaluation results.

print("\nPRETRAINED RCAN EVALUATION")

print("=" * 45)

print(
    summary.round(6)
)


# Save the overall evaluation summary.

summary_path = (
    OUTPUT_DIR / "pretrained_metrics_summary.csv"
)

summary.to_csv(
    summary_path
)


print("\nSummary saved:", summary_path)


# ============================================================
# 16. VISUAL COMPARISON OF LR, HR AND PREDICTED IMAGES
# ============================================================

# Select the first five test images for visualization.

image_files = sorted(
    TEST_HR_DIR.glob("*.tif")
)[:5]


# Create a figure with five rows and three columns.
#
# Column 1: Landsat-8 LR image.
# Column 2: Sentinel-2 HR reference image.
# Column 3: RCAN predicted HR image.

fig, axes = plt.subplots(

    5,

    3,

    figsize=(16, 24)

)


# ------------------------------------------------------------
# IMAGE CONTRAST STRETCHING
# ------------------------------------------------------------

def display_scale(img):

    """
    Calculate the 2nd and 98th percentile values
    for image visualization.

    Percentile stretching improves image contrast
    without changing the original image data.
    """

    valid = img[np.isfinite(img)]

    if valid.size == 0:

        return 0, 1

    vmin, vmax = np.percentile(
        valid,
        [2, 98]
    )

    if vmax <= vmin:

        vmax = vmin + 1

    return vmin, vmax


# ------------------------------------------------------------
# DISPLAY IMAGE PAIRS
# ------------------------------------------------------------

for i, hr_path in enumerate(image_files):

    filename = hr_path.name


    # Read LR Landsat image.

    with rasterio.open(
        TEST_LR_DIR / filename
    ) as src:

        lr = src.read(1).astype(np.float32)


    # Read HR Sentinel-2 image.

    with rasterio.open(
        hr_path
    ) as src:

        hr = src.read(1).astype(np.float32)


    # Read RCAN predicted image.

    with rasterio.open(
        PRED_DIR / filename
    ) as src:

        pred = src.read(1).astype(np.float32)


    # Store the three images and their titles.

    images = [

        lr,

        hr,

        pred

    ]

    titles = [

        "LR - Landsat 8",

        "HR - Sentinel-2",

        "Predicted HR - RCAN"

    ]


    # Display all three images in the corresponding row.

    for j, (img, title) in enumerate(
        zip(images, titles)
    ):

        vmin, vmax = display_scale(img)


        axes[i, j].imshow(

            img,

            cmap="gray",

            vmin=vmin,

            vmax=vmax

        )


        axes[i, j].set_title(

            f"{title}\n"
            f"Range: {vmin:.4f} - {vmax:.4f}"

        )


        axes[i, j].axis("off")


# Adjust spacing between subplots.

plt.tight_layout()


# Save the visual comparison as a high-resolution PNG.

visualization_path = (
    OUTPUT_DIR / "RCAN_visual_comparison.png"
)

plt.savefig(

    visualization_path,

    dpi=200,

    bbox_inches="tight"

)


# Display the comparison figure.

plt.show()


print(
    "\nVisual comparison saved:",
    visualization_path
)


# ============================================================
# 17. FINAL PROJECT SUMMARY
# ============================================================

print("\n==========================================")

print("PRETRAINED RCAN BASELINE COMPLETED")

print("==========================================")

print("Total testing pairs:", len(test_pairs))

print("Successfully evaluated:", len(df))

print("Predictions directory:", PRED_DIR)

print("Individual metrics:", individual_metrics_path)

print("Summary metrics:", summary_path)

print("Visualization:", visualization_path)

print("\nAll processing steps completed successfully!")
