
# ============================================================
# 5. RCAN FINE-TUNING FOR LANDSAT 8/9 TO SENTINEL-2
# ============================================================
# Project: Satellite Image Super-Resolution
# Model: Residual Channel Attention Network (RCAN)
# Scale: 3x
# Training images: 1050
# Testing images: 225
# GPU: NVIDIA RTX 4000 Ada Generation
# ============================================================

import os
import sys
import re
import time
import random

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from rasterio.transform import Affine
from tqdm.auto import tqdm

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from scipy.ndimage import binary_erosion


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(
    r"C:\Users\ARF-G5\Desktop\Abhijeet"
)

os.chdir(BASE_DIR)

DATASET_DIR = BASE_DIR / "prepared_dataset"

TRAIN_LR_DIR = DATASET_DIR / "train" / "LR"
TRAIN_HR_DIR = DATASET_DIR / "train" / "HR"

TEST_LR_DIR = DATASET_DIR / "test" / "LR"
TEST_HR_DIR = DATASET_DIR / "test" / "HR"

RCAN_CODE = (
    BASE_DIR / "RCAN" / "RCAN_TestCode" / "code"
)

CHECKPOINT = (
    BASE_DIR / "checkpoints" / "RCAN_BIX3.pt"
)

FINETUNE_DIR = (
    BASE_DIR / "results" / "finetuned_rcan"
)

OUTPUT_DIR = (
    BASE_DIR / "RCAN_finetuned_model_2_results"
)

PRED_DIR = OUTPUT_DIR / "predictions"

FINETUNE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PRED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. GPU CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("RCAN FINE-TUNING CONFIGURATION")
print("=" * 65)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory / 1024**3,
            2
        ),
        "GB"
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

else:
    print("WARNING: CUDA is unavailable. Using CPU.")


# ============================================================
# 4. TRAINING DATASET PAIRING
# ============================================================

train_lr_files = sorted(
    TRAIN_LR_DIR.glob("*.tif")
)

train_hr_files = sorted(
    TRAIN_HR_DIR.glob("*.tif")
)

lr_names = {
    f.name for f in train_lr_files
}

hr_names = {
    f.name for f in train_hr_files
}

common_names = sorted(
    lr_names & hr_names
)

assert len(common_names) > 0, (
    "No matching training image pairs found!"
)

assert lr_names == hr_names, (
    "Missing training image pairs!"
)

train_pairs = [
    (
        TRAIN_LR_DIR / name,
        TRAIN_HR_DIR / name
    )
    for name in common_names
]

print("\nTraining image pairs:", len(train_pairs))


# ============================================================
# 5. TEST DATASET PAIRING
# ============================================================

test_lr_files = sorted(
    TEST_LR_DIR.glob("*.tif")
)

test_hr_files = sorted(
    TEST_HR_DIR.glob("*.tif")
)

test_lr_names = {
    f.name for f in test_lr_files
}

test_hr_names = {
    f.name for f in test_hr_files
}

test_common_names = sorted(
    test_lr_names & test_hr_names
)

assert len(test_common_names) > 0, (
    "No matching test image pairs found!"
)

assert test_lr_names == test_hr_names, (
    "Missing test image pairs!"
)

test_pairs = [
    (
        TEST_LR_DIR / name,
        TEST_HR_DIR / name
    )
    for name in test_common_names
]

print("Testing image pairs:", len(test_pairs))


# ============================================================
# 6. DATASET CLASS
# ============================================================

class LandsatSentinelDataset(Dataset):

    def __init__(self, pairs):

        self.pairs = pairs

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        lr_path, hr_path = self.pairs[idx]

        # ----------------------------------------------------
        # Read Landsat LR image
        # ----------------------------------------------------

        with rasterio.open(lr_path) as src:

            lr_dn = src.read(1).astype(
                np.float32
            )

            lr_nodata = src.nodata

        # ----------------------------------------------------
        # Read Sentinel-2 HR image
        # ----------------------------------------------------

        with rasterio.open(hr_path) as src:

            hr_dn = src.read(1).astype(
                np.float32
            )

            hr_nodata = src.nodata

        # ----------------------------------------------------
        # Reflectance conversion
        # ----------------------------------------------------

        # Landsat Collection 2 Level-2
        lr_ref = (
            lr_dn * 0.0000275 - 0.2
        )

        # Sentinel-2 Level-2A
        hr_ref = (
            hr_dn / 10000.0
        )

        # ----------------------------------------------------
        # Convert invalid pixels to zero
        # ----------------------------------------------------

        if lr_nodata is not None:

            lr_ref[
                lr_dn == lr_nodata
            ] = 0.0

        if hr_nodata is not None:

            hr_ref[
                hr_dn == hr_nodata
            ] = 0.0

        lr_ref = np.nan_to_num(
            lr_ref,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        hr_ref = np.nan_to_num(
            hr_ref,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Convert single-band images to three channels
        # ----------------------------------------------------

        lr_rgb = np.stack(
            [lr_ref * 255.0] * 3,
            axis=0
        )

        hr_rgb = np.stack(
            [hr_ref * 255.0] * 3,
            axis=0
        )

        # ----------------------------------------------------
        # Convert NumPy arrays to PyTorch tensors
        # ----------------------------------------------------

        lr_tensor = torch.from_numpy(
            lr_rgb.astype(np.float32)
        )

        hr_tensor = torch.from_numpy(
            hr_rgb.astype(np.float32)
        )

        return lr_tensor, hr_tensor


# ============================================================
# 7. CREATE TRAINING DATASET
# ============================================================

BATCH_SIZE = 4

train_dataset = LandsatSentinelDataset(
    train_pairs
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("\nTraining batches:", len(train_loader))


# ============================================================
# 8. CREATE TEST DATASET
# ============================================================

test_dataset = LandsatSentinelDataset(
    test_pairs
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Testing batches:", len(test_loader))


# ============================================================
# 9. VERIFY DATASET DIMENSIONS
# ============================================================

lr_sample, hr_sample = next(
    iter(train_loader)
)

print("\nDataset Verification")
print("-" * 45)

print(
    "Training LR shape:",
    lr_sample.shape
)

print(
    "Training HR shape:",
    hr_sample.shape
)

print(
    "Testing LR shape:",
    next(iter(test_loader))[0].shape
)

print("\nPart 1 completed successfully!")

# ============================================================
# 10. LOAD PRETRAINED RCAN BIX3 MODEL
# ============================================================

print("\n" + "=" * 65)
print("LOADING PRETRAINED RCAN MODEL")
print("=" * 65)

assert RCAN_CODE.exists(), (
    f"RCAN code directory not found: {RCAN_CODE}"
)

assert CHECKPOINT.exists(), (
    f"Pretrained checkpoint not found: {CHECKPOINT}"
)

# Add RCAN source code directory
sys.path.insert(
    0,
    str(RCAN_CODE)
)

from model import rcan


# ============================================================
# 11. LOAD PRETRAINED CHECKPOINT
# ============================================================

state_dict = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

# Handle checkpoints containing model_state_dict
if "model_state_dict" in state_dict:

    state_dict = state_dict[
        "model_state_dict"
    ]

# Detect RCAN architecture from checkpoint keys
groups = set()
blocks = set()

for key in state_dict.keys():

    match = re.match(
        r"body\.(\d+)\.body\.(\d+)\.body\.0\.weight",
        key
    )

    if match:

        groups.add(
            int(match.group(1))
        )

        blocks.add(
            int(match.group(2))
        )

assert groups and blocks, (
    "Unable to detect RCAN architecture "
    "from checkpoint keys!"
)

n_resgroups = max(groups) + 1
n_resblocks = max(blocks) + 1

print(
    "Residual groups:",
    n_resgroups
)

print(
    "Residual blocks:",
    n_resblocks
)


# ============================================================
# 12. CONFIGURE RCAN ARCHITECTURE
# ============================================================

from types import SimpleNamespace

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

# Initialize RCAN architecture
model = rcan.make_model(args)

# Load pretrained weights
model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(DEVICE)

print("\nPretrained RCAN loaded successfully!")

print("Scale: 3x")

print("Device:", DEVICE)

print(
    "Architecture:",
    type(model).__name__
)


# ============================================================
# 13. VERIFY MODEL INPUT AND OUTPUT
# ============================================================

model.eval()

with torch.no_grad():

    sample_lr = lr_sample[:1].to(
        DEVICE
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=DEVICE.type == "cuda"
    ):

        sample_prediction = model(
            sample_lr
        )

print("\nModel Verification")
print("-" * 45)

print(
    "Input shape:",
    tuple(sample_lr.shape)
)

print(
    "Output shape:",
    tuple(sample_prediction.shape)
)

expected_height = sample_lr.shape[-2] * 3
expected_width = sample_lr.shape[-1] * 3

assert sample_prediction.shape[-2:] == (
    expected_height,
    expected_width
), "RCAN output is not 3x the LR dimensions!"

print("RCAN 3x output verified successfully!")

del sample_lr
del sample_prediction

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 14. FINE-TUNING CONFIGURATION
# ============================================================

print("\n" + "=" * 65)
print("RCAN FINE-TUNING CONFIGURATION")
print("=" * 65)

# Loss function
criterion = nn.L1Loss()

# Adam optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-5,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

# Training configuration
EPOCHS = 10

train_losses = []

epoch_times = []

best_loss = float("inf")

# Mixed precision gradient scaler
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=DEVICE.type == "cuda"
)

print("Epochs:", EPOCHS)

print("Batch size:", BATCH_SIZE)

print("Initial learning rate:", 1e-5)

print("Loss function: L1 Loss")

print("Optimizer: Adam")

print("Mixed precision:", DEVICE.type == "cuda")


# ============================================================
# 15. RCAN FINE-TUNING TRAINING LOOP
# ============================================================

print("\n" + "=" * 65)
print("STARTING RCAN FINE-TUNING")
print("=" * 65)

training_start = time.time()

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    start_time = time.time()

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        leave=True
    )

    for batch_idx, (lr, hr) in enumerate(
        progress_bar
    ):

        # Move data to GPU
        lr = lr.to(
            DEVICE,
            non_blocking=True
        )

        hr = hr.to(
            DEVICE,
            non_blocking=True
        )

        # Clear previous gradients
        optimizer.zero_grad(
            set_to_none=True
        )

        # Mixed precision forward pass
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda"
        ):

            prediction = model(lr)

            loss = criterion(
                prediction,
                hr
            )

        # Backpropagation
        scaler.scale(
            loss
        ).backward()

        # Optimizer update
        scaler.step(
            optimizer
        )

        scaler.update()

        # Accumulate training loss
        running_loss += loss.detach().item()

        current_avg_loss = (
            running_loss / (batch_idx + 1)
        )

        # Update progress bar
        progress_bar.set_postfix(

            batch_loss=f"{loss.item():.6f}",

            avg_loss=f"{current_avg_loss:.6f}",

            lr=f"{optimizer.param_groups[0]['lr']:.2e}"

        )

    # --------------------------------------------------------
    # Epoch statistics
    # --------------------------------------------------------

    epoch_loss = (
        running_loss / len(train_loader)
    )

    train_losses.append(epoch_loss)

    scheduler.step(epoch_loss)

    epoch_time = time.time() - start_time

    epoch_times.append(epoch_time)

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"| Loss: {epoch_loss:.6f} "
        f"| LR: {optimizer.param_groups[0]['lr']:.2e} "
        f"| Time: {epoch_time:.2f}s"
    )

    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            FINETUNE_DIR / "best_finetuned_rcan.pth"
        )

        print(
            "Best model checkpoint saved!"
        )

    # --------------------------------------------------------
    # Save latest checkpoint
    # --------------------------------------------------------

    torch.save(

        {

            "epoch": epoch + 1,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "scheduler_state_dict": scheduler.state_dict(),

            "best_training_loss": best_loss,

            "train_losses": train_losses

        },

        FINETUNE_DIR / "latest_finetuned_rcan_checkpoint.pth"

    )

    print("-" * 65)


# ============================================================
# 16. TRAINING COMPLETION
# ============================================================

total_training_time = (
    time.time() - training_start
)

print("\n" + "=" * 65)

print("RCAN FINE-TUNING COMPLETED")

print("=" * 65)

print(
    "Best training loss:",
    round(best_loss, 6)
)

print(
    "Total training time:",
    round(total_training_time, 2),
    "seconds"
)

print(
    "Total training time:",
    round(total_training_time / 3600, 2),
    "hours"
)

# Save training history
training_history = pd.DataFrame({

    "Epoch": range(1, EPOCHS + 1),

    "Training_Loss": train_losses,

    "Epoch_Time_Seconds": epoch_times

})

training_history.to_csv(

    FINETUNE_DIR / "training_history.csv",

    index=False

)

print(
    "Training history saved successfully!"
)


# ============================================================
# 17. LOAD BEST FINE-TUNED RCAN MODEL
# ============================================================

best_checkpoint_path = (
    FINETUNE_DIR / "best_finetuned_rcan.pth"
)

model.load_state_dict(

    torch.load(

        best_checkpoint_path,

        map_location=DEVICE,

        weights_only=True

    )

)

model = model.to(DEVICE)

model.eval()

print(
    "\nBest fine-tuned RCAN model loaded successfully!"
)


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

torch.save(

    model.state_dict(),

    FINETUNE_DIR / "RCAN_finetuned_model_2.pth"

)

torch.save(

    {

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "model_training_status": "fine_tuned",

        "best_training_loss": best_loss,

        "model_name": "RCAN_finetuned_model_2",

        "epochs": EPOCHS,

        "scale": 3

    },

    FINETUNE_DIR / "RCAN_finetuned_model_2_checkpoint.pth"

)

print(
    "\nFinal RCAN model and checkpoint saved successfully!"
)

print(
    "Model directory:",
    FINETUNE_DIR.resolve()
)

# ============================================================
# 19. LOAD BEST FINE-TUNED RCAN MODEL
# ============================================================

print("\n" + "=" * 65)
print("LOADING BEST FINE-TUNED RCAN MODEL")
print("=" * 65)

best_checkpoint_path = (
    FINETUNE_DIR / "best_finetuned_rcan.pth"
)

model.load_state_dict(

    torch.load(
        best_checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

)

model = model.to(DEVICE)

model.eval()

print(
    "Best fine-tuned RCAN model loaded successfully!"
)


# ============================================================
# 20. CREATE PREDICTION OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    BASE_DIR / "RCAN_finetuned_model_2_results"
)

PRED_DIR = OUTPUT_DIR / "predictions"

PRED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "Prediction directory:",
    PRED_DIR.resolve()
)


# ============================================================
# 21. GENERATE SUPER-RESOLUTION PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("GENERATING RCAN SUPER-RESOLUTION PREDICTIONS")
print("=" * 65)

inference_times = []

model.eval()

with torch.no_grad():

    for i, (lr, hr) in enumerate(

        tqdm(
            test_loader,
            desc="RCAN Model 2 Predictions"
        )

    ):

        start = time.time()

        lr = lr.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # RCAN inference using mixed precision
        # ----------------------------------------------------

        with torch.autocast(

            device_type="cuda",

            dtype=torch.float16,

            enabled=DEVICE.type == "cuda"

        ):

            prediction = model(lr)

        # ----------------------------------------------------
        # Convert prediction to NumPy
        # ----------------------------------------------------

        pred_rgb = (
            prediction
            .squeeze(0)
            .float()
            .cpu()
            .numpy()
        )

        # Average three replicated channels
        pred_ref = (
            np.mean(pred_rgb, axis=0) / 255.0
        )

        # ----------------------------------------------------
        # Get corresponding test image
        # ----------------------------------------------------

        lr_path, hr_path = test_pairs[i]

        # ----------------------------------------------------
        # Read original LR GeoTIFF metadata
        # ----------------------------------------------------

        with rasterio.open(lr_path) as src:

            profile = src.profile.copy()

            lr_transform = src.transform

            lr_width = src.width

            lr_height = src.height

        # ----------------------------------------------------
        # Update GeoTIFF metadata
        # ----------------------------------------------------

        profile.update(

            height=pred_ref.shape[0],

            width=pred_ref.shape[1],

            count=1,

            dtype="float32",

            transform=(
                lr_transform *
                Affine.scale(
                    lr_width / pred_ref.shape[1],
                    lr_height / pred_ref.shape[0]
                )
            ),

            nodata=np.nan,

            compress="deflate"

        )

        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        output_path = (
            PRED_DIR / lr_path.name
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                pred_ref.astype(np.float32),
                1
            )

        inference_times.append(
            time.time() - start
        )


# ============================================================
# 22. INFERENCE PERFORMANCE SUMMARY
# ============================================================

total_inference_time = sum(
    inference_times
)

average_inference_time = np.mean(
    inference_times
)

print("\n" + "=" * 65)

print("RCAN PREDICTION GENERATION COMPLETED")

print("=" * 65)

print(
    "Total predictions:",
    len(inference_times)
)

print(
    "Total inference time:",
    round(total_inference_time, 2),
    "seconds"
)

print(
    "Average inference time:",
    round(average_inference_time, 4),
    "seconds/image"
)

print(
    "Predictions saved in:",
    PRED_DIR.resolve()
)


# ============================================================
# 23. EVALUATION METRICS
# ============================================================

print("\n" + "=" * 65)

print("STARTING RCAN MODEL 2 EVALUATION")

print("=" * 65)

results = []

for i, (lr_path, hr_path) in enumerate(

    tqdm(
        test_pairs,
        desc="Evaluating RCAN Predictions"
    )

):

    pred_path = (
        PRED_DIR / lr_path.name
    )

    # --------------------------------------------------------
    # Read predicted image
    # --------------------------------------------------------

    with rasterio.open(pred_path) as src:

        pred = src.read(1).astype(
            np.float32
        )

        pred_transform = src.transform

        pred_crs = src.crs

        pred_nodata = src.nodata

    # --------------------------------------------------------
    # Read Sentinel-2 HR reference
    # --------------------------------------------------------

    with rasterio.open(hr_path) as src:

        hr_dn = src.read(1).astype(
            np.float32
        )

        hr_transform = src.transform

        hr_crs = src.crs

        hr_nodata = src.nodata

    # --------------------------------------------------------
    # Sentinel-2 reflectance conversion
    # --------------------------------------------------------

    hr = hr_dn / 10000.0

    # --------------------------------------------------------
    # Verify spatial alignment
    # --------------------------------------------------------

    if pred.shape != hr.shape:

        raise ValueError(
            f"Shape mismatch: {hr_path.name}"
        )

    if (
        pred_transform != hr_transform
        or pred_crs != hr_crs
    ):

        raise ValueError(
            f"Georeferencing mismatch: {hr_path.name}"
        )

    # --------------------------------------------------------
    # Valid pixel mask
    # --------------------------------------------------------

    valid = (
        np.isfinite(pred)
        &
        np.isfinite(hr)
        &
        np.isfinite(hr_dn)
    )

    if pred_nodata is not None:

        if np.isnan(pred_nodata):

            valid &= ~np.isnan(pred)

        else:

            valid &= pred != pred_nodata

    if hr_nodata is not None:

        if np.isnan(hr_nodata):

            valid &= ~np.isnan(hr_dn)

        else:

            valid &= hr_dn != hr_nodata

    if valid.sum() == 0:

        print(
            "Skipping:",
            hr_path.name
        )

        continue

    # --------------------------------------------------------
    # Extract valid pixels
    # --------------------------------------------------------

    p = pred[valid].astype(
        np.float64
    )

    h = hr[valid].astype(
        np.float64
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        np.mean((p - h) ** 2)
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = np.mean(
        np.abs(p - h)
    )

    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr = peak_signal_noise_ratio(
        h,
        p,
        data_range=1.0
    )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

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

    _, ssim_map = structural_similarity(

        hr_filled,

        pred_filled,

        data_range=1.0,

        full=True

    )

    # Exclude invalid pixels and surrounding pixels
    ssim_mask = binary_erosion(

        valid,

        structure=np.ones((7, 7))

    )

    if ssim_mask.any():

        ssim = np.mean(
            ssim_map[ssim_mask]
        )

    else:

        ssim = np.nan

    # --------------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------------

    if (
        np.std(p) > 0
        and np.std(h) > 0
    ):

        correlation = np.corrcoef(
            p,
            h
        )[0, 1]

    else:

        correlation = np.nan

    # --------------------------------------------------------
    # Store evaluation results
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

    if (i + 1) % 25 == 0:

        print(
            f"Evaluated {i + 1}/{len(test_pairs)} images"
        )


# ============================================================
# 24. CREATE EVALUATION DATAFRAME
# ============================================================

df = pd.DataFrame(results)

assert len(df) > 0, (
    "No valid evaluation results generated!"
)

# Save individual image metrics
metrics_path = (

    OUTPUT_DIR /
    "finetuned_rcan_model_2_metrics_per_image.csv"

)

df.to_csv(

    metrics_path,

    index=False

)

print("\n" + "=" * 65)

print("RCAN MODEL 2 EVALUATION COMPLETED")

print("=" * 65)

print(
    "Images evaluated:",
    len(df)
)

print(
    "Metrics saved:",
    metrics_path
)


# ============================================================
# 25. CALCULATE OVERALL EVALUATION METRICS
# ============================================================

metric_columns = [

    "PSNR",

    "SSIM",

    "RMSE",

    "MAE",

    "Correlation"

]

overall_metrics = df[
    metric_columns
].mean()

print("\n========== RCAN MODEL 2 OVERALL METRICS ==========")

print(
    overall_metrics.round(6)
)

# Save overall metrics
overall_metrics.to_csv(

    OUTPUT_DIR /
    "finetuned_rcan_model_2_overall_metrics.csv",

    header=["Value"]

)

print(
    "\nOverall metrics saved successfully!"
)


# ============================================================
# 26. STATISTICAL SUMMARY
# ============================================================

summary = df[
    metric_columns
].agg(

    [
        "mean",

        "std",

        "min",

        "max"

    ]

).T

print("\nFINE-TUNED RCAN MODEL 2 EVALUATION")

print("=" * 55)

print(
    summary.round(6)
)

# Save statistical summary
summary_path = (

    OUTPUT_DIR /
    "finetuned_rcan_model_2_metrics_summary.csv"

)

summary.to_csv(
    summary_path
)

print(
    "\nModel 2 evaluation summary saved successfully!"
)

print(
    summary_path.resolve()
)


# ============================================================
# 27. SAVE COMPLETE EVALUATION REPORT
# ============================================================

report_path = (
    OUTPUT_DIR / "RCAN_Model_2_Evaluation_Report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RCAN FINE-TUNED MODEL 2 EVALUATION REPORT\n"
    )

    f.write("=" * 55 + "\n\n")

    f.write(
        f"Total test images: {len(df)}\n"
    )

    f.write(
        f"Total inference time: {total_inference_time:.2f} seconds\n"
    )

    f.write(
        f"Average inference time: {average_inference_time:.4f} seconds/image\n\n"
    )

    f.write(
        "OVERALL EVALUATION METRICS\n"
    )

    f.write(
        overall_metrics.to_string()
    )

    f.write("\n\nSTATISTICAL SUMMARY\n")

    f.write(
        summary.to_string()
    )

print(
    "\nComplete evaluation report saved:",
    report_path.resolve()
)

print("\nPART 3 COMPLETED SUCCESSFULLY!")

# ============================================================
# 28. VISUAL COMPARISON CONFIGURATION
# ============================================================

print("\n" + "=" * 65)

print("GENERATING RCAN VISUAL COMPARISONS")

print("=" * 65)

# Input directories
lr_dir = BASE_DIR / "prepared_dataset" / "test" / "LR"

hr_dir = BASE_DIR / "prepared_dataset" / "test" / "HR"

# Pretrained RCAN predictions
pretrained_dir = (
    BASE_DIR / "results" / "pretrained_rcan" / "predictions"
)

# Fine-tuned RCAN Model 2 predictions
finetuned_dir = (
    BASE_DIR /
    "RCAN_finetuned_model_2_results" /
    "predictions"
)

# Output directory
OUTPUT_DIR = (
    BASE_DIR / "RCAN_finetuned_model_2_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 29. DISPLAY SCALING FUNCTION
# ============================================================

def display_scale(img):

    valid = img[np.isfinite(img)]

    if len(valid) == 0:

        return 0, 1

    vmin, vmax = np.percentile(
        valid,
        [2, 98]
    )

    if vmin == vmax:

        vmax = vmin + 1e-6

    return vmin, vmax


# ============================================================
# 30. SELECT FIRST FIVE TEST IMAGES
# ============================================================

image_files = sorted(
    hr_dir.glob("*.tif")
)[:5]

assert len(image_files) > 0, (
    "No test images available for visualization!"
)


# ============================================================
# 31. CREATE FIVE-IMAGE VISUAL COMPARISON
# ============================================================

fig, axes = plt.subplots(

    5,

    4,

    figsize=(20, 24)

)

for i, hr_path in enumerate(image_files):

    filename = hr_path.name

    # --------------------------------------------------------
    # Read LR image
    # --------------------------------------------------------

    with rasterio.open(
        lr_dir / filename
    ) as src:

        lr = src.read(1).astype(
            np.float32
        )

    # --------------------------------------------------------
    # Read HR image
    # --------------------------------------------------------

    with rasterio.open(
        hr_path
    ) as src:

        hr = src.read(1).astype(
            np.float32
        )

    # --------------------------------------------------------
    # Read pretrained RCAN prediction
    # --------------------------------------------------------

    with rasterio.open(
        pretrained_dir / filename
    ) as src:

        pretrained = src.read(1).astype(
            np.float32
        )

    # --------------------------------------------------------
    # Read fine-tuned RCAN prediction
    # --------------------------------------------------------

    with rasterio.open(
        finetuned_dir / filename
    ) as src:

        finetuned = src.read(1).astype(
            np.float32
        )

    # --------------------------------------------------------
    # Images for comparison
    # --------------------------------------------------------

    images = [

        lr,

        hr,

        pretrained,

        finetuned

    ]

    titles = [

        "LR - Landsat 8/9",

        "HR - Sentinel-2",

        "Pretrained RCAN",

        "Fine-tuned RCAN Model 2"

    ]

    # --------------------------------------------------------
    # Plot images
    # --------------------------------------------------------

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

            f"{title}\nRange: {vmin:.4f} - {vmax:.4f}",

            fontsize=10

        )

        axes[i, j].axis("off")


# Overall title
plt.suptitle(

    "Visual Comparison: Pretrained RCAN vs Fine-tuned RCAN Model 2",

    fontsize=18,

    y=1.01

)

plt.tight_layout()

# Save comparison figure
comparison_path = (

    OUTPUT_DIR /
    "RCAN_Pretrained_vs_Finetuned_Model2_Comparison.png"

)

plt.savefig(

    comparison_path,

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print(
    "\nFive-image visual comparison completed!"
)

print(
    "Saved:",
    comparison_path.resolve()
)


# ============================================================
# 32. INDIVIDUAL PATCH COMPARISON
# ============================================================

print("\nGenerating individual patch comparison...")


# Select any one patch
# 0 = first patch
# 1 = second patch
# 2 = third patch

patch_index = 0

all_image_files = sorted(
    hr_dir.glob("*.tif")
)

assert patch_index < len(all_image_files), (
    "Invalid patch index!"
)

hr_path = all_image_files[patch_index]

filename = hr_path.name


# ------------------------------------------------------------
# Read LR image
# ------------------------------------------------------------

with rasterio.open(
    lr_dir / filename
) as src:

    lr = src.read(1).astype(
        np.float32
    )


# ------------------------------------------------------------
# Read HR image
# ------------------------------------------------------------

with rasterio.open(
    hr_path
) as src:

    hr = src.read(1).astype(
        np.float32
    )


# ------------------------------------------------------------
# Read pretrained prediction
# ------------------------------------------------------------

with rasterio.open(
    pretrained_dir / filename
) as src:

    pretrained = src.read(1).astype(
        np.float32
    )


# ------------------------------------------------------------
# Read fine-tuned prediction
# ------------------------------------------------------------

with rasterio.open(
    finetuned_dir / filename
) as src:

    finetuned = src.read(1).astype(
        np.float32
    )


# ============================================================
# 33. CREATE INDIVIDUAL PATCH VISUALIZATION
# ============================================================

images = [

    lr,

    hr,

    pretrained,

    finetuned

]

titles = [

    "LR - Landsat 8/9",

    "HR - Sentinel-2",

    "Pretrained RCAN",

    "Fine-tuned RCAN"

]

fig, axes = plt.subplots(

    2,

    2,

    figsize=(12, 10)

)

axes = axes.flatten()

for ax, img, title in zip(
    axes,
    images,
    titles
):

    vmin, vmax = display_scale(img)

    ax.imshow(

        img,

        cmap="gray",

        vmin=vmin,

        vmax=vmax

    )

    ax.set_title(

        title,

        fontsize=12

    )

    ax.axis("off")


plt.suptitle(

    f"Visual Comparison: RCAN Models\nPatch: {filename}",

    fontsize=15

)

plt.tight_layout()


# Save individual patch comparison
patch_comparison_path = (

    OUTPUT_DIR /
    f"Patch_{patch_index}_Comparison.png"

)

plt.savefig(

    patch_comparison_path,

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print(
    "\nIndividual patch comparison completed!"
)

print(
    "Patch:",
    filename
)

print(
    "Saved:",
    patch_comparison_path.resolve()
)


# ============================================================
# 34. TRAINING LOSS VISUALIZATION
# ============================================================

print("\nGenerating training loss graph...")


fig, ax = plt.subplots(

    figsize=(10, 6)

)

ax.plot(

    range(1, len(train_losses) + 1),

    train_losses,

    marker="o",

    linewidth=2,

    color="blue",

    label="Training Loss"

)

ax.set_title(

    "RCAN Fine-Tuning Training Loss",

    fontsize=14

)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "L1 Training Loss"
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.5
)

ax.legend()

plt.tight_layout()


# Save training loss graph
loss_graph_path = (

    FINETUNE_DIR /
    "RCAN_Finetuning_Training_Loss.png"

)

plt.savefig(

    loss_graph_path,

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print(
    "Training loss graph saved:",
    loss_graph_path.resolve()
)


# ============================================================
# 35. FINAL MODEL PERFORMANCE SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("FINAL RCAN FINE-TUNING PERFORMANCE SUMMARY")

print("=" * 70)

print("\nModel: RCAN Fine-tuned Model 2")

print("Architecture: Residual Channel Attention Network")

print("Upscaling factor: 3x")

print("Training images:", len(train_pairs))

print("Testing images:", len(test_pairs))

print("Training epochs:", EPOCHS)

print("Batch size:", BATCH_SIZE)

print("Device:", DEVICE)

print("\nBest Training Loss:")

print(round(best_loss, 6))

print("\nOverall Evaluation Metrics:")

print(overall_metrics.round(6))

print("\nInference Performance:")

print(
    "Total inference time:",
    round(total_inference_time, 2),
    "seconds"
)

print(
    "Average inference time:",
    round(average_inference_time, 4),
    "seconds/image"
)


# ============================================================
# 36. FINAL OUTPUT FILE SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("GENERATED OUTPUT FILES")

print("=" * 70)

output_files = [

    FINETUNE_DIR / "best_finetuned_rcan.pth",

    FINETUNE_DIR / "latest_finetuned_rcan_checkpoint.pth",

    FINETUNE_DIR / "RCAN_finetuned_model_2.pth",

    FINETUNE_DIR / "RCAN_finetuned_model_2_checkpoint.pth",

    FINETUNE_DIR / "training_history.csv",

    FINETUNE_DIR / "RCAN_Finetuning_Training_Loss.png",

    OUTPUT_DIR / "finetuned_rcan_model_2_metrics_per_image.csv",

    OUTPUT_DIR / "finetuned_rcan_model_2_overall_metrics.csv",

    OUTPUT_DIR / "finetuned_rcan_model_2_metrics_summary.csv",

    OUTPUT_DIR / "RCAN_Model_2_Evaluation_Report.txt",

    OUTPUT_DIR / "RCAN_Pretrained_vs_Finetuned_Model2_Comparison.png",

    OUTPUT_DIR / f"Patch_{patch_index}_Comparison.png"

]

for file_path in output_files:

    if file_path.exists():

        print(
            "[SAVED]",
            file_path.name
        )

    else:

        print(
            "[MISSING]",
            file_path.name
        )


# ============================================================
# 37. FINAL COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)

print("RCAN FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY!")

print("=" * 70)

print(
    "\nFine-tuned model directory:"
)

print(
    FINETUNE_DIR.resolve()
)

print(
    "\nPrediction directory:"
)

print(
    PRED_DIR.resolve()
)

print(
    "\nEvaluation directory:"
)

print(
    OUTPUT_DIR.resolve()
)

print(
    "\nAll RCAN fine-tuning and evaluation operations completed!"
)
