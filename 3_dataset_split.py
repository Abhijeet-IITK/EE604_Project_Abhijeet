
# ============================================================
# 3. DATASET SPLITTING
# ============================================================

# This script divides the paired satellite image dataset into
# three separate groups:
#
# 1. Training dataset   - 70%
# 2. Validation dataset - 15%
# 3. Testing dataset    - 15%
#
# Both HR (High Resolution) and LR (Low Resolution) images
# are kept together as matching pairs during the splitting.
#
# The random seed ensures that the same split can be reproduced
# whenever this script is executed with the same input files.


# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

from pathlib import Path
import random
import shutil


# ============================================================
# SET RANDOM SEED
# ============================================================

# Set a fixed random seed to make the dataset splitting
# reproducible.
#
# This means that the same random arrangement of image pairs
# will be generated every time the script is executed,
# provided the input files remain unchanged.

random.seed(42)


# ============================================================
# DEFINE DATASET DIRECTORIES
# ============================================================

# Main project directory containing the satellite dataset.
BASE_DIR = Path(
    r"C:\Users\ARF-G5\Desktop\Abhijeet"
)

# Directory containing High Resolution Sentinel-2 images.
HR_DIR = BASE_DIR / "Dataset" / "HR_Sentinel2"

# Directory containing Low Resolution Landsat-8 images.
LR_DIR = BASE_DIR / "Dataset" / "LR_Landsat8"

# Main output directory where the split dataset will be saved.
OUTPUT_DIR = BASE_DIR / "prepared_dataset"


# ============================================================
# FIND MATCHING HR AND LR IMAGE PAIRS
# ============================================================

# Create dictionaries containing the available HR image files.
#
# The filename without its extension (stem) is used as the key.
# This helps identify matching HR and LR images.

hr_files = {
    f.stem: f
    for f in HR_DIR.glob("*.tif")
}


# Create a dictionary containing the LR image files.
# The same filename-based matching approach is used.

lr_files = {
    f.stem: f
    for f in LR_DIR.glob("*.tif")
}


# Find filenames that exist in both HR and LR directories.
#
# Only images with matching filenames are included.
# Sorting makes the initial list order consistent.

common_names = sorted(
    set(hr_files) & set(lr_files)
)


# Create a list containing matching HR-LR image pairs.
#
# Each pair contains:
# First element  -> HR image
# Second element -> LR image

pairs = [
    (hr_files[name], lr_files[name])
    for name in common_names
]


# Display the total number of successfully matched image pairs.
print("Total paired images:", len(pairs))


# ============================================================
# SHUFFLE AND SPLIT THE DATASET
# ============================================================

# Randomly shuffle all image pairs before splitting.
#
# This prevents the dataset from being divided according
# to the original filename order.

random.shuffle(pairs)


# Calculate the total number of available image pairs.
total = len(pairs)


# Calculate the number of image pairs for each dataset split.
#
# Training   = 70% of the total dataset.
# Validation = 15% of the total dataset.
# Testing    = Remaining image pairs.

train_count = int(total * 0.70)

val_count = int(total * 0.15)


# Select the first 70% of the shuffled pairs for training.
train_pairs = pairs[:train_count]


# Select the next 15% of the pairs for validation.
val_pairs = pairs[
    train_count:train_count + val_count
]


# Assign all remaining pairs to the testing dataset.
test_pairs = pairs[
    train_count + val_count:
]


# Display the number of image pairs in each dataset split.
print("Training:", len(train_pairs))
print("Validation:", len(val_pairs))
print("Testing:", len(test_pairs))


# ============================================================
# CREATE TRAIN, VALIDATION AND TEST DATASET FOLDERS
# ============================================================

# Store all three dataset splits in a dictionary.
#
# Each key represents a dataset category, and its value
# contains the corresponding HR-LR image pairs.

splits = {
    "train": train_pairs,
    "val": val_pairs,
    "test": test_pairs
}


# ============================================================
# COPY IMAGE PAIRS INTO THEIR RESPECTIVE FOLDERS
# ============================================================

# Loop through each dataset category.
#
# split_name  -> Name of the dataset split.
# split_pairs -> List of matching HR-LR image pairs.

for split_name, split_pairs in splits.items():

    # Create the output directory for HR images.
    hr_out = OUTPUT_DIR / split_name / "HR"

    # Create the output directory for LR images.
    lr_out = OUTPUT_DIR / split_name / "LR"


    # Create the required directories.
    #
    # parents=True creates any missing parent folders.
    # exist_ok=True prevents an error if folders already exist.

    hr_out.mkdir(
        parents=True,
        exist_ok=True
    )

    lr_out.mkdir(
        parents=True,
        exist_ok=True
    )


    # Loop through all matching image pairs in the current split.
    for hr, lr in split_pairs:

        # Copy the High Resolution image into the HR folder.
        #
        # copy2() preserves file metadata such as modification time.

        shutil.copy2(
            hr,
            hr_out / hr.name
        )


        # Copy the corresponding Low Resolution image
        # into the LR folder.

        shutil.copy2(
            lr,
            lr_out / lr.name
        )


    # Display the number of image pairs copied successfully
    # into the current dataset category.

    print(
        f"{split_name}: {len(split_pairs)} pairs copied"
    )


# ============================================================
# DATASET SPLITTING COMPLETED
# ============================================================

print("\nDataset splitting completed successfully!")

print("Output directory:", OUTPUT_DIR)
