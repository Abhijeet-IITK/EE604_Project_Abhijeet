
# ============================================================
# PROJECT MODULE 2: SATELLITE DATASET INSPECTOR
# ============================================================
# Description:
# This program provides a graphical interface for inspecting
# paired Low Resolution (LR) Landsat-8 and High Resolution
# (HR) Sentinel-2 satellite image patches.
# Main functionalities:
# 1. Display LR and HR satellite image pairs side by side.
# 2. Visually inspect the quality of generated image patches.
# 3. Approve suitable image pairs using the KEEP option.
# 4. Reject unsuitable image pairs using the DISCARD option.
# 5. Navigate through patches using buttons or keyboard shortcuts.
# 6. Save approved image pairs into a separate dataset folder.
# 7. Maintain a CSV file containing previous review decisions.
# Input:
# Generated LR and HR GeoTIFF image patches.
# Output:
# Approved LR-HR image pairs and a CSV review history.

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import csv
import shutil
import tkinter as tk

# Used to display error messages in the GUI.
from tkinter import messagebox

# Matplotlib is used to display satellite images.
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# NumPy is used for image array operations.
import numpy as np

# Rasterio is used to read GeoTIFF satellite images.
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing Low Resolution Landsat-8 patches.
LR_DIR = "SR_Dataset/LR_Landsat8"

# Folder containing High Resolution Sentinel-2 patches.
HR_DIR = "SR_Dataset/HR_Sentinel2"

# Main folder where approved patches will be stored.
APPROVED_DIR = "SR_Approved_Dataset"

# Separate folders for approved LR and HR patches.
APPROVED_LR = os.path.join(
    APPROVED_DIR, "LR_Landsat8"
)

APPROVED_HR = os.path.join(
    APPROVED_DIR, "HR_Sentinel2"
)

# CSV file used to store review decisions.
REVIEW_FILE = os.path.join(
    APPROVED_DIR, "review_history.csv"
)

# Create the approved dataset folders if they do not exist.
os.makedirs(APPROVED_LR, exist_ok=True)
os.makedirs(APPROVED_HR, exist_ok=True)


# ============================================================
# LOAD PREVIOUS REVIEW HISTORY
# ============================================================

# Dictionary to store previously reviewed filenames and decisions.
# Example: {"patch_00001.tif": "KEEP"}
reviewed = {}

# Check whether a review history file already exists.
if os.path.exists(REVIEW_FILE):

    # Open the CSV file in read mode.
    with open(REVIEW_FILE, "r", newline="") as f:

        # Read the CSV data using column names.
        reader = csv.DictReader(f)

        # Store each filename and its review decision.
        for row in reader:

            reviewed[row["filename"]] = row["decision"]


# ============================================================
# FIND MATCHING LR AND HR PATCHES
# ============================================================

# Get all TIFF filenames from the LR folder.
lr_files = set(
    f for f in os.listdir(LR_DIR)
    if f.endswith(".tif")
)

# Get all TIFF filenames from the HR folder.
hr_files = set(
    f for f in os.listdir(HR_DIR)
    if f.endswith(".tif")
)

# Check whether both folders contain identical filenames.
# Every LR patch must have a matching HR patch.
if lr_files != hr_files:

    raise RuntimeError(
        "LR and HR folders do not contain identical filenames. "
        "Check your paired dataset."
    )

# Sort all matching patch filenames.
all_files = sorted(lr_files)

# Select only patches that have not been reviewed before.
# This prevents already reviewed patches from appearing again.
files = [
    f for f in all_files
    if f not in reviewed
]

# Keep track of the currently displayed patch.
index = 0

# ============================================================
# IMAGE DISPLAY FUNCTION
# ============================================================

def load_image(path):

    # Open the GeoTIFF image using Rasterio.
    with rasterio.open(path) as src:

        # Read all image bands and convert them to float32.
        img = src.read().astype(np.float32)

    # --------------------------------------------------------
    # CASE 1: SINGLE-BAND IMAGE
    # --------------------------------------------------------

    if img.shape[0] == 1:

        # Extract the only available band.
        band = img[0]

        # Identify valid pixels.
        valid = np.isfinite(band)

        if valid.any():

            # Calculate the 2nd and 98th percentiles.
            # This improves image contrast for visualization.
            low, high = np.percentile(
                band[valid], [2, 98]
            )

            # Avoid division or display problems if both values
            # are equal.
            if high <= low:
                high = low + 1

            # Return the image as a grayscale image.
            return band, "gray", low, high

        # Default display values if no valid pixels exist.
        return band, "gray", 0, 1

    # --------------------------------------------------------
    # CASE 2: TWO-BAND IMAGE
    # --------------------------------------------------------

    if img.shape[0] == 2:

        # Duplicate the second band to create three channels.
        # This allows the image to be displayed as RGB.
        img = np.concatenate(
            [img, img[1:2]],
            axis=0
        )

    # --------------------------------------------------------
    # CASE 3: THREE OR MORE BANDS
    # --------------------------------------------------------

    # Create an empty RGB image.
    rgb = np.zeros(
        (img.shape[1], img.shape[2], 3),
        dtype=np.float32
    )

    # Process the first three bands as RGB channels.
    for i in range(3):

        band = img[i]

        # Identify valid pixel values.
        valid = np.isfinite(band)

        if valid.any():

            # Calculate contrast limits using percentiles.
            low, high = np.percentile(
                band[valid], [2, 98]
            )

            if high > low:

                # Normalize pixel values between 0 and 1.
                # Clip removes values outside the display range.
                rgb[:, :, i] = np.clip(
                    (band - low) / (high - low),
                    0, 1
                )

    # Return the RGB image.
    return rgb, None, None, None


# ============================================================
# SAVE REVIEW DECISION
# ============================================================

def save_decision(filename, decision):

    # Prevent reviewing the same patch more than once.
    if filename in reviewed:
        raise RuntimeError("This patch was already reviewed.")

    # --------------------------------------------------------
    # KEEP DECISION
    # --------------------------------------------------------

    if decision == "KEEP":

        # Get the original LR and HR file paths.
        lr_src = os.path.join(LR_DIR, filename)
        hr_src = os.path.join(HR_DIR, filename)

        # Define destination paths in the approved dataset.
        lr_dst = os.path.join(APPROVED_LR, filename)
        hr_dst = os.path.join(APPROVED_HR, filename)

        # Prevent overwriting previously approved files.
        if os.path.exists(lr_dst) or os.path.exists(hr_dst):

            raise FileExistsError(
                f"Approved output already exists: {filename}"
            )

        # Copy the LR patch to the approved folder.
        shutil.copy2(lr_src, lr_dst)

        try:

            # Copy the matching HR patch.
            shutil.copy2(hr_src, hr_dst)

        except Exception:

            # If copying the HR patch fails,
            # remove the LR copy to avoid an incomplete pair.
            if os.path.exists(lr_dst):
                os.remove(lr_dst)

            raise

    # --------------------------------------------------------
    # SAVE DECISION IN CSV FILE
    # --------------------------------------------------------

    # Check whether this is a new review history file.
    new_file = not os.path.exists(REVIEW_FILE)

    # Open the CSV file in append mode.
    # Previous review records are preserved.
    with open(REVIEW_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        # Add column headers when creating the file for the first time.
        if new_file:
            writer.writerow(["filename", "decision"])

        # Save the filename and its decision.
        writer.writerow([filename, decision])

    # Update the review dictionary.
    reviewed[filename] = decision

# ============================================================
# CREATE MAIN TKINTER WINDOW
# ============================================================

root = tk.Tk()

# Set the application window title.
root.title("Satellite Super Resolution - Dataset Inspector")

# Set the window dimensions.
root.geometry("1450x850")

# Set a dark background color.
root.configure(bg="#202020")


# ============================================================
# TOP INFORMATION SECTION
# ============================================================

# Main application heading.
info_label = tk.Label(
    root,
    text="Satellite Image Pair Inspection",
    font=("Arial", 20, "bold"),
    fg="white",
    bg="#202020"
)

info_label.pack(pady=10)

# Label to display the current patch number.
counter_label = tk.Label(
    root,
    font=("Arial", 13),
    fg="white",
    bg="#202020"
)

counter_label.pack()

# Label to display the current filename.
filename_label = tk.Label(
    root,
    font=("Arial", 11),
    fg="#cccccc",
    bg="#202020"
)

filename_label.pack(pady=5)


# ============================================================
# CREATE MATPLOTLIB FIGURE
# ============================================================

# Create a Matplotlib figure for displaying two images.
fig = Figure(
    figsize=(13, 6),
    dpi=100,
    facecolor="#202020"
)

# Create two subplots.
# The first subplot displays Landsat-8 LR.
# The second subplot displays Sentinel-2 HR.
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Embed the Matplotlib figure inside the Tkinter window.
canvas = FigureCanvasTkAgg(
    fig,
    master=root
)

# Display the figure and allow it to expand with the window.
canvas.get_tk_widget().pack(
    fill=tk.BOTH,
    expand=True,
    padx=15,
    pady=10
)


# ============================================================
# STATUS LABEL
# ============================================================

# Displays the result of KEEP or DISCARD decisions.
status_label = tk.Label(
    root,
    text="",
    font=("Arial", 12, "bold"),
    fg="#00dd88",
    bg="#202020"
)

status_label.pack(pady=5)

# ============================================================
# DISPLAY CURRENT LR AND HR IMAGE PAIR
# ============================================================

def show_pair():

    # Clear both previous images before displaying new ones.
    ax1.clear()
    ax2.clear()

    # Check whether any pending patches are available.
    if not files:

        # Display a message when all patches have been reviewed.
        ax1.text(
            0.5, 0.5,
            "No pending patches",
            color="white",
            ha="center",
            va="center"
        )

        # Hide the second subplot.
        ax2.axis("off")

        # Update the information labels.
        counter_label.config(text="Review completed")

        filename_label.config(text="")

        # Refresh the Matplotlib canvas.
        canvas.draw()

        return

    # Get the filename of the currently selected patch.
    filename = files[index]

    # Create the full paths for both images.
    lr_path = os.path.join(LR_DIR, filename)
    hr_path = os.path.join(HR_DIR, filename)

    # Load the Landsat-8 LR image.
    lr_img, lr_cmap, lr_min, lr_max = load_image(lr_path)

    # Load the Sentinel-2 HR image.
    hr_img, hr_cmap, hr_min, hr_max = load_image(hr_path)

    # Display the LR image.
    ax1.imshow(
        lr_img,
        cmap=lr_cmap,
        vmin=lr_min,
        vmax=lr_max
    )

    # Add a title to the LR image.
    ax1.set_title(
        "LANDSAT 8 - LR (256 x 256)",
        color="white",
        fontsize=13
    )

    # Hide axis markings.
    ax1.axis("off")

    # Display the HR image.
    ax2.imshow(
        hr_img,
        cmap=hr_cmap,
        vmin=hr_min,
        vmax=hr_max
    )

    # Add a title to the HR image.
    ax2.set_title(
        "SENTINEL-2 - HR (768 x 768)",
        color="white",
        fontsize=13
    )

    # Hide axis markings.
    ax2.axis("off")

    # Adjust subplot spacing.
    fig.tight_layout()

    # Refresh the image display.
    canvas.draw()

    # Display the current patch number.
    counter_label.config(
        text=f"Patch {index + 1} / {len(files)}"
    )

    # Display the current filename.
    filename_label.config(
        text=f"Filename: {filename}"
    )


# ============================================================
# NAVIGATION FUNCTIONS
# ============================================================

def next_patch():

    # Access the global patch index.
    global index

    # Move to the next patch if one is available.
    if index < len(files) - 1:
        index += 1

    # Refresh the displayed images.
    show_pair()


def previous_patch():

    # Access the global patch index.
    global index

    # Move to the previous patch if possible.
    if index > 0:
        index -= 1

    # Refresh the displayed images.
    show_pair()


# ============================================================
# HANDLE KEEP AND DISCARD DECISIONS
# ============================================================

def decide(decision):

    # Access the global patch index.
    global index

    # Return if there are no patches left to review.
    if not files:
        return

    # Get the currently displayed filename.
    filename = files[index]

    try:

        # Save the selected review decision.
        save_decision(filename, decision)

    except Exception as e:

        # Display an error if saving the decision fails.
        messagebox.showerror(
            "Save error",
            str(e)
        )

        return

    # Display the decision in the status label.
    status_label.config(
        text=f"{filename} : {decision}"
    )

    # Remove the reviewed patch from the pending list.
    files.pop(index)

    # Adjust the index if the last patch was removed.
    if index >= len(files):
        index = max(0, len(files) - 1)

    # Display the next available patch.
    show_pair()

# ============================================================
# CREATE BUTTON FRAME
# ============================================================

# Frame to hold all navigation and decision buttons.
button_frame = tk.Frame(
    root,
    bg="#202020"
)

button_frame.pack(pady=15)


# ============================================================
# BUTTON CREATION FUNCTION
# ============================================================

def make_button(text, command, color):

    # Create and return a customized Tkinter button.
    return tk.Button(
        button_frame,
        text=text,
        command=command,
        font=("Arial", 12, "bold"),
        bg=color,
        fg="white",
        activeforeground="white",
        relief=tk.FLAT,
        width=15,
        height=2,
        cursor="hand2"
    )


# ============================================================
# ADD BUTTONS TO THE INTERFACE
# ============================================================

# Move to the previous patch.
make_button(
    "PREVIOUS",
    previous_patch,
    "#555555"
).grid(row=0, column=0, padx=8)

# Approve and copy the current LR-HR pair.
make_button(
    "KEEP",
    lambda: decide("KEEP"),
    "#168548"
).grid(row=0, column=1, padx=8)

# Reject the current patch pair.
make_button(
    "DISCARD",
    lambda: decide("DISCARD"),
    "#b52b32"
).grid(row=0, column=2, padx=8)

# Skip the current patch without saving a decision.
make_button(
    "SKIP",
    next_patch,
    "#a16a16"
).grid(row=0, column=3, padx=8)


# ============================================================
# KEYBOARD SHORTCUTS
# ============================================================

# Press K to keep the current patch.
root.bind("k", lambda event: decide("KEEP"))

# Press D to discard the current patch.
root.bind("d", lambda event: decide("DISCARD"))

# Press S to move to the next patch.
root.bind("s", lambda event: next_patch())

# Press the left arrow key to view the previous patch.
root.bind("<Left>", lambda event: previous_patch())

# Press the right arrow key to view the next patch.
root.bind("<Right>", lambda event: next_patch())


# ============================================================
# START APPLICATION
# ============================================================

# Display dataset statistics in the terminal.
print("Total patches:", len(all_files))
print("Already reviewed:", len(reviewed))
print("Pending patches:", len(files))

# Display the first pending image pair.
show_pair()

# Start the Tkinter event loop.
# This keeps the GUI active and responsive.
root.mainloop()
