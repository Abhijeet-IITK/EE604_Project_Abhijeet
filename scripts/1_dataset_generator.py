
# ============================================================
# SATELLITE SUPER-RESOLUTION DATASET GENERATOR
# ============================================================
# This application generates paired Low Resolution (LR) and
# High Resolution (HR) satellite image patches.
#
# Input:
#   1. Landsat 8 satellite image (30 m resolution)
#   2. Sentinel-2 satellite image (10 m resolution)
#
# Output:
#   1. LR patches from Landsat 8
#   2. HR patches from Sentinel-2
#
# Scale Factor: 3x
# LR Patch Size: 256 x 256
# HR Patch Size: 768 x 768
# Output Format: GeoTIFF
# ============================================================


# -------------------- IMPORT LIBRARIES --------------------

import os
import re
import math
import queue
import threading
import tkinter as tk

from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
import rasterio

from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window


# ============================================================
# MAIN APPLICATION CLASS
# ============================================================

class SRDatasetGenerator:

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(self, root):

        # Store the main application window.
        self.root = root

        # Configure window title and dimensions.
        self.root.title("Satellite Super Resolution Dataset Generator")
        self.root.geometry("950x700")
        self.root.minsize(850, 600)
        self.root.resizable(True, True)

        # Queue transfers messages from the background thread
        # to the main GUI thread.
        self.events = queue.Queue()

        # Event used to request cancellation.
        self.cancel_event = threading.Event()

        # Track whether processing is running.
        self.running = False

        # Create the graphical interface.
        self.create_gui()

        # Check for new events every 100 milliseconds.
        self.root.after(100, self.process_events)

    # ========================================================
    # GUI DESIGN
    # ========================================================

    def create_gui(self):

        # Configure the GUI theme and button appearance.
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TButton",
            font=("Arial", 10)
        )

        # Configure the main heading style.
        style.configure(
            "Header.TLabel",
            font=("Arial", 19, "bold"),
            foreground="#17365d"
        )

        # Main container for all GUI components.
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # Application heading.
        ttk.Label(
            main,
            text="Satellite Super-Resolution Dataset Generator",
            style="Header.TLabel"
        ).pack(pady=8)

        # Application subtitle.
        ttk.Label(
            main,
            text="Landsat 8 -> Sentinel-2 | 3x Super Resolution | Abhijeet"
        ).pack(pady=5)

        # ----------------------------------------------------
        # INPUT FILE SECTION
        # ----------------------------------------------------

        input_frame = ttk.LabelFrame(
            main,
            text="Input Satellite Images",
            padding=12
        )
        input_frame.pack(fill="x", pady=8)

        # Variables to store selected file paths.
        self.ls8_path = tk.StringVar()
        self.s2_path = tk.StringVar()
        self.output_path = tk.StringVar(value="SR_Dataset")

        # Add file selection rows.
        self.add_file_row(
            input_frame,
            "Landsat 8:",
            self.ls8_path,
            self.browse_ls8
        )

        self.add_file_row(
            input_frame,
            "Sentinel-2:",
            self.s2_path,
            self.browse_s2
        )

        self.add_file_row(
            input_frame,
            "Output folder:",
            self.output_path,
            self.browse_output
        )

        # ----------------------------------------------------
        # DATASET CONFIGURATION SECTION
        # ----------------------------------------------------

        config_frame = ttk.LabelFrame(
            main,
            text="Dataset Configuration",
            padding=12
        )
        config_frame.pack(fill="x", pady=8)

        # Default satellite band selections.
        self.ls8_bands = tk.StringVar(value="1")
        self.s2_bands = tk.StringVar(value="1")

        # Default patch sizes and scale factor.
        self.lr_size = tk.StringVar(value="256")
        self.hr_size = tk.StringVar(value="768")
        self.scale = tk.StringVar(value="3")

        # Default resampling method.
        self.resampling = tk.StringVar(value="bilinear")

        # Store configuration labels and variables.
        fields = [
            ("Landsat band indices:", self.ls8_bands),
            ("Sentinel band indices:", self.s2_bands),
            ("LR patch size:", self.lr_size),
            ("HR patch size:", self.hr_size),
            ("Scale factor:", self.scale)
        ]

        # Create labels and entry boxes.
        for i, (label, variable) in enumerate(fields):

            ttk.Label(
                config_frame,
                text=label
            ).grid(
                row=i,
                column=0,
                sticky="w",
                padx=8,
                pady=5
            )

            ttk.Entry(
                config_frame,
                textvariable=variable,
                width=20
            ).grid(
                row=i,
                column=1,
                sticky="w",
                padx=8,
                pady=5
            )

        # Resampling method label.
        ttk.Label(
            config_frame,
            text="Resampling:"
        ).grid(
            row=5,
            column=0,
            sticky="w",
            padx=8,
            pady=5
        )

        # Dropdown menu for interpolation selection.
        ttk.Combobox(
            config_frame,
            textvariable=self.resampling,
            values=["bilinear", "cubic"],
            state="readonly",
            width=17
        ).grid(
            row=5,
            column=1,
            sticky="w",
            padx=8,
            pady=5
        )

        # Example of band selection format.
        ttk.Label(
            config_frame,
            text="Example bands: 1 or 1,2,3"
        ).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w",
            padx=8
        )

        # ----------------------------------------------------
        # OUTPUT INFORMATION SECTION
        # ----------------------------------------------------

        output_frame = ttk.LabelFrame(
            main,
            text="Output Information",
            padding=12
        )
        output_frame.pack(fill="x", pady=8)

        ttk.Label(
            output_frame,
            text="LR: 256 x 256 | HR: 768 x 768 | GeoTIFF | Append mode"
        ).pack(anchor="w")

        # Label showing the current application status.
        self.status_label = ttk.Label(
            output_frame,
            text="Ready",
            foreground="green"
        )
        self.status_label.pack(anchor="w", pady=5)

        # ----------------------------------------------------
        # PROGRESS BAR SECTION
        # ----------------------------------------------------

        progress_frame = ttk.LabelFrame(
            main,
            text="Generation Progress",
            padding=12
        )
        progress_frame.pack(fill="x", pady=8)

        # Progress bar showing patch generation progress.
        self.progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate"
        )
        self.progress.pack(fill="x", pady=8)

        # Label displaying the number of processed patches.
        self.progress_label = ttk.Label(
            progress_frame,
            text="0 / 0 patches"
        )
        self.progress_label.pack(anchor="w")

        # ----------------------------------------------------
        # PROCESSING LOG SECTION
        # ----------------------------------------------------

        log_frame = ttk.LabelFrame(
            main,
            text="Processing Log",
            padding=10
        )
        log_frame.pack(fill="both", expand=True, pady=8)

        # Scrolled text box for processing messages.
        self.log = scrolledtext.ScrolledText(
            log_frame,
            height=5,
            font=("Consolas", 9),
            state="disabled"
        )
        self.log.pack(fill="both", expand=True)

        # ----------------------------------------------------
        # CONTROL BUTTONS
        # ----------------------------------------------------

        button_frame = ttk.Frame(main)
        button_frame.pack(fill="x", pady=15)

        # Start processing button.
        self.start_button = ttk.Button(
            button_frame,
            text="START PROCESSING",
            command=self.start_generation
        )

        self.start_button.pack(
            side="left",
            padx=10,
            ipadx=20,
            ipady=8
        )

        # Cancel processing button.
        self.cancel_button = ttk.Button(
            button_frame,
            text="CANCEL",
            command=self.cancel_generation,
            state="disabled"
        )

        self.cancel_button.pack(
            side="left",
            padx=10,
            ipadx=15,
            ipady=8
        )

        # Button to clear the processing log.
        ttk.Button(
            button_frame,
            text="CLEAR LOG",
            command=self.clear_log
        ).pack(
            side="right",
            padx=10
        )

    # ========================================================
    # FILE BROWSER FUNCTIONS
    # ========================================================

    def add_file_row(self, parent, label, variable, command):

        # Create a frame for the file selection row.
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)

        # Display the file selection label.
        ttk.Label(
            frame,
            text=label,
            width=20
        ).pack(side="left")

        # Entry box displaying the selected path.
        ttk.Entry(
            frame,
            textvariable=variable
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=5
        )

        # Browse button.
        ttk.Button(
            frame,
            text="Browse",
            command=command
        ).pack(side="right")

    def browse_ls8(self):

        # Open a dialog to select the Landsat 8 image.
        path = filedialog.askopenfilename(
            filetypes=[
                ("GeoTIFF", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.ls8_path.set(path)

    def browse_s2(self):

        # Open a dialog to select the Sentinel-2 image.
        path = filedialog.askopenfilename(
            filetypes=[
                ("Satellite images", "*.jp2 *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.s2_path.set(path)

    def browse_output(self):

        # Open a dialog to select the output directory.
        path = filedialog.askdirectory()

        if path:
            self.output_path.set(path)

    # ========================================================
    # LOGGING FUNCTIONS
    # ========================================================

    def write_log(self, message):

        # Enable the text box before adding a message.
        self.log.configure(state="normal")

        # Insert the message at the end of the log.
        self.log.insert("end", message + "\n")

        # Automatically scroll to the latest message.
        self.log.see("end")

        # Disable editing to prevent accidental modifications.
        self.log.configure(state="disabled")

    def clear_log(self):

        # Enable the text box.
        self.log.configure(state="normal")

        # Remove all existing messages.
        self.log.delete("1.0", "end")

        # Disable editing again.
        self.log.configure(state="disabled")

    # ========================================================
    # BAND PARSING FUNCTION
    # ========================================================

    @staticmethod
    def parse_bands(value):

        # Convert comma-separated band numbers into integers.
        try:

            bands = [
                int(x.strip())
                for x in value.split(",")
                if x.strip()
            ]

            # Band numbering starts from 1.
            if not bands or min(bands) < 1:
                raise ValueError

            return bands

        except ValueError:

            # Display an error for invalid band input.
            raise ValueError(
                "Enter valid bands, e.g. 1 or 1,2,3"
            )

    # ========================================================
    # FIND NEXT PATCH NUMBER
    # ========================================================

    @staticmethod
    def get_next_patch_number(lr_dir, hr_dir):

        # Regular expression to identify patch filenames.
        pattern = re.compile(r"patch_(\d+)\.tif$")

        # Store existing LR and HR patch numbers.
        lr_numbers = set()
        hr_numbers = set()

        # Read patch numbers from the LR directory.
        for filename in os.listdir(lr_dir):

            match = pattern.fullmatch(filename)

            if match:
                lr_numbers.add(int(match.group(1)))

        # Read patch numbers from the HR directory.
        for filename in os.listdir(hr_dir):

            match = pattern.fullmatch(filename)

            if match:
                hr_numbers.add(int(match.group(1)))

        # Ensure both directories contain matching patch files.
        if lr_numbers != hr_numbers:

            raise RuntimeError(
                "LR and HR folders contain mismatched patch files."
            )

        # Return the next available patch number.
        return max(lr_numbers, default=-1) + 1

    # ========================================================
    # START DATASET GENERATION
    # ========================================================

    def start_generation(self):

        # Prevent starting another process if one is running.
        if self.running:
            return

        try:

            # Get selected file paths and output directory.
            ls8_file = self.ls8_path.get().strip()
            s2_file = self.s2_path.get().strip()
            output_dir = self.output_path.get().strip()

            # Check whether the Landsat file exists.
            if not os.path.isfile(ls8_file):
                raise ValueError("Select a valid Landsat file.")

            # Check whether the Sentinel file exists.
            if not os.path.isfile(s2_file):
                raise ValueError("Select a valid Sentinel file.")

            # Check whether the output folder is selected.
            if not output_dir:
                raise ValueError("Select an output folder.")

            # Read the selected band numbers.
            ls8_bands = self.parse_bands(
                self.ls8_bands.get()
            )

            s2_bands = self.parse_bands(
                self.s2_bands.get()
            )

            # Both images must have equal numbers of bands.
            if len(ls8_bands) != len(s2_bands):

                raise ValueError(
                    "Select equal numbers of bands."
                )

            # Convert patch sizes and scale factor into integers.
            lr_size = int(self.lr_size.get())
            hr_size = int(self.hr_size.get())
            scale = int(self.scale.get())

            # Ensure all values are positive.
            if min(lr_size, hr_size, scale) <= 0:
                raise ValueError("Sizes must be positive.")

            # Check the relationship between LR and HR sizes.
            if hr_size != lr_size * scale:

                raise ValueError(
                    "HR size must equal LR size x scale factor."
                )

            # Validate the selected resampling method.
            if self.resampling.get() not in ("bilinear", "cubic"):
                raise ValueError("Invalid resampling method.")

            # Store all configuration values in a dictionary.
            self.config = {
                "ls8_file": ls8_file,
                "s2_file": s2_file,
                "output_dir": output_dir,
                "ls8_bands": ls8_bands,
                "s2_bands": s2_bands,
                "lr_size": lr_size,
                "hr_size": hr_size,
                "scale": scale,
                "resampling": self.resampling.get()
            }

        except Exception as e:

            # Display configuration errors.
            messagebox.showerror(
                "Configuration error",
                str(e)
            )

            return

        # Reset the cancellation event.
        self.cancel_event.clear()

        # Mark processing as running.
        self.running = True

        # Disable Start and enable Cancel buttons.
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")

        # Reset the progress bar.
        self.progress["value"] = 0

        self.progress_label.configure(
            text="Starting..."
        )

        # Update the processing status.
        self.status_label.configure(
            text="Processing...",
            foreground="#b06a00"
        )

        self.write_log("Starting dataset generation...")

        # Create a background thread for dataset generation.
        # This prevents the GUI from freezing during processing.
        self.worker = threading.Thread(
            target=self.generate_dataset,
            daemon=True
        )

        # Start the background thread.
        self.worker.start()

    # ========================================================
    # CANCEL DATASET GENERATION
    # ========================================================

    def cancel_generation(self):

        # Set the event to request cancellation.
        self.cancel_event.set()

        # Update the application status.
        self.status_label.configure(
            text="Cancellation requested...",
            foreground="#b06a00"
        )

        self.write_log("Cancellation requested.")

        # Disable the Cancel button after the request.
        self.cancel_button.configure(
            state="disabled"
        )

    # ========================================================
    # GUI EVENT HANDLER
    # ========================================================

    def process_events(self):

        # Read messages sent by the background processing thread.
        try:

            while True:

                # Get the next available event from the queue.
                event = self.events.get_nowait()

                # Identify the type of event.
                event_type = event[0]

                # --------------------------------------------
                # DISPLAY LOG MESSAGES
                # --------------------------------------------

                if event_type == "log":

                    # Display the received message in the log.
                    self.write_log(event[1])

                # --------------------------------------------
                # INITIALIZE PROGRESS BAR
                # --------------------------------------------

                elif event_type == "init":

                    # Get the total number of patches.
                    total = event[1]

                    # Set the progress bar limits.
                    self.progress["maximum"] = total
                    self.progress["value"] = 0

                    # Display initial progress.
                    self.progress_label.configure(
                        text=f"0 / {total} patches"
                    )

                # --------------------------------------------
                # UPDATE PROCESSING PROGRESS
                # --------------------------------------------

                elif event_type == "progress":

                    # Extract progress information.
                    _, processed, total, generated, skipped = event

                    # Update the progress bar.
                    self.progress["value"] = processed

                    # Display processed, generated and skipped patches.
                    self.progress_label.configure(
                        text=(
                            f"{processed}/{total} | "
                            f"Generated: {generated} | "
                            f"Skipped: {skipped}"
                        )
                    )

                # --------------------------------------------
                # HANDLE PROCESSING COMPLETION
                # --------------------------------------------

                elif event_type == "finished":

                    # Extract final processing information.
                    _, status, generated, skipped, next_count = event

                    # Mark processing as stopped.
                    self.running = False

                    # Enable the Start button again.
                    self.start_button.configure(
                        state="normal"
                    )

                    # Disable the Cancel button.
                    self.cancel_button.configure(
                        state="disabled"
                    )

                    # Update the status label based on the result.
                    if status == "SUCCESS":

                        self.status_label.configure(
                            text="Dataset generation completed!",
                            foreground="green"
                        )

                    else:

                        self.status_label.configure(
                            text="Generation cancelled.",
                            foreground="#b06a00"
                        )

                    # ----------------------------------------
                    # DISPLAY FINAL PROCESSING INFORMATION
                    # ----------------------------------------

                    self.write_log("--------------------------------")

                    self.write_log(
                        f"Status: {status}"
                    )

                    self.write_log(
                        f"Generated patches: {generated}"
                    )

                    self.write_log(
                        f"Skipped patches: {skipped}"
                    )

                    self.write_log(
                        f"Next patch number: {next_count}"
                    )

                    self.write_log(
                        "Older patches preserved."
                    )

                    # ----------------------------------------
                    # SHOW COMPLETION MESSAGE
                    # ----------------------------------------

                    if status == "SUCCESS":

                        # Display success message.
                        messagebox.showinfo(
                            "Completed",
                            f"Dataset generation completed!\n\n"
                            f"Generated: {generated}\n"
                            f"Skipped: {skipped}"
                        )

                    else:

                        # Display cancellation message.
                        messagebox.showwarning(
                            "Cancelled",
                            f"Generation stopped.\n\n"
                            f"Generated: {generated}\n"
                            f"Skipped: {skipped}"
                        )

                # --------------------------------------------
                # HANDLE PROCESSING ERRORS
                # --------------------------------------------

                elif event_type == "error":

                    # Mark processing as stopped.
                    self.running = False

                    # Enable the Start button again.
                    self.start_button.configure(
                        state="normal"
                    )

                    # Disable the Cancel button.
                    self.cancel_button.configure(
                        state="disabled"
                    )

                    # Update the application status.
                    self.status_label.configure(
                        text="Error occurred.",
                        foreground="red"
                    )

                    # Display error details in the processing log.
                    self.write_log(
                        "ERROR: " + event[1]
                    )

                    # Show an error dialog.
                    messagebox.showerror(
                        "Processing error",
                        event[1]
                    )

        except queue.Empty:

            # No more events are currently available.
            pass

        # Check for new events every 100 milliseconds.
        self.root.after(
            100,
            self.process_events
        )


# ============================================================
# APPLICATION STARTUP
# ============================================================

# This condition ensures that the application starts
# only when this Python file is executed directly.

if __name__ == "__main__":

    # Create the main Tkinter window.
    root = tk.Tk()

    # Create an instance of the dataset generator.
    app = SRDatasetGenerator(root)

    # Start the Tkinter event loop.
    # This keeps the GUI running and responsive.
    root.mainloop()
