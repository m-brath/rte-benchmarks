#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTE Benchmarks Overview - Generate flux analysis plots.

This script processes RTE benchmark results and generates comprehensive visualization
plots showing radiative flux distributions and statistics across different atmospheric
setups (CKDMIP, RCE, RFMIP) and radiation types (LW, SW).

For each setup and radiation type, the script:
    1. Loads reference flux data from NetCDF files
    2. Computes mean and standard deviation of upwelling and downwelling fluxes
    3. Creates altitude-stratified histograms of flux distributions
    4. Generates a 2x3 subplot figure containing:
       - Mean upwelling/downwelling fluxes with altitude
       - Flux variability (standard deviation) with altitude
       - 2D histograms (flux vs. altitude) showing flux distribution patterns

Output plots are saved as PDF files in the plots/ directory.

Dependencies:
    - numpy, matplotlib
    - xarray (via rte_aux_functions)
    
Author: Manfred Brath
"""

import os
import pathlib
import numpy as np
import rte_aux_functions as raf
import matplotlib.pyplot as plt

# =============================================================================
# Configuration and Constants
# =============================================================================

# Get the directory where this script is located
script_dir = pathlib.Path(__file__).parent.resolve()

# Input and output directories
data_folder = script_dir.parent / "data/"
results_folder = script_dir.parent / "results/"
plot_folder = script_dir.parent / "plots/"

# Create plot directory if it doesn't exist
os.makedirs(plot_folder, exist_ok=True)

# Radiation types to process: LW (longwave) and SW (shortwave)
radtype = ["LW", "SW"]

# Reference data file naming conventions
prefix = "Reference_fluxes_Nf"
fid = ".nc"

# Number of frequency points in reference data
Nf_sw = 100001  # Shortwave
Nf_lw = 100000  # Longwave

# Color map for plots (RGBA format)
cmap = np.array(
    [
        [0.0, 0.44701, 0.74101, 1.0],      # Blue
        [0.85001, 0.32501, 0.09801, 1.0],  # Orange
        [0.92901, 0.69401, 0.12501, 1.0],  # Yellow
        [0.49401, 0.18401, 0.55601, 1.0],  # Purple
        [0.46601, 0.67401, 0.18801, 1.0],  # Green
        [0.30101, 0.74501, 0.93301, 1.0],  # Cyan
        [0.63501, 0.07801, 0.18401, 1.0],  # Red
    ]
)

# =============================================================================
# Load and Identify Data
# =============================================================================
# Scan data folder for all XML files and extract unique setup names
# (CKDMIP, RCE, RFMIP) from file names using the convention:
# rte-examples-arts_<type>-<setup>.xml

data_files = list(data_folder.glob("*.xml"))

# Extract setup identifiers from file names (4th component after split by '-')
setups = [str(df.stem).split("-")[3] for df in data_files]
setups = list(set(setups))  # Remove duplicates
setups.sort()

print(f"Found {len(setups)} setup(s): {setups}")

# =============================================================================
# %% loop through the data
# =============================================================================

for setup in setups:
    print(f"Processing setup: {setup}")

    for rad in radtype:

        tempfiolder = results_folder / f"{setup}/{rad}/"

        if rad == "SW":
            N = Nf_sw
            xlim = [0, 1400]
        else:
            N = Nf_lw
            xlim = [0, 500]

        data = raf.xr.open_dataset(tempfiolder / f"{prefix}{N}{fid}")

        # calculate mean and std over all variants
        mean_flux_up = data["flux_clearsky_up"].mean(dim="column").to_numpy()
        std_flux_up = data["flux_clearsky_up"].std(dim="column").to_numpy()

        mean_flux_down = data["flux_clearsky_down"].mean(dim="column").to_numpy()
        std_flux_down = data["flux_clearsky_down"].std(dim="column").to_numpy()

        # Now we create our altitude bins for the histogram using the mean altitude profile and the min and max altitude values
        min_altitude = data["altitude"].min().to_numpy()
        max_altitude = data["altitude"].max().to_numpy()

        # Mean altitude profile over all variants and columns
        mean_altitude = (
            data["altitude"].mean(dim="column").mean(dim="variant").to_numpy()
        )

        # Combine the mean altitude profile with the min and max altitude values to create a set of altitude bins for the histogram
        altitude_edges = np.sort(
            np.concatenate(([min_altitude], mean_altitude, [max_altitude]))
        )
        fluxes = np.linspace(xlim[0], xlim[1], 101)

        # calculate 2d histogram for the fluxes over all variants and columns
        hist_up, xedges_up, yedges_up = np.histogram2d(
            np.abs(data["flux_clearsky_up"].to_numpy().flatten()),
            data["altitude"].to_numpy().flatten(),
            bins=[fluxes, altitude_edges],
        )
        hist_down, xedges_down, yedges_down = np.histogram2d(
            np.abs(data["flux_clearsky_down"].to_numpy().flatten()),
            data["altitude"].to_numpy().flatten(),
            bins=[fluxes, altitude_edges],
        )

        # Hide bins with fewer than one count so they remain colorless in the plot
        hist_up = np.ma.masked_where(hist_up < 1, hist_up)
        hist_down = np.ma.masked_where(hist_down < 1, hist_down)

        # now plot the results
        fig, ax = plt.subplots(
            2, 3, figsize=(20.9 / 2.54, 20.9 / 2.54), sharex=True, sharey=True
        )

        # ===== Row 1: Upwelling Fluxes =====
        
        # [0, 0] Mean upwelling flux profile
        ax[0, 0].plot(
            abs(mean_flux_up.T),
            mean_altitude[:, np.newaxis],
            color=cmap[0, :],
            alpha=0.2,
        )
        ax[0, 0].set_title("Mean Flux Up")
        ax[0, 0].set_ylabel("Altitude")
        ax[0, 0].set_xlabel("Flux")
        ax[0, 0].grid(which="both", linestyle=":", linewidth=0.25)
        ax[0, 0].spines["right"].set_visible(False)
        ax[0, 0].spines["top"].set_visible(False)

        # [0, 1] Standard deviation of upwelling flux
        ax[0, 1].plot(
            std_flux_up.T, mean_altitude[:, np.newaxis], color=cmap[1, :], alpha=0.2
        )
        ax[0, 1].set_title("Std Flux Up")
        ax[0, 1].set_ylabel("Altitude")
        ax[0, 1].set_xlabel("Flux")
        ax[0, 1].grid(which="both", linestyle=":", linewidth=0.25)
        ax[0, 1].spines["right"].set_visible(False)
        ax[0, 1].spines["top"].set_visible(False)

        # [0, 2] 2D histogram of upwelling flux vs altitude
        # Shows the distribution of flux values at each altitude level
        pcm = ax[0, 2].pcolormesh(
            xedges_up,
            yedges_up,
            hist_up.T,
            cmap="inferno_r",
            shading="auto",
            rasterized=True,
        )
        ax[0, 2].set_title("2D Histogram Flux Up")
        ax[0, 2].set_ylabel("Altitude")
        ax[0, 2].set_xlabel("Flux")
        ax[0, 2].grid(which="both", linestyle=":", linewidth=0.25)
        ax[0, 2].spines["right"].set_visible(False)
        ax[0, 2].spines["top"].set_visible(False)
        cbar = fig.colorbar(pcm, ax=ax[0, 2], orientation="vertical")
        cbar.set_label("Counts")

        # ===== Row 2: Downwelling Fluxes =====

        # [1, 0] Mean downwelling flux profile
        ax[1, 0].plot(
            abs(mean_flux_down.T),
            mean_altitude[:, np.newaxis],
            color=cmap[2, :],
            alpha=0.2,
        )
        ax[1, 0].set_title("Mean Flux Down")
        ax[1, 0].set_ylabel("Altitude")
        ax[1, 0].set_xlabel("Flux")
        ax[1, 0].grid(which="both", linestyle=":", linewidth=0.25)
        ax[1, 0].spines["right"].set_visible(False)
        ax[1, 0].spines["top"].set_visible(False)

        # [1, 1] Standard deviation of downwelling flux
        ax[1, 1].plot(
            std_flux_down.T, mean_altitude[:, np.newaxis], color=cmap[3, :], alpha=0.2
        )
        ax[1, 1].set_title("Std Flux Down")
        ax[1, 1].set_ylabel("Altitude")
        ax[1, 1].set_xlabel("Flux")
        ax[1, 1].grid(which="both", linestyle=":", linewidth=0.25)
        ax[1, 1].spines["right"].set_visible(False)
        ax[1, 1].spines["top"].set_visible(False)

        # [1, 2] 2D histogram of downwelling flux vs altitude
        # Shows the distribution of flux values at each altitude level
        pcm = ax[1, 2].pcolormesh(
            xedges_down,
            yedges_down,
            hist_down.T,
            cmap="inferno_r",
            shading="auto",
            rasterized=True,
        )
        ax[1, 2].set_title("2D Histogram Flux Down")
        ax[1, 2].set_ylabel("Altitude")
        ax[1, 2].set_xlabel("Flux")
        ax[1, 2].grid(which="both", linestyle=":", linewidth=0.25)
        ax[1, 2].spines["right"].set_visible(False)
        ax[1, 2].spines["top"].set_visible(False)
        cbar = fig.colorbar(pcm, ax=ax[1, 2], orientation="vertical")
        cbar.set_label("Counts")
        ax[1, 2].set_xlim(xlim)

        # =====================================================================
        # Step 5: Finalize and save figure
        # =====================================================================

        # Add overall title indicating the atmospheric setup and radiation type
        fig.suptitle(f"Setup: {setup}, Radiation Type: {rad}", fontsize=16, y=1.0)

        # Save figure as high-resolution PDF
        output_file = plot_folder / f"overview_{setup}_{rad}_Nf{N}.pdf"
        plt.savefig(output_file, dpi=300)
        print(f"    Saved plot to: {output_file}")
        # plt.close(fig)

print("\nProcessing complete. All plots have been generated.")
