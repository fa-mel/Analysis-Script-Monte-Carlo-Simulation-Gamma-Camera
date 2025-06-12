import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ==============================================================================
# 1. MAIN CONFIGURATION (TO BE EDITED BY USER)
# ==============================================================================

SIMULATION_FILES_TO_ANALYZE = [
    # EXAMPLE - REPLACE WITH YOUR ACTUAL FILES AND PATHS
    {"filename": "geom_A_5_2.hsb", "collimator": "A", "thickness": 2, "distance": 5},
    {"filename": "geom_A_10_2.hsb", "collimator": "A", "thickness": 2, "distance": 10},
    {"filename": "geom_A_15_2.hsb", "collimator": "A", "thickness": 2, "distance": 15},
    {"filename": "geom_A_5_10.hsb", "collimator": "A", "thickness": 10, "distance": 5},
    {"filename": "geom_A_10_10.hsb", "collimator": "A", "thickness": 10, "distance": 10},
    {"filename": "geom_A_15_10.hsb", "collimator": "A", "thickness": 10, "distance": 15},
    {"filename": "geom_B_5_2.hsb", "collimator": "B", "thickness": 2, "distance": 5},
    {"filename": "geom_B_10_2.hsb", "collimator": "B", "thickness": 2, "distance": 10},
    {"filename": "geom_B_15_2.hsb", "collimator": "B", "thickness": 2, "distance": 15},
    {"filename": "geom_B_5_10.hsb", "collimator": "B", "thickness": 10, "distance": 5},
    {"filename": "geom_B_10_10.hsb", "collimator": "B", "thickness": 10, "distance": 10},
    {"filename": "geom_B_15_10.hsb", "collimator": "B", "thickness": 10, "distance": 15},
    {"filename": "geom_C_5_2.hsb", "collimator": "C", "thickness": 2, "distance": 5},
    {"filename": "geom_C_10_2.hsb", "collimator": "C", "thickness": 2, "distance": 10},
    {"filename": "geom_C_15_2.hsb", "collimator": "C", "thickness": 2, "distance": 15},
    {"filename": "geom_C_5_10.hsb", "collimator": "C", "thickness": 10, "distance": 5},
    {"filename": "geom_C_10_10.hsb", "collimator": "C", "thickness": 10, "distance": 10},
    {"filename": "geom_C_15_10.hsb", "collimator": "C", "thickness": 10, "distance": 15},
]

# Total number of photons emitted by the source in the simulation.
TOTAL_EMITTED_PHOTONS = 10000000

# Energy windowing to isolate the photopeak.
USE_ENERGY_WINDOW = True
LOWER_ENERGY_BOUND = 0.119  # MeV
UPPER_ENERGY_BOUND = 0.161  # MeV

# --- Output Control Flags ---
SHOW_INDIVIDUAL_PLOTS = True
GENERATE_LATEX_OUTPUT = True
SHOW_PSF_HEATMAP = True
HEATMAP_BINS = 100
CALCULATE_FIGURE_OF_MERIT = True


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================


def gaussian_func(x, a, mu, sigma):
    """A standard 1D Gaussian function for curve fitting."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def calculate_fwhm_and_plot(
    data, data_label, ax, color_hist="skyblue", color_fit="orangered"
):
    """Calculates the FWHM of a 1D distribution and optionally plots the fit."""
    if data.size < 10:  # Not enough data for a meaningful fit
        if SHOW_INDIVIDUAL_PLOTS and ax:
            ax.text(0.5, 0.5, "Not enough data to fit", ha="center", va="center")
        return None, None, None, None

    counts, bin_edges = np.histogram(data, bins="auto", density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guesses for the fit parameters
    mu_initial, sigma_initial = np.mean(data), np.std(data)
    if sigma_initial == 0:
        sigma_initial = 1e-6  # Avoid division by zero
    a_initial = np.max(counts) if counts.size > 0 else 1.0

    try:
        params, pcov = curve_fit(
            gaussian_func,
            bin_centers,
            counts,
            p0=[a_initial, mu_initial, sigma_initial],
            maxfev=5000,
        )
        a_fit, mu_fit, sigma_fit = params
        sigma_fit = abs(sigma_fit)  # Sigma must be positive

        errors = np.sqrt(np.diag(pcov))
        std_err_mu, std_err_sigma = errors[1], errors[2]

        k_fwhm = 2 * np.sqrt(2 * np.log(2))
        fwhm = k_fwhm * sigma_fit
        delta_fwhm = k_fwhm * std_err_sigma

        if SHOW_INDIVIDUAL_PLOTS and ax:
            x_fit_plot = np.linspace(bin_edges[0], bin_edges[-1], 200)
            pdf_fitted_plot = gaussian_func(x_fit_plot, a_fit, mu_fit, sigma_fit)
            ax.hist(
                data,
                bins="auto",
                density=True,
                edgecolor="grey",
                alpha=0.7,
                label="Histogram Data",
                color=color_hist,
            )
            fit_label = (
                f"Gaussian Fit:\n"
                f"  $\mu={mu_fit:.3f} \pm {std_err_mu:.3f}$ cm\n"
                f"  $\sigma={sigma_fit:.3f} \pm {std_err_sigma:.3f}$ cm\n"
                f"  FWHM={fwhm:.3f} $\pm$ {delta_fwhm:.3f} cm"
            )
            ax.plot(
                x_fit_plot, pdf_fitted_plot, color=color_fit, linewidth=2.5, label=fit_label
            )
            ax.set_title(f"PSF Profile: {data_label}")
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle="--")

        return mu_fit, sigma_fit, fwhm, delta_fwhm

    except Exception as e:
        if SHOW_INDIVIDUAL_PLOTS and ax:
            print(f"  > Warning: Gaussian fit failed for {data_label}. Error: {e}")
            ax.hist(data, bins="auto", density=True, label="Data (Fit Failed)", color=color_hist)
            ax.set_title(f"PSF Profile: {data_label} (Fit Failed)")
        return None, None, None, None


def plot_energy_spectrum(ntuple, config_str, base_filename, output_dir):
    """Plots the energy spectrum of detected photons."""
    if not SHOW_INDIVIDUAL_PLOTS:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(
        ntuple[:, 8], bins=150, range=(0, 0.2), edgecolor="grey", alpha=0.75
    )  # ntuple[:, 8] is 'eslab'
    if USE_ENERGY_WINDOW:
        plt.axvline(
            LOWER_ENERGY_BOUND,
            color="red",
            ls="--",
            lw=1.5,
            label=f"Lower Bound ({LOWER_ENERGY_BOUND:.3f} MeV)",
        )
        plt.axvline(
            UPPER_ENERGY_BOUND,
            color="red",
            ls="--",
            lw=1.5,
            label=f"Upper Bound ({UPPER_ENERGY_BOUND:.3f} MeV)",
        )
        plt.legend()
    plt.title(f"Energy Spectrum\n{config_str}")
    plt.xlabel("Deposited Energy (MeV)")
    plt.ylabel("Counts")
    plt.grid(True, linestyle="--")
    plt.savefig(os.path.join(output_dir, f"energy_spectrum_{base_filename}.pdf"))
    plt.show()


def plot_psf_with_heatmap(
    x_data, y_data, x_label, y_label, title_str, config_str, filename, output_dir
):
    """Generates a 2D scatter plot and heatmap of the PSF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(f"{title_str}\n{config_str}", fontsize=16)

    ax1.scatter(x_data, y_data, marker="o", alpha=0.3, s=15, c="cornflowerblue")
    ax1.set_title("2D Scatter Plot")
    ax1.set_xlabel(f"{x_label} (cm)")
    ax1.set_ylabel(f"{y_label} (cm)")
    ax1.grid(True, linestyle=":")
    ax1.axis("equal")

    _, _, _, im = ax2.hist2d(x_data, y_data, bins=HEATMAP_BINS, cmap="viridis")
    ax2.set_title("2D Heatmap")
    ax2.set_xlabel(f"{x_label} (cm)")
    ax2.set_ylabel(f"{y_label} (cm)")
    ax2.axis("equal")
    fig.colorbar(im, ax=ax2, label="Number of Events")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def analyze_simulation_file(
    file_info, total_emitted, use_window, lower_e, upper_e, output_dir
):
    """Main analysis function for a single simulation file."""
    filename = file_info["filename"]
    print(f"\n{'='*30}\n--- Starting Analysis for file: {filename} ---")
    config_str = f"Coll: {file_info['collimator']}, Thick: {file_info['thickness']}mm, Dist: {file_info['distance']}cm"
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    try:
        with open(filename, "rb") as fid:
            A = np.fromfile(fid, dtype=np.float32)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File '{filename}' not found. Skipping.")
        return None

    ntuple_label = [
        "source_num", "history", "xorig", "yorig", "zorig", "a1orig", "a2orig", "a3orig",
        "eslab", "xslab", "yslab", "zslab",
        "eslabesr", "xslabesr", "yslabesr",
    ]
    num_vars = len(ntuple_label)

    if A.size == 0 or A.size % num_vars != 0:
        print(f"CRITICAL ERROR: File '{filename}' is empty or corrupted. Skipping.")
        return None

    # Reshape array using Fortran-style ordering
    ntuple = A.reshape((num_vars, A.size // num_vars), order="F").T
    print(f"Data loaded from '{filename}'. Ntuple dimensions: {ntuple.shape}")

    plot_energy_spectrum(ntuple, config_str, base_filename, output_dir)

    # --- Sensitivity Analysis ---
    if use_window:
        detected_mask = (ntuple[:, 8] > lower_e) & (ntuple[:, 8] < upper_e)
    else:
        detected_mask = ntuple[:, 8] > 0

    num_detected = np.sum(detected_mask)
    sensitivity = num_detected / total_emitted
    delta_sens = np.sqrt(num_detected) / total_emitted if num_detected > 0 else 0.0
    print(f"Detected photons: {num_detected}")
    print(f"Calculated Sensitivity: ({sensitivity:.4e} +/- {delta_sens:.4e})")

    results = {
        k: None
        for k in [
            "fwhm_x_true", "dfwhm_x_true", "fwhm_y_true", "dfwhm_y_true",
            "fwhm_x_esr", "dfwhm_x_esr", "fwhm_y_esr", "dfwhm_y_esr",
            "fom",
        ]
    }
    results.update({"sensitivity": sensitivity, "delta_sensitivity": delta_sens})

    # --- Spatial Resolution (FWHM) Analysis ---
    ntuple_det = ntuple[detected_mask, :]
    if ntuple_det.shape[0] < 10:
        print("Not enough detected photons for spatial resolution analysis.")
        results.update(file_info)
        return results

    # Resolution from collimator geometry (true interaction coordinates)
    print("\n  A. FWHM - Neglecting Scintillator Blurring (using xslab, yslab)")
    xslab, yslab = ntuple_det[:, 9], ntuple_det[:, 10]
    axs_proj = [None, None]
    if SHOW_INDIVIDUAL_PLOTS:
        if SHOW_PSF_HEATMAP:
            plot_psf_with_heatmap(
                xslab,
                yslab,
                ntuple_label[9],
                ntuple_label[10],
                "2D PSF (Neglecting Blurring)",
                config_str,
                f"psf2d_true_{base_filename}.pdf",
                output_dir,
            )
        fig_proj, axs_proj = plt.subplots(1, 2, figsize=(17, 6.5))
        fig_proj.suptitle(f"1D Projections - Neglecting Blurring\n{config_str}")

    _, _, results["fwhm_x_true"], results["dfwhm_x_true"] = calculate_fwhm_and_plot(
        xslab, ntuple_label[9], axs_proj[0]
    )
    _, _, results["fwhm_y_true"], results["dfwhm_y_true"] = calculate_fwhm_and_plot(
        yslab, ntuple_label[10], axs_proj[1]
    )
    if SHOW_INDIVIDUAL_PLOTS:
        fig_proj.savefig(os.path.join(output_dir, f"fwhm_proj_true_{base_filename}.pdf"))
        plt.show()

    # Resolution including scintillator blurring (reconstructed coordinates)
    print("\n  B. FWHM - Considering Scintillator Blurring (using xslabesr, yslabesr)")
    xslabesr, yslabesr = ntuple_det[:, 13], ntuple_det[:, 14]
    axs_proj_esr = [None, None]
    if SHOW_INDIVIDUAL_PLOTS:
        if SHOW_PSF_HEATMAP:
            plot_psf_with_heatmap(
                xslabesr,
                yslabesr,
                ntuple_label[13],
                ntuple_label[14],
                "2D PSF (Considering Blurring)",
                config_str,
                f"psf2d_esr_{base_filename}.pdf",
                output_dir,
            )
        fig_proj_esr, axs_proj_esr = plt.subplots(1, 2, figsize=(17, 6.5))
        fig_proj_esr.suptitle(f"1D Projections - Considering Blurring\n{config_str}")

    _, _, results["fwhm_x_esr"], results["dfwhm_x_esr"] = calculate_fwhm_and_plot(
        xslabesr, ntuple_label[13], axs_proj_esr[0], "lightcoral", "darkred"
    )
    _, _, results["fwhm_y_esr"], results["dfwhm_y_esr"] = calculate_fwhm_and_plot(
        yslabesr, ntuple_label[14], axs_proj_esr[1], "lightcoral", "darkred"
    )
    if SHOW_INDIVIDUAL_PLOTS:
        fig_proj_esr.savefig(os.path.join(output_dir, f"fwhm_proj_esr_{base_filename}.pdf"))
        plt.show()

    results.update(file_info)
    return results


def get_mean_fwhm(res, type="esr"):
    """Calculates the mean of the FWHM values in X and Y."""
    fwhm_x = res.get(f"fwhm_x_{type}")
    fwhm_y = res.get(f"fwhm_y_{type}")
    valid_fwhms = [f for f in [fwhm_x, fwhm_y] if f is not None]
    return np.mean(valid_fwhms) if valid_fwhms else None


def plot_summary_graphs(all_results, output_dir):
    """Generates summary plots combining results from all analyzed files."""
    print("\n--- Generating and Saving Summary Graphs ---")
    plt.style.use("seaborn-v0_8-whitegrid")

    grouped_data = {}
    for res in all_results:
        key = (res["collimator"], res["thickness"])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(res)
    for key in grouped_data:
        grouped_data[key].sort(key=lambda x: x["distance"])

    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_data)))
    markers = ["o", "s", "^", "D", "v", "p"]
    style_map = {
        key: (colors[i], markers[i % len(markers)])
        for i, key in enumerate(sorted(grouped_data.keys()))
    }

    # --- Sensitivity Summary Plot ---
    fig_sens = plt.figure(figsize=(10, 7))
    for key, results_list in sorted(grouped_data.items()):
        col, thick = key
        color, marker = style_map[key]
        distances = [r["distance"] for r in results_list]
        sensitivities = [r["sensitivity"] * 100 for r in results_list]
        errors = [r["delta_sensitivity"] * 100 for r in results_list]
        plt.errorbar(
            distances,
            sensitivities,
            yerr=errors,
            fmt="-",
            marker=marker,
            capsize=5,
            label=f"Coll. {col}, {thick}mm",
            color=color,
        )
    plt.xlabel("Source-Collimator Distance (cm)")
    plt.ylabel("Sensitivity (%)")
    plt.title("Sensitivity vs. Distance for Various Configurations")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    fig_sens.savefig(os.path.join(output_dir, "summary_sensitivity.pdf"))
    plt.show()

    # --- FWHM Summary Plot ---
    fig_fwhm, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for key, results_list in sorted(grouped_data.items()):
        col, thick = key
        color, marker = style_map[key]
        distances = [r["distance"] for r in results_list]

        fwhm_true = [get_mean_fwhm(r, "true") for r in results_list]
        dfwhm_true = [
            np.mean([d for d in [r.get("dfwhm_x_true"), r.get("dfwhm_y_true")] if d is not None])
            for r in results_list
        ]
        fwhm_esr = [get_mean_fwhm(r, "esr") for r in results_list]
        dfwhm_esr = [
            np.mean([d for d in [r.get("dfwhm_x_esr"), r.get("dfwhm_y_esr")] if d is not None])
            for r in results_list
        ]

        dist_true, fwhm_true_plot, dfwhm_true_plot = zip(
            *[(d, f, e) for d, f, e in zip(distances, fwhm_true, dfwhm_true) if f is not None]
        )
        dist_esr, fwhm_esr_plot, dfwhm_esr_plot = zip(
            *[(d, f, e) for d, f, e in zip(distances, fwhm_esr, dfwhm_esr) if f is not None]
        )

        ax1.errorbar(
            dist_true,
            fwhm_true_plot,
            yerr=dfwhm_true_plot,
            fmt="-",
            marker=marker,
            capsize=5,
            label=f"Coll. {col}, {thick}mm",
            color=color,
        )
        ax2.errorbar(
            dist_esr,
            fwhm_esr_plot,
            yerr=dfwhm_esr_plot,
            fmt="-",
            marker=marker,
            capsize=5,
            label=f"Coll. {col}, {thick}mm",
            color=color,
        )

    ax1.set_title("FWHM vs. Distance (Neglecting Blurring)")
    ax2.set_title("FWHM vs. Distance (Considering Blurring)")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Source-Collimator Distance (cm)")
        ax.legend()
    ax1.set_ylabel("FWHM (cm)")
    fig_fwhm.suptitle("Spatial Resolution Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_fwhm.savefig(os.path.join(output_dir, "summary_fwhm.pdf"))
    plt.show()


# ==============================================================================
# 3. FIGURE OF MERIT PLOTTING
# ==============================================================================


def plot_fom_summary(all_results, output_dir):
    """Generates the Figure of Merit (FoM) trade-off plot."""
    print("\n--- Generating Figure of Merit Summary Graph ---")
    fig_fom, ax = plt.subplots(figsize=(12, 9))

    color_map = {"A": plt.cm.Blues, "B": plt.cm.Greens, "C": plt.cm.Reds}
    thickness_to_shade = {2: 0.5, 10: 0.9}
    distance_to_marker = {5: "o", 10: "s", 15: "^"}

    for res in all_results:
        fwhm = get_mean_fwhm(res, "esr")
        sens = res["sensitivity"]
        if fwhm and sens:
            collimator, thickness, distance = (
                res["collimator"],
                res["thickness"],
                res["distance"],
            )
            cmap, shade, marker = (
                color_map.get(collimator, plt.cm.Greys),
                thickness_to_shade.get(thickness, 0.7),
                distance_to_marker.get(distance, "x"),
            )
            point_color = cmap(shade)
            ax.scatter(
                fwhm,
                sens,
                s=150,
                alpha=0.9,
                color=point_color,
                marker=marker,
                edgecolors="black",
                linewidth=0.7,
            )

    # Highlight the best configuration based on FoM
    best_res = max(all_results, key=lambda r: r.get("fom", 0))
    best_fwhm, best_sens = get_mean_fwhm(best_res, "esr"), best_res.get("sensitivity")
    if best_fwhm and best_sens:
        ax.scatter(
            best_fwhm,
            best_sens,
            s=500,
            color="gold",
            marker="*",
            edgecolors="black",
            linewidth=1.5,
            zorder=10,
            label="Best FoM",
        )
        best_label = (
            f"Best: C{best_res['collimator']} T{best_res['thickness']} D{best_res['distance']}"
        )
        ax.annotate(
            best_label,
            xy=(best_fwhm, best_sens),
            xytext=(best_fwhm, best_sens - 0.00015),
            ha="center",
            va="top",
            arrowprops=dict(
                facecolor="black", shrink=0.05, width=1, headwidth=8, connectionstyle="arc3"
            ),
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.7),
        )

    ax.set_xlabel("Spatial Resolution FWHM (cm)", fontsize=12)
    ax.set_ylabel("Sensitivity (counts/emitted)", fontsize=12)
    ax.set_title("Performance Trade-off: Sensitivity vs. Resolution", fontsize=16)
    ax.grid(True, which="both", linestyle="--")

    # Custom Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map["A"](0.7), edgecolor="black", label="Collimator A"),
        Patch(facecolor=color_map["B"](0.7), edgecolor="black", label="Collimator B"),
        Patch(facecolor=color_map["C"](0.7), edgecolor="black", label="Collimator C"),
        Line2D([0], [0], color="w", label=""),  # Spacer
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=10, label="Distance 5 cm"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", markersize=10, label="Distance 10 cm"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey", markersize=10, label="Distance 15 cm"),
        Line2D([0], [0], color="w", label=""),  # Spacer
        Patch(facecolor="lightgrey", edgecolor="black", label="Thickness 2 mm (Lighter)"),
        Patch(facecolor="dimgrey", edgecolor="black", label="Thickness 10 mm (Darker)"),
        Line2D([0], [0], color="w", label=""),  # Spacer
        Line2D([0], [0], marker="*", color="gold", label="Best FoM", markersize=15, linestyle="None", markeredgecolor="black"),
    ]
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        title="Legend",
    )

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    fig_fom.savefig(os.path.join(output_dir, "summary_figure_of_merit.pdf"))
    plt.show()


def generate_latex_output(all_results, output_dir):
    """Generates LaTeX code for tables and figures."""
    print(f"\n\n{'='*30}\n--- Generating LaTeX Code Output ---")

    def format_value_with_uncertainty(
        value, uncertainty, value_fmt=".3f", uncertainty_fmt=".3f", threshold=1e-3
    ):
        if value is None:
            return "N/A"
        if uncertainty is not None and uncertainty < threshold and uncertainty != 0:
            uncertainty_fmt = ".1e"
        uncertainty_str = (
            f" \\pm {format(uncertainty, uncertainty_fmt)}" if uncertainty is not None else ""
        )
        return f"${format(value, value_fmt)}{uncertainty_str}$"

    results_by_collimator = {}
    for res in all_results:
        coll = res["collimator"]
        if coll not in results_by_collimator:
            results_by_collimator[coll] = []
        results_by_collimator[coll].append(res)

    for coll, results_list in sorted(results_by_collimator.items()):
        print(f"\n--- LaTeX Code for Collimator {coll} Table ---")
        print(
            r"\begin{table}[h!]" + "\n"
            r"  \centering" + "\n"
            f"  \\caption{{Summary of Results for Collimator {coll}}}" + "\n"
            f"  \\label{{tab:summary_coll_{coll.lower()}}}" + "\n"
            r"  \begin{tabular}{|c|c|c|c|c|}" + "\n"
            r"    \hline" + "\n"
            r"    \textbf{Thickness} & \textbf{Distance} & \textbf{Sensitivity (\%)} & \textbf{FWHM (no blur) (cm)} & \textbf{FWHM (blur) (cm)} \\" + "\n"
            r"    (mm) & (cm) & & & \\" + "\n"
            r"    \hline"
        )
        results_list.sort(key=lambda x: (x["thickness"], x["distance"]))
        for res in results_list:
            sens_str = format_value_with_uncertainty(
                res["sensitivity"] * 100, res["delta_sensitivity"] * 100
            )
            fwhm_true_str = format_value_with_uncertainty(
                get_mean_fwhm(res, "true"),
                np.mean([d for d in [res.get("dfwhm_x_true"), res.get("dfwhm_y_true")] if d is not None]),
            )
            fwhm_esr_str = format_value_with_uncertainty(
                get_mean_fwhm(res, "esr"),
                np.mean([d for d in [res.get("dfwhm_x_esr"), res.get("dfwhm_y_esr")] if d is not None]),
            )
            print(f"    {res['thickness']} & {res['distance']} & {sens_str} & {fwhm_true_str} & {fwhm_esr_str} \\\\")
        print(r"    \hline" + "\n" + r"  \end{tabular}" + "\n" + r"\end{table}")

    print("\n--- LaTeX Code for Including Summary Graphs ---")
    print(
        r"\begin{figure}[h!]" + "\n"
        r"  \centering" + "\n"
        f"  \\includegraphics[width=0.8\\textwidth]{{{output_dir}/summary_sensitivity.pdf}}\n"
        r"  \caption{Summary graph of Sensitivity as a function of distance.}" + "\n"
        r"  \label{fig:summary_sensitivity}" + "\n"
        r"\end{figure}" + "\n\n"
        r"\begin{figure}[h!]" + "\n"
        r"  \centering" + "\n"
        f"  \\includegraphics[width=\\textwidth]{{{output_dir}/summary_fwhm.pdf}}\n"
        r"  \caption{Summary graph of FWHM as a function of distance, comparing the case without blurring (left) and with blurring (right).}" + "\n"
        r"  \label{fig:summary_fwhm}" + "\n"
        r"\end{figure}"
    )

    if CALCULATE_FIGURE_OF_MERIT:
        print("\n--- LaTeX Code for Figure of Merit ---")
        print(
            r"\begin{figure}[h!]" + "\n"
            r"  \centering" + "\n"
            f"  \\includegraphics[width=0.8\\textwidth]{{{output_dir}/summary_figure_of_merit.pdf}}\n"
            r"  \caption{Performance trade-off between sensitivity and spatial resolution. The best configuration according to the Figure of Merit (FoM) is highlighted.}" + "\n"
            r"  \label{fig:summary_fom}" + "\n"
            r"\end{figure}" + "\n\n"
            r"\begin{table}[h!]" + "\n"
            r"  \centering" + "\n"
            r"  \caption{Ranking of configurations based on the Figure of Merit (FoM = Sensitivity / FWHM).}" + "\n"
            r"  \label{tab:fom_ranking}" + "\n"
            r"  \begin{tabular}{|c|c|c|c|c|}" + "\n"
            r"    \hline" + "\n"
            r"    \textbf{Rank} & \textbf{Configuration} & \textbf{FoM} & \textbf{Sensitivity (\%)} & \textbf{Mean FWHM (cm)} \\" + "\n"
            r"    \hline"
        )

        for i, res in enumerate(
            sorted(all_results, key=lambda x: x.get("fom", 0), reverse=True)
        ):
            if res.get("fom") is not None:
                config_str = (
                    f"C-{res['collimator']}, T-{res['thickness']}, D-{res['distance']}"
                )
                fom_str = f"${res['fom']:.2e}$"
                sens_str = format_value_with_uncertainty(
                    res["sensitivity"] * 100, res["delta_sensitivity"] * 100, ".4f"
                )
                fwhm_str = format_value_with_uncertainty(
                    get_mean_fwhm(res, "esr"),
                    np.mean([d for d in [res.get("dfwhm_x_esr"), res.get("dfwhm_y_esr")] if d is not None]),
                    ".4f",
                )
                print(f"    {i+1} & {config_str} & {fom_str} & {sens_str} & {fwhm_str} \\\\")
        print(r"    \hline" + "\n" + r"  \end{tabular}" + "\n" + r"\end{table}")


# ==============================================================================
# 4. MAIN SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    output_dir = "analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    all_results = [
        result
        for file_info in SIMULATION_FILES_TO_ANALYZE
        if (
            result := analyze_simulation_file(
                file_info,
                TOTAL_EMITTED_PHOTONS,
                USE_ENERGY_WINDOW,
                LOWER_ENERGY_BOUND,
                UPPER_ENERGY_BOUND,
                output_dir,
            )
        )
    ]

    if not all_results:
        print("No valid results were collected. Cannot generate summary outputs.")
    else:
        # Calculate Figure of Merit (FoM = Sensitivity / FWHM)
        if CALCULATE_FIGURE_OF_MERIT:
            for res in all_results:
                fwhm = get_mean_fwhm(res, "esr")
                if fwhm and res["sensitivity"]:
                    res["fom"] = res["sensitivity"] / (fwhm)

        # Print a formatted text summary table to the console
        print("\n" + "=" * 80 + "\n--- TEXT SUMMARY TABLE OF RESULTS ---")
        header = (
            f"{'File':<20s} | {'Coll':<4} | {'Thick':<5} | {'Dist':<4} | {'Sens(%)':<8} | "
            f"{'d_Sens(%)':<11} | {'FWHM_x_true':<13} | {'dFWHM_x_true':<12} | "
            f"{'FWHM_y_true':<13} | {'dFWHM_y_true':<12} | {'FWHM_x_esr':<12} | "
            f"{'dFWHM_x_esr':<11} | {'FWHM_y_esr':<12} | {'dFWHM_y_esr':<11}"
        )
        print(header)
        print("-" * len(header))
        for res in sorted(
            all_results, key=lambda x: (x["collimator"], x["thickness"], x["distance"])
        ):
            def format_val(val, fmt_str):
                return f"{val:{fmt_str}}" if val is not None else "N/A"

            print(
                f"{res['filename']:<20s} | {res['collimator']:<4} | {res['thickness']:<5} | {res['distance']:<4} | "
                f"{res['sensitivity']*100:8.4f} | {res['delta_sensitivity']*100:11.4f} | "
                f"{format_val(res.get('fwhm_x_true'), '>13.4f')} | {format_val(res.get('dfwhm_x_true'), '>12.4f')} | "
                f"{format_val(res.get('fwhm_y_true'), '>13.4f')} | {format_val(res.get('dfwhm_y_true'), '>12.4f')} | "
                f"{format_val(res.get('fwhm_x_esr'), '>12.4f')} | {format_val(res.get('dfwhm_x_esr'), '>11.4f')} | "
                f"{format_val(res.get('fwhm_y_esr'), '>12.4f')} | {format_val(res.get('dfwhm_y_esr'), '>11.4f')}"
            )

        # Print a ranked list based on the Figure of Merit
        if CALCULATE_FIGURE_OF_MERIT:
            print("\n" + "=" * 80 + "\n--- FIGURE OF MERIT (FoM) RANKING ---")
            print(
                f"{'Rank':<5} | {'File':<20} | {'FoM (Sens/FWHM)':<20} | "
                f"{'Sensitivity (%)':<15} | {'Mean FWHM_esr (cm)':<20}"
            )
            print("-" * 90)
            for i, res in enumerate(
                sorted(all_results, key=lambda x: x.get("fom", 0), reverse=True)
            ):
                if res.get("fom") is not None:
                    mean_fwhm = get_mean_fwhm(res, "esr")
                    print(
                        f"{i+1:<5} | {res['filename']:<20} | {res['fom']:<20.3e} | "
                        f"{res['sensitivity']*100:<15.4f} | {mean_fwhm:<20.4f}"
                    )

        plot_summary_graphs(all_results, output_dir)

        if CALCULATE_FIGURE_OF_MERIT:
            plot_fom_summary(all_results, output_dir)

        if GENERATE_LATEX_OUTPUT:
            generate_latex_output(all_results, output_dir)
