#!/usr/bin/env python3

"""
gulls_viz.py

Author: Farzaneh Zohrabi
Date: August 2025
Affiliation: Louisiana State University / RGES PIT

Part of the gulls microlensing simulation pipeline.

Generates visual panels for simulated microlensing events, including:
light curves, planetary signal zoom-ins, caustic structures, Einstein rings, 
source/lens trajectories, and key event parameters. Designed to facilitate event-level 
inspection, validation.

"""
__author__ = "Farzaneh Zohrabi"
__date__ = "2025-08-26"
__version__ = "1.0"


import argparse
import os
import math
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import cm, colormaps
import VBMicrolensing

VBM = VBMicrolensing.VBMicrolensing()

EVENT_TYPE      = "Bound Planets"
SEMIMAJOR_AXIS  = "Log-uniform 0.3–30 AU"
PARAMETRIZATION = "Caustic Region of Interest(CROIN)"
MASS_TAGS = ["m-20", "m-10", "m00", "m10", "m20", "m30", "m40"]
MASS_TAG_TO_MEARTH = {
    "m-20": 0.01,
    "m-10": 0.1,
    "m00": 1.0,
    "m10": 10.0,
    "m20": 100.0,
    "m30": 1000.0,
    "m40": 10000.0
}
#Check weather file and make sure the events peak in within the seasons.
def get_weather_file_path(num_seasons):
    if num_seasons == 1:
        return "./WFIRSTLSST1-72.weather"
    elif num_seasons == 6:
        return "./WFIRST6-72.weather"
    else:
        print(f"Invalid number of seasons: {num_seasons}. Must be 1 or 6.")
        sys.exit(1)

def load_weather_sequence(file_path):
    weather_sequence = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                time, condition = line.split()
                weather_sequence[float(time)] = int(condition)
            except ValueError:
                print(f"Skipping invalid line in file: {line}")
    return weather_sequence

def in_season(t0, Paramfile, weather_sequence):
    if int(t0) < 0 or int(t0) > Paramfile['NUM_SIM_DAYS']:
        return 0
    time_key = round(t0 * 4) / 4
    return 1 if weather_sequence.get(time_key, 0) > 0 else 0


def extract_numbers_from_filename(fname):
    base = os.path.splitext(os.path.splitext(fname)[0])[0]  # remove .lc then .det
    parts = base.split('_')
    try:
        subrun  = int(parts[-3])
        field   = int(parts[-2])
        eventid = int(parts[-1])
        return (subrun, field, eventid)
    except Exception:
        return None

def load_events_table(filename):
    if filename.endswith('.hdf5'):
        try:
            return pd.read_hdf(filename)
        except:
            with pd.HDFStore(filename) as store:
                key = store.keys()[0]
                return store[key]
    else:
        return pd.read_csv(filename, sep=r'\s+')

def load_field_coords(fields_file):
    if not fields_file or not os.path.exists(fields_file):
        return {}
    df = pd.read_csv(fields_file, delim_whitespace=True, comment='#', header=None,  low_memory=False)
    if df.shape[1] < 3:
        df = pd.read_csv(fields_file, sep=',', comment='#', header=None)
    coords = {}
    for _, row in df.iterrows():
        try:
            f = int(row[0]); l = float(row[1]); b = float(row[2])
            coords[f] = (l, b)
        except:
            continue
    return coords
# Data summary
def mass_bin(out_file):
    base = os.path.splitext(os.path.splitext(out_file)[0])[0]  # remove .lc then .det
    parts = base.split('_')
    mass_tag = str(parts[-1])
    mass = MASS_TAG_TO_MEARTH[mass_tag]
    planet_mass = f"{mass:.2g} M⊕"
    print(planet_mass)
    return planet_mass
    

    
    
def write_fancy_summary_page(pdf, N_total, N_detected,
                             raw_detection_rate, weighted_detection_rate,
                             detected_fields, field_coords,planet_mass):
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    table_data = [
        ["Event Type", EVENT_TYPE],
        ["Planet Mass", planet_mass],
        ["Semimajor axis", SEMIMAJOR_AXIS],
        ["Parametrization", PARAMETRIZATION],
        #["Total events", f"{N_total:,}"],
        #["Detections", f"{N_detected:,}"],
        #["Raw detection rate", f"{raw_detection_rate:.2f}"],
        #["Weighted detection rate", f"{weighted_detection_rate:.2f}"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Summary Statistic", "Value"],
        cellLoc='left',
        loc='center',
        colWidths=[0.40, 0.40],
        bbox=[0.08, 0.47, 0.85, 0.46]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    for (r,c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=14)
            cell.set_facecolor("#40466e")
        else:
            cell.set_facecolor("#f9fafe" if r % 2 == 0 else "#dbe7fa")
        if c == 1:
            cell.get_text().set_ha("left")
    if detected_fields and field_coords:
        rows = [["Field#", "l (deg)", "b (deg)"]]
        for f in sorted(detected_fields):
            l,b = field_coords.get(f,(None,None))
            rows.append([
                f"{f:>3}",
                f"{l:.3f}" if l is not None else "--",
                f"{b:.3f}" if b is not None else "--"
            ])
        field_tab = ax.table(
            cellText=rows,
            cellLoc='center',
            loc='bottom',
            bbox=[0.08, 0.04, 0.38, 0.37]
        )
        field_tab.auto_set_font_size(False)
        field_tab.set_fontsize(16)
        for (r,c), cell in field_tab.get_celld().items():
            if r == 0:
                cell.set_text_props(weight='bold', color='white', fontsize=16)
                cell.set_facecolor("#246ba1")
            else:
                cell.set_facecolor("#f4f8fa" if r % 2 == 0 else "#cfe0f7")
    plt.tight_layout(rect=[0,0,1,1])
    pdf.savefig(fig); plt.close(fig)

# lc header - the first row without # in the latest lc files is the header
'''
HEADER_LC = [
    "Simulation_time", "measured_relative_flux", "measured_relative_flux_error",
    "true_relative_flux", "true_relative_flux_error", "observatory_code",
    "saturation_flag", "best_single_lens_fit", "parallax_shift_t", "parallax_shift_u",
    "BJD", "source_x", "source_y", "lens1_x", "lens1_y", "lens2_x", "lens2_y",
    "x", "y", "z",
]
# header sample for fisher matrix case
HEADER_LC_fisher =["Simulation_time", "measured_relative_flux", "measured_relative_flux_error", "true_relative_flux", "true_relative_flux_error",
 "observatory_code", "saturation_flag", "best_single_lens_fit", "parallax_shift_t", "parallax_shift_u", "BJD", "source_x", "source_y", "lens1_x" ,
 "lens1_y", "lens2_x", "lens2_y","parallax_shift_x", "parallax_shift_y", "parallax_shift_z", "ObsGroup_0_dF_t0", "ObsGroup_0_dF_tE",
  "ObsGroup_0_dF_u0", "ObsGroup_0_dF_alpha", "ObsGroup_0_dF_s", "ObsGroup_0_dF_q", "ObsGroup_0_dF_rs", "ObsGroup_0_dF_piEN",
  " ObsGroup_0_dF_piEE", "ObsGroup_0_dF_Fbase0" ,"ObsGroup_0_dF_fs0", "ObsGroup_0_dF_Fbase1",
    "ObsGroup_0_dF_fs1", "ObsGroup_0_dF_Fbase2", "ObsGroup_0_dF_fs2 "
]
'''
# plot the lightcurve and zoom around the planetary signal
def plot_lightcurve_with_zoom(ax_zoom, ax_main, lc_path, detection_row, obsgroup):

    with open(lc_path) as fin:
        skiprows = 0
        while True:
            pos = fin.tell()
            line = fin.readline()
            if not line or not line.strip().startswith("#"):
                fin.seek(pos)
                break
            skiprows += 1

    data = pd.read_csv(lc_path, sep=r'\s+',
                        skiprows=skiprows,header=0,  low_memory=False)
    data = data[data['observatory_code'] == 0]
    if data.empty:
        ax_main.text(0.5, 0.5, "No obs 0 data", ha='center', va='center', fontsize=16)
        ax_zoom.axis('off')
        return

    with open(lc_path) as f:
        header_rows = [next(f).strip() for _ in range(4)]
    header_vals = [row.split() for row in header_rows]
    fs0 = float(header_vals[0][1])
    m0  = float(header_vals[-1][1]) + 2.5 * np.log10(fs0)

    def mag(F):
        return m0 - 2.5 * np.log10(F)
    def magerr(F, E):
        return 2.5 / np.log(10) * (E / F)

    T = data['Simulation_time'].values
    F = data['measured_relative_flux'].values
    E = data['measured_relative_flux_error'].values
    model = data['true_relative_flux'].values
    single_fit = data['best_single_lens_fit'].values

    t0     = detection_row['t0lens1']
    tE     = detection_row['tE_ref']
    tcroin = detection_row['tcroin']
    rcroin = detection_row['rcroin']

    if np.isnan(t0) or np.isnan(tE) or np.isnan(tcroin):
        ax_main.errorbar(T, mag(F), yerr=magerr(F,E),
                         fmt='.', markersize=1, alpha=0.3, zorder=1,rasterized=True)
        ax_main.plot(T, mag(model), '-', color='black',
                     linewidth=2, zorder=2, label="Best Model",rasterized=True)
        ax_main.plot(T, mag(single_fit), '--', color='red',
                     linewidth=2, zorder=3, label="Single Lens Fit",rasterized=True)
        ax_main.set_title(os.path.basename(lc_path), fontsize=16)
        ax_main.invert_yaxis()
        cols = [f"ObsGroup_{obsgroup}_chi2","t0lens1","tE_ref","u0lens1","rho","piEN","piEE"]
        labs = ["Δχ²","t₀","t_E","u₀","ρ","π_EN","π_EE"]
        texts = []
        for c, lab in zip(cols, labs):
            v = detection_row.get(c, np.nan)
            texts.append(f"{lab}: {v:.6f}" if not np.isnan(v) else f"{lab}: n/a")
        ax_main.text(0.03, 0.97, "\n".join(texts),
                     transform=ax_main.transAxes, fontsize=16,
                     va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))
        ax_main.set_xlabel("Time (days)", fontsize=16)
        ax_main.set_ylabel("F146 Magnitude", fontsize=16)
        ax_main.legend(fontsize=14)
        ax_zoom.axis('off')  # no zoom
        return

    
    # compute residuals for the zoomed plot
    residuals = np.abs((F - single_fit) / E)

    mask_window = (T >= t0 - 2*tE) & (T <= t0 + 2*tE)
    if mask_window.any():
        temp = residuals.copy()
        temp[~mask_window] = -np.inf     
        idx_peak = np.argmax(temp)
        t_peak = T[idx_peak]

        half_width = 0.1 * tE
        t_min_z = t_peak - half_width
        t_max_z = t_peak + half_width
        mask_zoom = (T >= t_min_z) & (T <= t_max_z)
    else:
        half_width = 0.1 * tE
        t_min_z, t_max_z = t0 - half_width, t0 + half_width
        mask_zoom = (T >= t_min_z) & (T <= t_max_z)

    ax_zoom.errorbar(
        T[mask_zoom],
        mag(F[mask_zoom]),
        yerr=magerr(F[mask_zoom], E[mask_zoom]),
        fmt='.', markersize=1.5, alpha=0.5, zorder=1,rasterized=True)
    ax_zoom.plot(T[mask_zoom], mag(model[mask_zoom]), '-', color='black',
                 linewidth=2, zorder=2,rasterized=True)
    ax_zoom.plot(T[mask_zoom], mag(single_fit[mask_zoom]), '--', color='red',
                 linewidth=2, zorder=3,rasterized=True)
    ax_zoom.set_xlim(t_min_z, t_max_z)
    ax_zoom.set_title("Planetary Signal", fontsize=16)
    ax_zoom.invert_yaxis()
    ax_zoom.set_ylabel("Magnitude", fontsize=16)
    ax_zoom.set_aspect('auto')

    #ax_zoom.axvline(tcroin, color='red', linestyle=':', linewidth=1.2, zorder=4)
    valid_data = ~np.isnan(F)
    # load weather to mask out-of-season points (6 seasons assumed)
    weather_seq = load_weather_sequence(get_weather_file_path(6))
    season = np.array([in_season(t, {'NUM_SIM_DAYS': 2010}, weather_seq) for t in T], dtype=bool)
    valid = valid_data & season
    ax_main.errorbar(T[valid], mag(F[valid]), yerr=magerr(F[valid],E[valid]),
                     fmt='.', markersize=1, alpha=0.3, zorder=1)
    mm = np.ma.masked_where(~valid, mag(model))
    ms = np.ma.masked_where(~valid, mag(single_fit))
    ax_main.plot(T, mm, '-', color='black',
                 linewidth=2, zorder=2, label="Best Model")
    ax_main.plot(T, ms, '--', color='red',
                 linewidth=2, zorder=3, label="Single Lens Fit")
    #ax_main.axvline(t0, color='blue', linestyle=':', linewidth=1.2, zorder=4)

    if tE<30:
    	ax_main.set_xlim(t0 - 1.5 * tE, t0 + 1.5 * tE)
    ax_main.invert_yaxis()

    cols = [f"ObsGroup_{obsgroup}_chi2","t0lens1","tE_ref","u0lens1","Planet_s","Planet_q","rho","piEN","piEE"]
    labs = ["Δχ²","t₀","t_E","u₀","s","q","ρ","π_EN","π_EE"]
    texts = []
    for c,lab in zip(cols, labs):
        v = detection_row.get(c, np.nan)
        if abs(v) < 1e-2 and v != 0:
            fmt = "{:.2e}"
        else:
            fmt = "{:.4f}"
        txt = f"{lab}: " + fmt.format(v)
        texts.append(txt)

    ax_main.text(0.03, 0.97, "\n".join(texts),
                 transform=ax_main.transAxes, fontsize=11,
                 va='top', ha='left',
                 bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))
    ax_main.set_xlabel("Time (days)", fontsize=16)
    ax_main.set_ylabel("F146 Magnitude", fontsize=16)
    ax_main.legend(fontsize=14)

# plot the caustics and source trajectory. also plots zoomed-ins for each individual caustic
def plot_caustics_panel(fig, spec, row, lc_path, cluster_thresh=0.05):

    # define base colormap for clusters
    thr = cluster_thresh
    base_cmap = colormaps['tab10']

    # 1) Load trajectory, skipping comment lines
    skip = 0
    with open(lc_path) as f:
        while f.readline().startswith('#'):
            skip += 1
    df = pd.read_csv(lc_path,
                     sep=r"\s+", 
                      skiprows=skip,header=0,
                     low_memory=False).apply(pd.to_numeric, errors='coerce')


    t, x, y = df['Simulation_time'].values, df['source_x'].values, df['source_y'].values

    t0, tE = row['t0lens1'], row['tE_ref']
    mask = (t >= t0 - 1.5*tE) & (t <= t0 + 1.5*tE)
    main_t = t[mask] if mask.any() else t
    main_x = x[mask] if mask.any() else x
    main_y = y[mask] if mask.any() else y

    caustics = VBM.Caustics(row['Planet_s'], row['Planet_q'])
    idx0 = np.argmin(np.abs(main_t - t0))
    x0, y0 = main_x[idx0], main_y[idx0]

    cents = [(np.mean(c[0]), np.mean(c[1])) for c in caustics]
    clusters = []
    for i, cen in enumerate(cents):
        placed = False
        for cl in clusters:
            if any(math.hypot(cen[0]-c0, cen[1]-c1) < thr for c0, c1 in cl['cents']):
                cl['idxs'].append(i); cl['cents'].append(cen)
                placed = True; break
        if not placed:
            clusters.append({'idxs':[i], 'cents':[cen]})

    # 5) Build nested grid
    n = len(clusters)
    nested = GridSpecFromSubplotSpec(2, n, subplot_spec=spec,
                                     height_ratios=[1,1.5],
                                     hspace=0.02, wspace=0.02)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True); fmt.set_powerlimits((0,0)); fmt.set_useOffset(False)

    no_merge = (len(clusters) == len(caustics))
    pad_factor = 0.9
    for i, cl in enumerate(clusters):
        ax = fig.add_subplot(nested[0, i])
        xs = np.hstack([caustics[j][0] for j in cl['idxs']])
        ys = np.hstack([caustics[j][1] for j in cl['idxs']])

        if no_merge:
            idx = cl['idxs'][0]
            cx = np.append(caustics[idx][0], caustics[idx][0][0])
            cy = np.append(caustics[idx][1], caustics[idx][1][0])
            ax.plot(cx, cy, color='blue', lw=1.5, rasterized=True)
            ax.scatter(main_x, main_y, s=1, color='black', alpha=0.4, rasterized=True)
            ax.scatter([x0], [y0], s=40, edgecolors='red', facecolors='none', rasterized=True)
            # let matplotlib autoscale and add 10% margins
            ax.relim()
            ax.autoscale_view()
            ax.margins(0.01)
            # scientific ticks
            ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0), useMathText=True)
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
        else:
            shades = base_cmap(np.linspace(0,1,len(cl['idxs'])))
            for j, idx in enumerate(cl['idxs']):
                cx = np.append(caustics[idx][0], caustics[idx][0][0])
                cy = np.append(caustics[idx][1], caustics[idx][1][0])
                ax.plot(cx, cy, color=shades[j], lw=2, alpha=0.9, rasterized=True)
                span = max(xs.ptp(), ys.ptp())
                mid_x, mid_y = xs.mean(), ys.mean()
                pad = pad_factor * span
                ax.set_xlim(mid_x - pad, mid_x + pad)
                ax.set_ylim(mid_y - pad, mid_y + pad)
            ax.scatter(main_x, main_y, s=1, color='black', alpha=0.4, rasterized=True)
            ax.scatter([x0], [y0], s=60, edgecolors='red', facecolors='none', rasterized=True)
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)

        ax.set_aspect('equal', adjustable='box')
        ax.set_box_aspect(1)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.tick_params(axis='both', labelsize=14)

        

    axm = fig.add_subplot(nested[1, :])
    for cx_raw, cy_raw in caustics:
        axm.plot(cx_raw, cy_raw, color='purple', lw=4,rasterized=True)
    axm.scatter(main_x, main_y, s=2, alpha=0.5,rasterized=True)
    
    #plots the Einstein Ring
    axm.add_patch(Circle((0,0) , 1, edgecolor='green', ls='--', fill=False, lw=1.5, rasterized=True))

    off = 2
    i1, i2 = max(idx0 - off, 0), min(idx0 + off, len(main_x) - 1)
    dx, dy = main_x[i2] - main_x[i1], main_y[i2] - main_y[i1]
    v = math.hypot(dx, dy)
    if v > 0:
        step = np.median(np.hypot(np.diff(main_x), np.diff(main_y)))
        axm.annotate('',
                     xy=(x0 + dx * 10 * step / v, y0 + dy * 10 * step / v),
                     xytext=(x0, y0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2, mutation_scale=25)
        )
    axm.set_xlabel('x (R_E)', fontsize=16)
    axm.set_ylabel('y (R_E)', fontsize=16)
    axm.set_aspect('equal', adjustable='box')
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True); fmt.set_powerlimits((-2, 3)); fmt.set_useOffset(False)
    axm.xaxis.set_major_formatter(fmt)
    axm.yaxis.set_major_formatter(fmt)
    axm.legend([], frameon=False)
    


def process_outfile(out_file, chi2_threshold, obsgroup, num_seasons, folder, fields_file, pdf):

    df = load_events_table(out_file)
    planet_mass = mass_bin(out_file)
    #("Loaded columns:", df.columns.tolist())
    #print("First few rows:", df.head())
    chi2_col = f"ObsGroup_{obsgroup}_chi2"
    if chi2_col not in df.columns:
        raise KeyError(f"{chi2_col} missing")

    detections = df[
        (df[chi2_col] > chi2_threshold) &
        (df["u0lens1"].abs() < 3) &
        (df["t0lens1"].apply(lambda t: in_season(
            t, {'NUM_SIM_DAYS': 2010},
            load_weather_sequence(get_weather_file_path(num_seasons))
        )))
    ]

    N_total    = len(df)
    N_detected = len(detections)
    raw_rate   = 100 * (N_detected / N_total if N_total else np.nan)
    weighted   = (detections['final_weight'].sum() / df['final_weight'].sum()
                  if 'final_weight' in df.columns else np.nan)
    det_fields    = set(map(int, detections['Field'].unique())) if N_detected else set()
    field_coords  = load_field_coords(fields_file)

    # find .det.lc files
    all_lc = [f for f in os.listdir(folder) if f.endswith('.det.lc')]
    keys   = set(zip(detections.SubRun,
                     detections.Field,
                     detections.EventID))
    sel    = [f for f in all_lc if extract_numbers_from_filename(f) in keys]
    sel.sort(key=lambda fn: extract_numbers_from_filename(fn)[2])
    

    write_fancy_summary_page(pdf, N_total, N_detected, raw_rate, weighted, det_fields, field_coords, planet_mass)
    for fname in sel:
            subrun, field, eid = extract_numbers_from_filename(fname)
            row = detections[
                (detections.SubRun == subrun) &
                (detections.Field == field) &
                (detections.EventID == eid)].iloc[0]
            lc_path = os.path.join(folder, fname)

            fig = plt.figure(figsize=(16, 8), constrained_layout=True)

            outer = GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.1, hspace=0.05,figure=fig)

            inner_lc = GridSpecFromSubplotSpec(
                2, 1,
                subplot_spec=outer[0],
                height_ratios=[1, 2],
                hspace=0.1,
            )
            ax_zoom = fig.add_subplot(inner_lc[0])
            ax_main = fig.add_subplot(inner_lc[1])
            plot_lightcurve_with_zoom(ax_zoom, ax_main, lc_path, row, obsgroup)


            plot_caustics_panel(fig, outer[1], row, lc_path)
            # writes lc file names on the side of the page to find the events of interest easier
            fig.text(
                        0.99, 0.01,
                        f"{fname}",
                        rotation=90,
                        ha='right',
                        va='bottom',
                        fontsize=12
                        )


            pdf.savefig(fig,
                        bbox_inches='tight',    # tightly crop to all artists
                        pad_inches=0.3, dpi=300)
            plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a multi-page PDF of simulated microlensing events with light curves and caustic plots, one event per page.")
    parser.add_argument("outfiles", nargs="+", help=".out/.hdf5 detection files (multiple)")
    parser.add_argument("--chi2",     type=float, default=160.0)
    parser.add_argument("--obsgroup", type=int,   default=0)
    parser.add_argument("--seasons",  type=int,   default=6)
    parser.add_argument("--folder",   type=str,   default=".")
    parser.add_argument("--fields",   type=str,   default=None)
    parser.add_argument("--pdfname",  type=str,   default="gulls_visual_diagnostics.pdf")
    args = parser.parse_args()

    with PdfPages(args.pdfname) as pdf:
        for out_file in args.outfiles:
            print(f"Processing: {out_file}")
            process_outfile(out_file, args.chi2, args.obsgroup,
                            args.seasons, args.folder, args.fields, pdf)

    print(f"Saved combined PDF to {args.pdfname}")

