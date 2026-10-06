# -*- coding: utf-8 -*-
"""
freefall_workbook.py

Helper module for the PHY1035 Freefall practical.

Handles routing a single truncated freefall run into the correct
per-projectile .xlsx workbook, building the workbook (with live Excel
formulas identical to the hand-out template) the first time a given
projectile is seen, and filling the next empty Run column on subsequent
repeats.

Designed to run on a Raspberry Pi 400. Only dependency beyond the standard
library is openpyxl (confirmed installed on the target machine).

Layout replicated from the existing template ("IR Sensor" sheet):
    B1                     projectile label
    A7/B7                  "Reduced chi squared" label + copy of AB405 (highlighted)
    D1/G1/J1/M1/P1         "Run 1".."Run 5" headers
    D3:Q3                  Time/Distance sub-headers for the 5 runs
    Run data columns (Time, Distance) live at:
        Run 1 -> D, E
        Run 2 -> G, H
        Run 3 -> J, K
        Run 4 -> M, N
        Run 5 -> P, Q
    Data rows                4 .. 403  (400 rows; was 4 .. 83)
    S..AC                    Mean/SEM/model/chi-squared formula block
    AD                       constant systematic uncertainty (per row)
    AE                       total uncertainty = sqrt(SEM^2 + systematic^2)
    AB405                    reduced chi-squared summary   (was AB85)
    Z406                     t0 offset (seconds, auto-est.) (was Z86)
    Z407                     systematic uncertainty (m)     (was Z87)

The summary cells keep the same position *relative to the end of the data
block* as the original template; their addresses are derived from
LAST_DATA_ROW, so always refer to them via the constants below.

Chi-squared now divides the (model-data)^2 by the *total* uncertainty
(AE = SEM combined in quadrature with a constant systematic term), not by
the SEM alone, so the reduced chi-squared is not detonated by the sub-mm
run-to-run SEM. The systematic value lives in a single editable cell
(SIGMA_SYS_CELL) and the AD column simply references it, so it reads as a
column of constant values but is changed in one place.
"""

import os
import openpyxl
from openpyxl.styles import Font, PatternFill

try:
    from t0_estimator import estimate_t0_multi
except Exception:
    estimate_t0_multi = None

# ---- Template constants (do not change without re-checking the hand-out sheet) ----
SHEET_NAME = "IR Sensor"
FIRST_DATA_ROW = 4
# Was 80 (~1.5 s at ~19 ms/sample), which silently clipped fan-slowed falls.
# 400 = N_SAMPLES - DROP_IDX in freefall_03, i.e. every post-release sample the
# acquisition loop can return, so the student's cursor click is now the only
# thing that truncates the data. If you lengthen the acquisition, raise this.
N_DATA_ROWS = 400
LAST_DATA_ROW = FIRST_DATA_ROW + N_DATA_ROWS - 1  # 403
T0_DEFAULT = 0.105  # seconds; overwritten by the auto-estimator once runs exist

# Summary cells sit just below the data block (same relative positions as the
# original AB85/Z86/Z87), derived from LAST_DATA_ROW so run data can never
# overwrite them.
CHI2_LABEL_CELL = f"AA{LAST_DATA_ROW + 2}"
CHI2_CELL = f"AB{LAST_DATA_ROW + 2}"            # reduced chi-squared
T0_LABEL_CELL = f"Y{LAST_DATA_ROW + 3}"
T0_CELL = f"Z{LAST_DATA_ROW + 3}"               # release delay t0
SIGMA_SYS_LABEL_CELL = f"Y{LAST_DATA_ROW + 4}"
SIGMA_SYS_CELL = f"Z{LAST_DATA_ROW + 4}"        # systematic uncertainty
SIGMA_SYS_DEFAULT = 0.01  # metres (~10 mm); VL53L1X absolute ranging error. Editable.

# Copy of the reduced chi-squared near the top of the sheet, so students don't
# have to scroll 400 rows to find it. B7 simply references CHI2_CELL.
TOP_CHI2_LABEL_CELL = "A7"
TOP_CHI2_CELL = "B7"

# Original 80-row layout - only used to upgrade workbooks written by the old code.
_LEGACY_LAST_DATA_ROW = 83
_LEGACY_SUMMARY_CELLS = ("AA85", "AB85", "Y86", "Z86", "Y87", "Z87")
_LEGACY_SIGMA_SYS_CELL = "Z87"

# (time_col, dist_col) for each of the 5 runs, 1-indexed to match openpyxl
RUN_COLS = {
    1: ("D", "E"),
    2: ("G", "H"),
    3: ("J", "K"),
    4: ("M", "N"),
    5: ("P", "Q"),
}
RUN_HEADER_COL = {1: "D", 2: "G", 3: "J", 4: "M", 5: "P"}


def _safe_name(name):
    """Turn a free-text projectile name into a safe filename stem."""
    keep = "-_.() "
    cleaned = "".join(c for c in name.strip() if c.isalnum() or c in keep)
    cleaned = cleaned.replace(" ", "_")
    return cleaned or "unnamed_projectile"


def workbook_path(folder, projectile_name):
    return os.path.join(folder, _safe_name(projectile_name) + ".xlsx")


def _write_summary_cells(ws, t0=T0_DEFAULT, sigma_sys=SIGMA_SYS_DEFAULT):
    """t0 offset + systematic uncertainty + highlighted reduced chi-squared."""
    ws[T0_LABEL_CELL] = "t offset"
    ws[T0_CELL] = t0
    ws[SIGMA_SYS_LABEL_CELL] = "systematic error (m)"
    ws[SIGMA_SYS_CELL] = sigma_sys
    ws[CHI2_LABEL_CELL] = "Reduced Chi squared"
    ws[CHI2_CELL] = (f"=SUM(AC{FIRST_DATA_ROW}:AC{LAST_DATA_ROW})/"
                     f"(COUNT(AC{FIRST_DATA_ROW}:AC{LAST_DATA_ROW})-1)")
    # mirror it at the top of the sheet (A7 label, B7 value)
    ws[TOP_CHI2_LABEL_CELL] = "Reduced chi squared"
    ws[TOP_CHI2_LABEL_CELL].font = Font(bold=True)
    ws[TOP_CHI2_CELL] = f"={CHI2_CELL}"
    # same highlight on both copies
    for addr in (CHI2_CELL, TOP_CHI2_CELL):
        ws[addr].fill = PatternFill(fill_type="solid", fgColor="FFFF00")  # yellow
        ws[addr].font = Font(bold=True, color="FF0000")                   # red


def _write_row_formulas(ws, first_row, last_row):
    """Per-row live formula block (S..AE) for rows first_row..last_row."""
    for r in range(first_row, last_row + 1):
        # references to the five runs' time / distance cells
        t_refs = ",".join(f"{RUN_COLS[run][0]}{r}" for run in (5, 4, 3, 2, 1))
        d_refs = ",".join(f"{RUN_COLS[run][1]}{r}" for run in (5, 4, 3, 2, 1))
        # IFERROR guards keep the empty rows (no data yet) blank instead of
        # showing #DIV/0! from AVERAGE/STDEV over empty cells. Filled rows are
        # unaffected. The reduced chi-squared was always correct regardless
        # (AC is gated on Y=5), this is purely so the sheet looks clean.
        ws[f"S{r}"] = f"=IFERROR(AVERAGE({t_refs})*0.001,\"\")"
        ws[f"T{r}"] = f"=S{r}"
        ws[f"U{r}"] = f"=IFERROR(AVERAGE({d_refs})*0.001,\"\")"
        ws[f"V{r}"] = f"=IFERROR(U{r}-$U${FIRST_DATA_ROW},\"\")"
        ws[f"W{r}"] = f"=IFERROR(STDEV({d_refs})*0.001,\"\")"
        ws[f"X{r}"] = f"=IFERROR(W{r}/SQRT(5),\"\")"
        ws[f"Z{r}"] = f"=IFERROR(0.5*9.81*MAX(0,(T{r}-${T0_CELL}))^2,\"\")"
        ws[f"AA{r}"] = f"=IFERROR((Z{r}-V{r})^2,\"\")"
        # constant systematic (references the single editable cell) ...
        ws[f"AD{r}"] = f"=${SIGMA_SYS_CELL}"
        # ... combined with the SEM in quadrature ...
        ws[f"AE{r}"] = f"=IFERROR(SQRT(X{r}^2+AD{r}^2),\"\")"
        # ... and chi-squared uses that total uncertainty, not the SEM alone.
        ws[f"AB{r}"] = f"=IFERROR(AA{r}/(AE{r})^2,\"\")"
        d_count_refs = ",".join(f"{RUN_COLS[run][1]}{r}" for run in (1, 2, 3, 4, 5))
        ws[f"Y{r}"] = f"=COUNT({d_count_refs})"
        ws[f"AC{r}"] = f'=IF(Y{r}=5,AB{r},"")'


def _build_new_workbook(projectile_name):
    """Create a fresh workbook with all headers and live formulas, no run data yet."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = SHEET_NAME

    bold = Font(bold=True)

    # --- labels / headers ---
    ws["A1"] = "Enter projectile name here   =>"
    ws["B1"] = projectile_name
    ws["B1"].font = bold

    for run, col in RUN_HEADER_COL.items():
        ws[f"{col}1"] = f"Run {run}"
        ws[f"{col}1"].font = bold

    # dynamic left-column labels (match template)
    ws["A3"] = '=B1 & " mean distance (m)"'
    ws["A4"] = '=B1 & " residuals squared"'
    ws["A5"] = '=B1 & " chi-squared"'

    # Time/Distance sub-headers for each run
    for run, (tcol, dcol) in RUN_COLS.items():
        ws[f"{tcol}3"] = "Time"
        ws[f"{dcol}3"] = "Distance"

    # analysis block headers (row 3)
    ws["S3"] = "Mean Time (s)"
    ws["T3"] = "Zeroed mean time (s)"
    ws["U3"] = "Mean distance (m)"
    ws["V3"] = "Zeroed mean distance (m)"
    ws["W3"] = "Stdev distance (m)"
    ws["X3"] = "Standard error of the mean (m)"
    ws["Y3"] = "Runs present"
    ws["Z3"] = "Model distance (m)"
    ws["AA3"] = "(model-data)^2"
    ws["AB3"] = "Chi squared"
    ws["AC3"] = "Chi squared (5 runs only)"
    ws["AD3"] = "Systematic error (m)"
    ws["AE3"] = "Total uncertainty (m)"

    # keep the headers visible when scrolling down a long (400-row) run
    ws.freeze_panes = "A4"

    _write_summary_cells(ws)
    _write_row_formulas(ws, FIRST_DATA_ROW, LAST_DATA_ROW)
    return wb


def _autosize_columns(ws, min_width=10):
    """
    Widen every column so its text labels are readable without clicking.
    Only text labels count (data and formula results are numbers that fit in the
    minimum width). The A3:A5 labels are formulas of the form =B1 & " ...",
    so their displayed length is worked out from the projectile name in B1.
    """
    from openpyxl.utils import get_column_letter
    name = str(ws["B1"].value or "")
    widths = {}
    for row in ws.iter_rows():
        for cell in row:
            v = cell.value
            if not isinstance(v, str):
                continue
            if v.startswith("=B1 &"):
                text = name + v.split('"')[1]          # displayed label
            elif v.startswith("="):
                continue                               # numeric formula
            else:
                text = v
            n = len(text) * (1.15 if cell.font and cell.font.b else 1.0)
            col = get_column_letter(cell.column)
            widths[col] = max(widths.get(col, 0), n)
    for col, n in widths.items():
        ws.column_dimensions[col].width = max(min_width, round(n + 2))


def _upgrade_legacy_layout(ws):
    """
    Upgrade a workbook written by the old 80-row version in place, so a
    condition started earlier with the old code can still take long runs.
    Run data is untouched; the user's systematic-error value is carried over.
    Returns True if an upgrade was performed.
    """
    if ws[f"S{LAST_DATA_ROW}"].value not in (None, ""):
        return False                       # already the current layout
    if ws[f"S{_LEGACY_LAST_DATA_ROW}"].value in (None, ""):
        return False                       # not a recognisable old workbook

    sigma_sys = ws[_LEGACY_SIGMA_SYS_CELL].value
    if not isinstance(sigma_sys, (int, float)):
        sigma_sys = SIGMA_SYS_DEFAULT

    # old summary cells sit inside the new data block - clear them first
    for addr in _LEGACY_SUMMARY_CELLS:
        ws[addr] = None
        ws[addr].fill = PatternFill(fill_type=None)
        ws[addr].font = Font()

    _write_summary_cells(ws, sigma_sys=sigma_sys)
    # rewrite all rows: the existing ones reference the old t0/sigma cells
    _write_row_formulas(ws, FIRST_DATA_ROW, LAST_DATA_ROW)
    ws.freeze_panes = "A4"
    return True


def next_empty_run(ws):
    """Return the lowest run number (1..5) whose data columns are empty, or None if full."""
    for run in (1, 2, 3, 4, 5):
        tcol = RUN_COLS[run][0]
        if ws[f"{tcol}{FIRST_DATA_ROW}"].value in (None, ""):
            return run
    return None


def runs_status(ws):
    """Return a dict {run: bool_filled} for reporting to the student."""
    status = {}
    for run in (1, 2, 3, 4, 5):
        tcol = RUN_COLS[run][0]
        status[run] = ws[f"{tcol}{FIRST_DATA_ROW}"].value not in (None, "")
    return status


def _read_runs_from_ws(ws):
    """Pull (times_s, distances_m) for every run that has data, zeroed like the sheet does."""
    runs = []
    for run in (1, 2, 3, 4, 5):
        tcol, dcol = RUN_COLS[run]
        ts, ds = [], []
        for r in range(FIRST_DATA_ROW, LAST_DATA_ROW + 1):
            tv = ws[f"{tcol}{r}"].value
            dv = ws[f"{dcol}{r}"].value
            if tv is None or dv is None or tv == "" or dv == "":
                continue
            ts.append(float(tv))
            ds.append(float(dv))
        if len(ts) >= 5:
            ts = [x / 1000.0 for x in ts]          # ms -> s
            ds = [x / 1000.0 for x in ds]          # mm -> m
            base = sum(ds[:5]) / 5.0                # early baseline
            ds = [d - base for d in ds]
            t0row = ts[0]
            ts = [t - t0row for t in ts]
            runs.append((ts, ds))
    return runs


def refresh_t0(ws):
    """Estimate t0 from all runs currently in the sheet and write it into T0_CELL.
    Returns the estimate (seconds) or None if not estimable / estimator unavailable."""
    if estimate_t0_multi is None:
        return None
    runs = _read_runs_from_ws(ws)
    if not runs:
        return None
    t0 = estimate_t0_multi(runs)
    if t0 is not None:
        ws[T0_CELL] = round(float(t0), 4)
    return t0


def write_run(folder, projectile_name, times_ms, distances_mm,
              run_number=None, overwrite=False):
    """
    Write one truncated run into the projectile's workbook.

    Returns
    -------
    (path, run_number, status_dict)
    """
    os.makedirs(folder, exist_ok=True)
    path = workbook_path(folder, projectile_name)

    if os.path.exists(path):
        wb = openpyxl.load_workbook(path)
        ws = wb[SHEET_NAME] if SHEET_NAME in wb.sheetnames else wb.active
        if _upgrade_legacy_layout(ws):
            print(f"  (Upgraded {os.path.basename(path)} from the old 80-row layout; "
                  f"summary cells are now {CHI2_CELL}/{T0_CELL}/{SIGMA_SYS_CELL}.)")
    else:
        wb = _build_new_workbook(projectile_name)
        ws = wb[SHEET_NAME]

    if run_number is None:
        run_number = next_empty_run(ws)
        if run_number is None:
            raise RuntimeError(
                f"All 5 run slots for '{projectile_name}' are already filled "
                f"({path}). Use overwrite=True with an explicit run_number to replace one."
            )

    if run_number not in RUN_COLS:
        raise ValueError(f"run_number must be 1..5, got {run_number}")

    tcol, dcol = RUN_COLS[run_number]
    already_filled = ws[f"{tcol}{FIRST_DATA_ROW}"].value not in (None, "")
    if already_filled and not overwrite:
        raise RuntimeError(
            f"Run {run_number} for '{projectile_name}' already contains data. "
            f"Refusing to overwrite (pass overwrite=True to force)."
        )

    n_in = min(len(times_ms), len(distances_mm))
    n = min(n_in, N_DATA_ROWS)
    if n_in > N_DATA_ROWS:
        # no longer silent: say so if the workbook can't hold the whole run
        print(f"  WARNING: run has {n_in} points but the workbook holds {N_DATA_ROWS}; "
              f"only the first {N_DATA_ROWS} were written (the .txt file has them all). "
              f"Increase N_DATA_ROWS in freefall_workbook.py.")

    # clear the column first (in case of overwrite of a longer previous run)
    for r in range(FIRST_DATA_ROW, LAST_DATA_ROW + 1):
        ws[f"{tcol}{r}"] = None
        ws[f"{dcol}{r}"] = None

    for i in range(n):
        r = FIRST_DATA_ROW + i
        ws[f"{tcol}{r}"] = float(times_ms[i])
        ws[f"{dcol}{r}"] = float(distances_mm[i])

    # keep the label fresh (protects against copy/paste mislabelling)
    ws["B1"] = projectile_name

    # make every column header readable at a glance
    _autosize_columns(ws)

    # auto-estimate t0 from all runs present and write it into the sheet
    try:
        refresh_t0(ws)
    except Exception:
        pass

    wb.save(path)
    return path, run_number, runs_status(ws)


if __name__ == "__main__":
    # quick self-test with synthetic data: a slow (fan-like) fall lasting ~5 s
    import numpy as np
    tmp = "/tmp/ff_test"
    t = np.arange(0, 260) * 19.0
    d = 0.5 * 1.5 * np.clip((t / 1000 - 0.11), 0, None) ** 2 * 1000   # g_eff ~ 1.5
    for run in range(1, 6):
        noise = np.random.normal(0, 3, size=len(d))
        path, rn, status = write_run(tmp, "Test Ball", t, d + noise)
        print(f"wrote run {rn}; status={status}; file={path}")
    try:
        write_run(tmp, "Test Ball", t, d)
    except RuntimeError as e:
        print("correctly refused 6th run:", e)
