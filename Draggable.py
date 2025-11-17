import json, math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Slider, Select, TextInput, Div, Button, RadioButtonGroup,
    NumberFormatter, Range1d,
    InlineStyleSheet, NumeralTickFormatter, HoverTool, PointDrawTool, CustomJS,
)
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral4

# ======================= THEME & CONSTANTS =======================

curdoc().title = "Population Projections for Korea — Bokeh Dashboard"
curdoc().template = "templates/dark_base.html"

# Add extra CSS via template variables as backup
curdoc().template_variables["extra_css"] = """
<style>
html, body {
  background-color: #020617 !important;
  min-height: 100vh !important;
}
</style>
"""

# Dark theme focused on figures & axes
curdoc().theme = Theme(json={
    "attrs": {
        "figure": {
            "background_fill_color": "#020617",
            "border_fill_color": "#020617",
            "outline_line_color": "#111827",
        },
        "Axis": {
            "major_label_text_color": "#E5E7EB",
            "major_label_text_font_size": "14pt",
            "axis_label_text_color": "#E5E7EB",
            "axis_label_text_font_size": "15pt",
        },
        "Title": {
            "text_color": "#F9FAFB",
            "text_font_size": "17pt",
        },
        "Legend": {
            "label_text_color": "#E5E7EB",
            "background_fill_alpha": 0.0,
            "border_line_alpha": 0.0,
        },
        "Grid": {
            "grid_line_color": "#374151",
        },
    }
})

# Inline CSS to give the whole app a cohesive dark look
dashboard_css = """
:host {
  --dashboard-bg: #020617;
  --dashboard-surface: #020617;
  --dashboard-border: #1f2937;
  --dashboard-text-main: #e5e7eb;
  --dashboard-text-muted: #9ca3af;
  --dashboard-accent: #38bdf8;

  background-color: var(--dashboard-bg) !important;
  color: var(--dashboard-text-main);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 18px;
  min-height: 100vh !important;
}

* {
  box-sizing: border-box;
}

html {
  background-color: #020617 !important;
  min-height: 100vh !important;
}

body {
  background-color: #020617 !important;
  min-height: 100vh !important;
}

.bk-root {
  background-color: transparent !important;
  min-height: 100vh !important;
}

.dashboard-root {
  background-color: transparent !important;
  min-height: 100vh !important;
}

/* Headings in the main panel */
.dashboard-main h1 {
  font-size: 1.8rem;
  margin-bottom: 0.25rem;
}
.dashboard-main h2 {
  font-size: 1.5rem;
  margin-top: 1.25rem;
  margin-bottom: 0.4rem;
}
.dashboard-main h3 {
  font-size: 1.25rem;
  margin-top: 1rem;
  margin-bottom: 0.35rem;
}

/* Paragraph text in the main panel */
.dashboard-main p {
  margin-top: 0.25rem;
  margin-bottom: 0.35rem;
  color: var(--dashboard-text-muted);
  line-height: 1.45;
  font-size: 1.1rem;
}

/* Section separators */
.dashboard-main hr {
  border: none;
  border-top: 1px solid rgba(148, 163, 184, 0.35);
  margin: 1.1rem 0;
}

/* Scenario radio button group */
.scenario-radio-group .bk-btn-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}

.scenario-radio-group .bk-btn {
  width: 100% !important;
  padding: 12px 16px !important;
  font-size: 16px !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  border: 2px solid #374151 !important;
  background-color: #1f2937 !important;
  background-image: none !important;
  color: #9ca3af !important;
  transition: all 0.2s ease !important;
  box-shadow: none !important;
  text-align: center !important;
}

.scenario-radio-group .bk-btn:hover {
  border-color: #4b5563 !important;
  background-color: #374151 !important;
  color: #d1d5db !important;
}

.scenario-radio-group .bk-btn.bk-active {
  border-color: #3b82f6 !important;
  background-color: #2563eb !important;
  background-image: none !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}

/* Sidebar and main content shells */
.dashboard-sidebar {
  background: radial-gradient(circle at top left, #020617 0, #020617 55%, #020617 100%);
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.5);
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
  padding: 16px 18px;
}

.dashboard-main {
  background: radial-gradient(circle at top, #020617 0, #020617 45%, #020617 100%);
  border-radius: 18px;
  border: 1px solid rgba(55, 65, 81, 0.8);
  box-shadow: 0 22px 60px rgba(15, 23, 42, 0.95);
  padding: 18px 22px 24px;
}
"""

dashboard_stylesheet = InlineStyleSheet(css=dashboard_css)

AGE_MAX = 117
AGES = np.arange(0, AGE_MAX + 1)
MALE, FEMALE = "Male", "Female"

# Data base year (where historical data comes from)
DATA_BASE_YEAR = 2020
# Display/projection start year (current year)
BASE_YEAR = 2025
FINAL_YEAR_DEFAULT = 2100
FINAL_YEAR_MIN = 2025
FINAL_YEAR_MAX = 2100

SCENARIOS = ["Low", "Medium", "High", "Custom"]
PRESET_SCENARIOS = ["Low", "Medium", "High"]

STATE = {"years": None, "male": None, "female": None, "total": None, "asfr_shapes": None, "initializing": True}

# ======================= HELPERS =======================

def _safe_norm(v):
    s = v.sum()
    return v / s if s > 0 else v

def interp_from_table(years, table: Dict[int, float]):
    ky = np.array(sorted(table.keys()), dtype=float)
    kv = np.array([table[int(k)] for k in ky], dtype=float)
    return np.interp(years, ky, kv)

def life_expectancy_from_mx(mx_):
    qx = 1.0 - np.exp(-mx_)
    lx = np.ones_like(mx_)
    for x in range(1, len(mx_)):
        lx[x] = lx[x - 1] * (1.0 - qx[x - 1])
    Lx = 0.5 * (lx[:-1] + lx[1:])
    return (Lx.sum() + 0.5 * lx[-1]) / lx[0]

# ================== Load PDF ASSUMPTIONS from JSON ==================

def load_assumptions():
    base_dir = Path(__file__).parent

    def _load_json(name: str):
        with open(base_dir / name, "r", encoding="utf-8") as f:
            return json.load(f)

    tfr_table_raw = _load_json("tfr_table.json")
    leb_table_raw = _load_json("leb_table.json")
    mig_table_raw = _load_json("mig_table.json")
    pdf_total_2070_raw = _load_json("pdf_total_2070.json")
    asfr_cal_raw = _load_json("asfr_calibration.json")

    pdf_params = {}
    # Only load preset scenarios from JSON (not Custom)
    for s in PRESET_SCENARIOS:
        tfr_s = {int(year): float(val) for year, val in tfr_table_raw[s].items()}
        leb_s = {}
        for sex in [MALE, FEMALE]:
            leb_s[sex] = {int(year): float(val) for year, val in leb_table_raw[s][sex].items()}
        mig_s = {int(year): float(val) for year, val in mig_table_raw[s].items()}

        pdf_params[s] = {
            "TFR_table": tfr_s,
            "LEB_table": leb_s,
            "MIG_table": mig_s,
            "mx_base": {MALE: None, FEMALE: None},
            "sex_split": {MALE: 0.50, FEMALE: 0.50},
            "asfr_calibration": float(asfr_cal_raw[s]),
        }

    pdf_total_2070 = {s: float(v) for s, v in pdf_total_2070_raw.items()}
    tfr_defaults = {
        s: {int(year): float(val) for year, val in tfr_table_raw[s].items()} for s in PRESET_SCENARIOS
    }
    return {"PDF_PARAMS": pdf_params, "PDF_TOTAL_2070": pdf_total_2070, "TFR_DEFAULTS": tfr_defaults}

ASSUMPTIONS = load_assumptions()
PDF_PARAMS: Dict[str, dict] = ASSUMPTIONS["PDF_PARAMS"]
PDF_TOTAL_2070: Dict[str, float] = ASSUMPTIONS["PDF_TOTAL_2070"]
TFR_TABLE_DEFAULT: Dict[str, Dict[int, float]] = ASSUMPTIONS["TFR_DEFAULTS"]
# Add Custom scenario based on Medium
TFR_TABLE_DEFAULT["Custom"] = dict(TFR_TABLE_DEFAULT["Medium"])
PDF_PARAMS["Custom"] = dict(PDF_PARAMS["Medium"])
PDF_TOTAL_2070["Custom"] = PDF_TOTAL_2070["Medium"]
TFR_TABLE_EDITABLE: Dict[str, Dict[int, float]] = {s: dict(TFR_TABLE_DEFAULT[s]) for s in SCENARIOS}

# Track current scenario
CURRENT_SCENARIO = {"name": "Medium"}

# ======================= BASE POPULATION =======================

def load_base_population(csv_path: str):
    df = pd.read_csv(csv_path)
    bin_map = {
        "0-4_Years_old": (0, 4), "5-9_Years_old": (5, 9), "10-14_Years_old": (10, 14),
        "15-19_Years_old": (15, 19), "20-24_Years_old": (20, 24), "25-29_Years_old": (25, 29),
        "30-34_Years_old": (30, 34), "35-39_Years_old": (35, 39), "40-44_Years_old": (40, 44),
        "45-49_Years_old": (45, 49), "50-54_Years_old": (50, 54), "55-59_Years_old": (55, 59),
        "60-64_Years_old": (60, 64), "65-69_Years_old": (65, 69), "70-74_Years_old": (70, 74),
        "75-79_Years_old": (75, 79), "80-84_Years_old": (80, 84), "85-89_Years_old": (85, 89),
        "90-94_Years_old": (90, 94), "95-99_Years_old": (95, 99), "100_Years_old_&_over": (100, 100),
    }
    pop = {MALE: np.zeros_like(AGES, dtype=float), FEMALE: np.zeros_like(AGES, dtype=float)}
    for sex in [MALE, FEMALE]:
        row_ = df[df["Sex"] == sex].iloc[0]
        for col, (a0, a1) in bin_map.items():
            total_bin = float(row_[col])
            if a1 == 100:
                # Spread the 100+ population across ages 100+ with exponential decay
                ages_100_plus = np.arange(100, AGE_MAX + 1)
                # Use exponential decay: more people at 100, fewer after
                weights = np.exp(-0.3 * (ages_100_plus - 100))
                weights = weights / weights.sum()  # Normalize to sum to 1
                pop[sex][100:AGE_MAX + 1] += total_bin * weights
            else:
                per_age = total_bin / (a1 - a0 + 1)
                pop[sex][a0:a1 + 1] += per_age
    return pop

# ======================= FERTILITY (PDF-style) =========================

def glg_asfr_shape(ages, mu, sigma, kappa):
    a = np.clip(ages.astype(float), 0, None)
    x = np.clip((a - 10.0) / max(sigma, 1e-6), 1e-6, None)
    kern = np.power(x, max(kappa, 1e-3) - 1.0) * np.exp(-x)
    g = np.exp(-0.5 * ((a - mu) / max(1.0, sigma)) ** 2)
    shape = kern * (0.5 + 0.5 * g)
    mask = (ages >= 15) & (ages <= 49)
    v = np.zeros_like(a)
    v[mask] = shape[mask]
    return _safe_norm(v)

def build_tfr_path(years, table: Dict[int, float]): 
    return interp_from_table(years, table)

def births_from_tfr_pdf(female_pop, tfr, asfr_shape, srb=1.055, calibration=1.0):
    asfr = asfr_shape * tfr
    births_total = float(np.dot(asfr, female_pop)) * calibration
    male_share = srb / (1.0 + srb)
    female_share = 1.0 / (1.0 + srb)
    return births_total * male_share, births_total * female_share

def fertility_module(years, scenario_name, tfr_table_for_scenario):
    tfr_path = build_tfr_path(years, tfr_table_for_scenario)
    mu_path = np.linspace(30.0, 31.5, len(years))
    sigma_path = np.linspace(6.0, 5.5, len(years))
    kappa_path = np.linspace(0.8, 1.0, len(years))
    asfr_shapes = [
        glg_asfr_shape(AGES, mu_path[i], sigma_path[i], kappa_path[i])
        for i in range(len(years))
    ]
    asfr_calibration = PDF_PARAMS[scenario_name]["asfr_calibration"]
    return np.array(asfr_shapes), tfr_path, asfr_calibration

# ======================= MORTALITY (PDF-style) =========================

def make_smooth_mx_from_e0_target(target_e0: float, sex: str) -> np.ndarray:
    a = AGES.astype(float)
    if sex == MALE:
        A0, B, C = 0.0005, 0.00008, 0.095
    else:
        A0, B, C = 0.0004, 0.00007, 0.09
    mx = A0 + B * np.exp(C * a)
    e0 = life_expectancy_from_mx(mx)
    s = max(0.05, target_e0 / max(e0, 1e-6))
    return mx / s

def lee_carter_path(mx_base: np.ndarray, leb_table: Dict[int, float], years: np.ndarray):
    years_arr = years.astype(float)
    ky = np.array(sorted(leb_table.keys()), dtype=float)
    ke = np.array([leb_table[int(k)] for k in ky], dtype=float)
    target_e0 = np.interp(years_arr, ky, ke)
    scales = np.zeros(len(years_arr))
    scales[0] = 0.0

    for i in range(1, len(years_arr)):
        t_e0 = target_e0[i]
        s_lo, s_hi = -2.0, 2.0
        for _ in range(40):
            mid = 0.5 * (s_lo + s_hi)
            e0_mid = life_expectancy_from_mx(mx_base * np.exp(mid))
            if e0_mid < t_e0:
                s_hi = mid
            else:
                s_lo = mid
        scales[i] = 0.5 * (s_lo + s_hi)
    return scales

def mortality_module(years, scenario_name):
    p = PDF_PARAMS[scenario_name]
    leb = p["LEB_table"]
    mx_base = {}
    for sex in [MALE, FEMALE]:
        e0_base = leb[sex][DATA_BASE_YEAR]
        mx0 = p["mx_base"][sex]
        mx_base[sex] = np.asarray(mx0, dtype=float) if mx0 is not None else make_smooth_mx_from_e0_target(e0_base, sex)
    years_arr = np.array(years, dtype=int)
    ktM = lee_carter_path(mx_base[MALE], leb[MALE], years_arr)
    ktF = lee_carter_path(mx_base[FEMALE], leb[FEMALE], years_arr)

    def survival_from_mx(mx):
        qx = 1.0 - np.exp(-mx)
        return np.clip(1.0 - qx, 0.0, 1.0)

    sM_list, sF_list = [], []
    for i in range(len(years) - 1):
        sM_list.append(survival_from_mx(mx_base[MALE] * np.exp(ktM[i])))
        sF_list.append(survival_from_mx(mx_base[FEMALE] * np.exp(ktF[i])))
    return np.array(sM_list), np.array(sF_list)

# ======================= MIGRATION (PDF-style) =========================

def age_profile_two_hump(child_w=0.35, peak=28, spread=6):
    prof = np.zeros_like(AGES, dtype=float)
    for a in AGES:
        ga1 = math.exp(-0.5 * ((a - peak) / spread) ** 2)
        ga2 = child_w * math.exp(-0.5 * ((a - 6) / 5) ** 2)
        prof[a] = ga1 + ga2
    return _safe_norm(prof)

def migration_module(years, scenario_name, sex_split, profile=None):
    p = PDF_PARAMS[scenario_name]
    mig_table = p["MIG_table"]
    profile = profile if profile is not None else age_profile_two_hump()
    years_arr = np.asarray(years, dtype=float)
    if len(years_arr) == 1:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float), profile
    totals = interp_from_table(years_arr, mig_table)
    m_share, f_share = sex_split[MALE], sex_split[FEMALE]
    netM, netF = totals * m_share, totals * f_share
    return netM, netF, profile

# ======================= Projection engine =========================

def run_projection_pdf(
    pop0, years, srb, asfr_shapes, tfr_path, netM_path, netF_path,
    mig_profile, survM_time, survF_time, asfr_calibration=1.0,
):
    T = len(years)
    male = np.zeros((T, AGE_MAX + 1))
    female = np.zeros((T, AGE_MAX + 1))
    male[0, :], female[0, :] = pop0[MALE].copy(), pop0[FEMALE].copy()

    for t in range(1, T):
        sM, sF = survM_time[t - 1], survF_time[t - 1]
        agedM = np.zeros(AGE_MAX + 1)
        agedF = np.zeros(AGE_MAX + 1)
        agedM[1:], agedF[1:] = male[t - 1, :-1] * sM[:-1], female[t - 1, :-1] * sF[:-1]
        agedM[-1] += male[t - 1, -1] * sM[-1]
        agedF[-1] += female[t - 1, -1] * sF[-1]
        agedM += mig_profile * netM_path[t - 1]
        agedF += mig_profile * netF_path[t - 1]
        bM, bF = births_from_tfr_pdf(agedF, tfr_path[t - 1], asfr_shapes[t - 1], srb=srb, calibration=asfr_calibration)
        agedM[0], agedF[0] = bM, bF
        male[t, :], female[t, :] = agedM, agedF

    return {"years": years, "male": male, "female": female, "tfr": tfr_path, "netM": netM_path, "netF": netF_path}

# ======================= Dependency ratios =========================

def dep_ratios(pop):
    y = pop[:, 0:15].sum(axis=1)
    w = pop[:, 15:65].sum(axis=1)
    o = pop[:, 65:].sum(axis=1)
    return y / w, o / w, (y + o) / w

# ======================= BOKEH WIDGETS & SOURCES =========================

title_div = Div(text="""
<h1 style="margin-bottom:0.2rem;">Population Projections for Korea — Interactive Bokeh Dashboard</h1>
<p style="margin-top:0;">
Assumptions use the PDF tables for TFR, life expectancy at birth, and net migration scenario anchors.
Base year fixed at 2025; horizon extended to 2100.
</p>
""")

csv_input = TextInput(title="Base population CSV path", value="korea_population_by_age.csv")
base_year_div = Div(text=f"<b>Base year:</b> {BASE_YEAR}")

last_year_slider = Slider(title="Last year (max 2100)", start=FINAL_YEAR_MIN, end=FINAL_YEAR_MAX, step=1, value=FINAL_YEAR_DEFAULT)

# Scenario radio button group
scenario_buttons = RadioButtonGroup(labels=["Low", "Medium", "High", "Custom"], active=1, width=320)
scenario_buttons.css_classes = ["scenario-radio-group"]

srb_slider = Slider(title="Sex ratio at birth (M/F)", start=1.00, end=1.10, step=0.001, value=1.055)
child_weight_slider = Slider(title="Child hump weight", start=0.0, end=1.0, step=0.05, value=0.35)
ya_peak_age_slider = Slider(title="Young-adult peak age", start=22, end=40, step=1, value=28)
ya_spread_slider = Slider(title="Young-adult spread (σ)", start=3, end=12, step=1, value=6)

base_pop_div = Div(text="<b>Base population (people)</b>: –")
last_pop_div = Div(text="<b>Last year population (people)</b>: –")
tfr_metric_div = Div(text="<b>TFR (base → last)</b>: –")
mig_metric_div = Div(text="<b>Net migration (base, M+F)</b>: –")
scenario_caption_div = Div(text="Active scenario: <b>Medium</b>")
flow_diag_div = Div(text="")
check_2070_div = Div(text="")

initial_scenario = CURRENT_SCENARIO["name"]
initial_tfr_table = TFR_TABLE_EDITABLE[initial_scenario]
# Filter to only show years from BASE_YEAR onwards
initial_tfr_years = [y for y in sorted(initial_tfr_table.keys()) if y >= BASE_YEAR]
initial_tfr_values = [initial_tfr_table[y] for y in initial_tfr_years]

tfr_anchors_source = ColumnDataSource(data={"year": initial_tfr_years, "tfr": initial_tfr_values})

# CustomJS to lock x-axis and prevent data corruption
lock_x_callback = CustomJS(args=dict(source=tfr_anchors_source), code="""
if (source._locking) {
    console.log('Callback already locked, skipping');
    return;
}

// Initialize fixed years on first run
if (!source._fixed_years) {
    console.log('Initializing fixed years:', source.data['year']);
    source._fixed_years = source.data['year'].slice();
    source._fixed_tfr = source.data['tfr'].slice();
    return;
}

const years = source.data['year'];
const tfr = source.data['tfr'];
const fixed_years = source._fixed_years;

console.log('Lock callback triggered');
console.log('Current years:', years);
console.log('Fixed years:', fixed_years);

// If length changed, it's a scenario change - accept and reset
if (years.length !== fixed_years.length) {
    console.log('Length changed - accepting as scenario change');
    source._fixed_years = years.slice();
    source._fixed_tfr = tfr.slice();
    return;
}

// Count how many year values differ
let year_diffs = 0;
for (let i = 0; i < years.length; i++) {
    if (Math.abs(years[i] - fixed_years[i]) > 0.001) {
        year_diffs++;
        console.log('Year diff at index', i, ':', years[i], 'vs', fixed_years[i]);
    }
}

console.log('Total year diffs:', year_diffs);

// If no years changed, just update TFR and return (vertical drag only)
if (year_diffs === 0) {
    console.log('No year changes - vertical drag only');
    source._fixed_tfr = tfr.slice();
    return;
}

// If only 1-2 years changed, it's a user drag - lock x and snap back
if (year_diffs <= 2) {
    console.log('User drag detected - snapping back');
    source._locking = true;
    // Create new data object to force visual update
    source.data = {
        year: fixed_years.slice(),
        tfr: tfr.slice()
    };
    source._locking = false;
    source._fixed_tfr = tfr.slice();
    return;
}

// If many/all years changed, it's likely a scenario change - accept new values
console.log('Many changes - accepting as scenario change');
source._fixed_years = years.slice();
source._fixed_tfr = tfr.slice();
""")
tfr_anchors_source.js_on_change('data', lock_x_callback)

tfr_path_source = ColumnDataSource(data={"year": [], "tfr": []})
tot_pop_source = ColumnDataSource(data={"year": [], "total": []})
mig_source = ColumnDataSource(data={"year": [], "male": [], "female": []})
dep_source = ColumnDataSource(data={"year": [], "young": [], "old": [], "total": []})
age_pyr_source = ColumnDataSource(data={"age": [], "sex": [], "value": [], "population": []})

# ======================= BOKEH FIGURES =========================

tot_pop_fig = figure(
    title="Total Population",
    x_axis_label="Year",
    y_axis_label="Population (millions)",
    height=468,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
tot_pop_fig.line("year", "total", source=tot_pop_source, line_width=2)
tot_pop_fig.yaxis.formatter = NumeralTickFormatter(format="0.0")
tot_pop_hover = HoverTool(tooltips=[("Year", "@year{0}"), ("Population (millions)", "@total{0.00}")], mode="vline")
tot_pop_fig.add_tools(tot_pop_hover)

tfr_fig = figure(
    title="Total Fertility Rate (TFR)",
    x_axis_label="Year",
    y_axis_label="Children per woman",
    height=396,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
tfr_fig.line("year", "tfr", source=tfr_path_source, line_width=2, legend_label="Interpolated path")
tfr_points_renderer = tfr_fig.circle("year", "tfr", source=tfr_anchors_source, size=10, color="orange", legend_label="Anchors (drag)")
tfr_fig.legend.location = "top_right"
tfr_hover = HoverTool(tooltips=[("Year", "@year{0}"), ("TFR", "@tfr{0.000}")], mode="vline")
tfr_fig.add_tools(tfr_hover)

# Add PointDrawTool for dragging anchors
tfr_draw_tool = PointDrawTool(renderers=[tfr_points_renderer], add=False)
tfr_fig.add_tools(tfr_draw_tool)
tfr_fig.toolbar.active_tap = tfr_draw_tool



mig_fig = figure(
    title="Net International Migration (per year)",
    x_axis_label="Year",
    y_axis_label="People",
    height=396,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
mig_fig.line("year", "male", source=mig_source, line_width=2, legend_label="Male")
mig_fig.line("year", "female", source=mig_source, line_width=2, line_dash="dashed", legend_label="Female")
mig_fig.legend.location = "top_right"
mig_hover = HoverTool(tooltips=[("Year", "@year{0}"), ("Male", "@male{0,0}"), ("Female", "@female{0,0}")], mode="vline")
mig_fig.add_tools(mig_hover)

pyramid_fig = figure(
    title="Age Pyramid",
    x_axis_label="Population (people)",
    y_axis_label="Age (years)",
    height=576,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
pyramid_fig.y_range.range_padding = 0.05
pyramid_fig.x_range = Range1d(-1, 1)
pyramid_fig.xaxis.formatter = NumeralTickFormatter(format="0,0")
pyramid_fig.hbar(
    y="age",
    right="value",
    left=0,
    height=0.8,
    source=age_pyr_source,
    fill_color=factor_cmap("sex", palette=Spectral4[0:2], factors=["Male", "Female"]),
)
pyramid_hover = HoverTool(tooltips=[("Age", "@age"), ("Sex", "@sex"), ("Population", "@population{0,0}")])
pyramid_fig.add_tools(pyramid_hover)

pyr_year_slider = Slider(title="Select year (age pyramid)", start=BASE_YEAR, end=FINAL_YEAR_DEFAULT, step=1, value=BASE_YEAR)

dep_fig = figure(
    title="Dependency Ratios",
    x_axis_label="Year",
    y_axis_label="Ratio",
    height=432,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
dep_fig.line("year", "young", source=dep_source, line_width=2, legend_label="Young/Wkg")
dep_fig.line("year", "old", source=dep_source, line_width=2, line_dash="dashed", legend_label="Old/Wkg")
dep_fig.line("year", "total", source=dep_source, line_width=2, line_dash="dotdash", legend_label="Total/Wkg")
dep_fig.legend.location = "top_left"
dep_hover = HoverTool(tooltips=[("Year", "@year{0}"), ("Young/Wkg", "@young{0.000}"), ("Old/Wkg", "@old{0.000}"), ("Total/Wkg", "@total{0.000}")], mode="vline")
dep_fig.add_tools(dep_hover)

# ======================= CALLBACK LOGIC =========================

def _update_pyramid():
    years, male, female = STATE.get("years"), STATE.get("male"), STATE.get("female")
    if years is None or male is None or female is None:
        return

    year_pick = int(pyr_year_slider.value)
    if year_pick < years[0]:
        year_pick = int(years[0])
        pyr_year_slider.value = year_pick
    if year_pick > years[-1]:
        year_pick = int(years[-1])
        pyr_year_slider.value = year_pick

    idx = int(year_pick - years[0])
    pop_m, pop_f = male[idx, :], female[idx, :]
    ages_cat = list(AGES) + list(AGES)
    sex = ["Male"] * len(AGES) + ["Female"] * len(AGES)
    values = np.concatenate([-pop_m, pop_f])
    pop = np.concatenate([pop_m, pop_f])

    age_pyr_source.data = {"age": ages_cat, "sex": sex, "value": values, "population": pop}
    xmax = float(np.max(np.abs(values))) if len(values) > 0 else 1.0
    pyramid_fig.x_range.start = -1.12 * xmax
    pyramid_fig.x_range.end = 1.12 * xmax
    pyramid_fig.title.text = f"Age Pyramid — {year_pick}"

def update_pyramid(attr, old, new):
    _update_pyramid()

def _update_projection():
    try:
        pop0 = load_base_population(csv_input.value)
        flow_msg = f"<p style='color:#7FDB51;'>Base population loaded from {DATA_BASE_YEAR} (calculations include {DATA_BASE_YEAR}-{BASE_YEAR-1} as historical baseline, displayed from {BASE_YEAR}).</p>"
    except Exception as e:
        flow_diag_div.text = f"<p style='color:#FF4136;'>Failed to read CSV: {e}</p>"
        return

    scenario_name = CURRENT_SCENARIO["name"]
    params = PDF_PARAMS[scenario_name]

    last_year = int(last_year_slider.value)
    # Calculate with full years from DATA_BASE_YEAR (2020) for accuracy
    calc_years = np.arange(DATA_BASE_YEAR, last_year + 1)
    # But only display years from BASE_YEAR (2025) onwards on graphs
    display_mask = calc_years >= BASE_YEAR
    years = calc_years  # Use full calculation years for projection

    anchors = tfr_anchors_source.data
    anchor_years = np.array(anchors.get("year", []), dtype=float)
    anchor_tfr = np.array(anchors.get("tfr", []), dtype=float)

    mask = ~np.isnan(anchor_years) & ~np.isnan(anchor_tfr)
    anchor_years, anchor_tfr = anchor_years[mask], anchor_tfr[mask]

    # Start with full default table (includes historical years)
    anchor_table = dict(TFR_TABLE_DEFAULT[scenario_name])

    # Overlay with user-editable anchors (only for BASE_YEAR and later)
    if len(anchor_years) > 0:
        anchor_years = anchor_years.astype(int)
        anchor_years[anchor_years < BASE_YEAR] = BASE_YEAR
        order = np.argsort(anchor_years)
        anchor_years, anchor_tfr = anchor_years[order], anchor_tfr[order]
        # Update only the editable years
        for y, v in zip(anchor_years, anchor_tfr):
            anchor_table[int(y)] = float(v)

    TFR_TABLE_EDITABLE[scenario_name] = dict(anchor_table)

    asfr_shapes, tfr_path, asfr_calibration = fertility_module(years, scenario_name, anchor_table)
    survM_time, survF_time = mortality_module(years, scenario_name)

    child_weight = float(child_weight_slider.value)
    ya_peak = float(ya_peak_age_slider.value)
    ya_spread = float(ya_spread_slider.value)
    mig_profile = age_profile_two_hump(child_weight, ya_peak, ya_spread)

    netM_path, netF_path, mig_profile_used = migration_module(years, scenario_name, params["sex_split"], profile=mig_profile)
    srb = float(srb_slider.value)

    proj = run_projection_pdf(
        pop0, years, srb, asfr_shapes, tfr_path, netM_path, netF_path,
        mig_profile_used, survM_time, survF_time, asfr_calibration=asfr_calibration,
    )

    male, female = proj["male"], proj["female"]
    total = male + female

    STATE["years"], STATE["male"], STATE["female"], STATE["total"], STATE["asfr_shapes"] = years, male, female, total, asfr_shapes

    # Filter data for display (only show BASE_YEAR onwards on graphs)
    display_years = years[display_mask]
    display_male = male[display_mask]
    display_female = female[display_mask]
    display_total = total[display_mask]

    tot_pop_source.data = {"year": display_years, "total": display_total.sum(axis=1) / 1_000_000}
    tfr_path_source.data = {"year": display_years, "tfr": proj["tfr"][display_mask]}
    mig_source.data = {"year": display_years, "male": proj["netM"][display_mask], "female": proj["netF"][display_mask]}

    y_w, o_w, t_w = dep_ratios(total)
    dep_source.data = {"year": display_years, "young": y_w[display_mask], "old": o_w[display_mask], "total": t_w[display_mask]}

    # Pyramid slider should only show display years (BASE_YEAR onwards)
    pyr_year_slider.start, pyr_year_slider.end = int(display_years[0]), int(display_years[-1])
    if pyr_year_slider.value < pyr_year_slider.start or pyr_year_slider.value > pyr_year_slider.end:
        pyr_year_slider.value = int(display_years[0])
    _update_pyramid()

    base_pop_total = float(total[0, :].sum())
    final_pop = float(total[-1, :].sum())

    base_pop_div.text = (
        f"<b>Base population ({DATA_BASE_YEAR}, people)</b>: {base_pop_total:,.0f}<br>"
        f"Male: {male[0, :].sum():,.0f}<br>Female: {female[0, :].sum():,.0f}"
    )
    last_pop_div.text = f"<b>Last year ({display_years[-1]}) population (people)</b>: {final_pop:,.0f}"
    tfr_metric_div.text = f"<b>TFR (base → last)</b>: {proj['tfr'][0]:.2f} → {proj['tfr'][-1]:.2f}"
    mig_metric_div.text = f"<b>Net migration (base, M+F)</b>: {int(proj['netM'][0] + proj['netF'][0]):,}"
    scenario_caption_div.text = f"Active scenario: <b>{scenario_name}</b>"

    if len(years) > 1:
        births_by_year = male[1:, 0] + female[1:, 0]
        total_births = float(births_by_year.sum())
        mig_by_year = proj["netM"][:-1] + proj["netF"][:-1]
        total_migration = float(mig_by_year.sum())
        total_deaths_estimated = base_pop_total + total_births + total_migration - final_pop

        flow_diag_div.text = (
            f"{flow_msg}"
            f"<p><b>Cumulative flows ({BASE_YEAR}-{display_years[-1]}):</b><br>"
            f"Total births: {total_births:,.0f}<br>"
            f"Total deaths (estimated): {total_deaths_estimated:,.0f}<br>"
            f"Total net migration: {total_migration:,.0f}</p>"
            f"<p><b>Final year ({display_years[-1]}) population:</b> {final_pop:,.0f} people<br>"
            f"Accounting check: {base_pop_total:,.0f} + {total_births:,.0f} "
            f"- {total_deaths_estimated:,.0f} + {total_migration:,.0f} = {final_pop:,.0f}</p>"
        )
    else:
        flow_diag_div.text = flow_msg

    if 2070 in years:
        idx_2070 = int(np.where(years == 2070)[0][0])
        model_2070 = float(total[idx_2070, :].sum())
        target_2070 = PDF_TOTAL_2070[scenario_name]
        diff_pct = (model_2070 - target_2070) / target_2070

        if abs(diff_pct) > 0.05:
            check_2070_div.text = (
                f"<p style='color:#FF4136;'>"
                f"⚠️ <b>2070 population deviates from PDF by {diff_pct * 100:.1f}%</b><br>"
                f"Model 2070 total ({scenario_name}): {model_2070 / 1_000_000:.2f}M<br>"
                f"PDF 2070 total ({scenario_name}): {target_2070 / 1_000_000:.2f}M<br>"
                f"This is more than the allowed ±5% tolerance."
                f"</p>"
            )
        else:
            check_2070_div.text = (
                f"<p style='color:#7FDB51;'>"
                f"✅ 2070 population is within ±5% of the PDF:<br>"
                f"Model 2070 total: {model_2070 / 1_000_000:.2f}M<br>"
                f"PDF 2070 total: {target_2070 / 1_000_000:.2f}M "
                f"({diff_pct * 100:.1f}% difference)."
                f"</p>"
            )
    else:
        check_2070_div.text = "<p>No 2070 output in the current horizon, so the PDF sanity check is skipped.</p>"

def update_projection(attr, old, new):
    _update_projection()

def on_scenario_change(attr, old, new):
    """Handle scenario button group change"""
    # Set flag to prevent auto-switch to Custom during scenario change
    STATE["changing_scenario"] = True

    scenario_name = SCENARIOS[new]
    CURRENT_SCENARIO["name"] = scenario_name

    # Update TFR anchors to match the scenario
    if scenario_name in PRESET_SCENARIOS:
        # For preset scenarios, reload from defaults
        tfr_table = dict(TFR_TABLE_DEFAULT[scenario_name])
        TFR_TABLE_EDITABLE[scenario_name] = tfr_table
    else:
        # For Custom, use whatever is currently in EDITABLE
        tfr_table = TFR_TABLE_EDITABLE.get(scenario_name, TFR_TABLE_DEFAULT["Medium"])

    # Filter to only show years from BASE_YEAR onwards
    years_list = [y for y in sorted(tfr_table.keys()) if y >= BASE_YEAR]
    tfr_list = [tfr_table[y] for y in years_list]

    # Update the data source
    tfr_anchors_source.data = {"year": years_list, "tfr": tfr_list}

    # Run projection
    _update_projection()

    # Clear flag after scenario change is complete
    STATE["changing_scenario"] = False

def on_tfr_manual_change(attr, old, new):
    """Auto-switch to Custom when TFR anchors are manually dragged"""
    # Skip during initialization to prevent auto-switch on page load
    if STATE.get("initializing", False):
        return

    # Skip if we're already processing a scenario change
    if STATE.get("changing_scenario", False):
        return

    # Check if the TFR values actually changed (not just initialization)
    old_tfr = old.get("tfr", []) if old else []
    new_tfr = new.get("tfr", [])

    # Must have both old and new values to compare
    if len(old_tfr) > 0 and len(new_tfr) > 0:
        # Check if the arrays are the same length (scenario change vs drag)
        if len(old_tfr) != len(new_tfr):
            # Length changed - this is a scenario change, not a drag
            return

        # Check if values are different (with tolerance for floating point)
        if not np.allclose(old_tfr, new_tfr, atol=1e-3):
            # Values changed - update projection
            # If in preset scenario, switch to Custom first
            if CURRENT_SCENARIO["name"] in PRESET_SCENARIOS:
                # Save the current edited values to Custom
                years = new.get("year", [])
                TFR_TABLE_EDITABLE["Custom"] = {int(y): float(v) for y, v in zip(years, new_tfr)}

                # Switch to Custom scenario (index 3 in SCENARIOS list)
                CURRENT_SCENARIO["name"] = "Custom"
                scenario_buttons.active = 3

            # Always run projection update when TFR values change
            _update_projection()

# ======================= WIRING CALLBACKS =========================

for w in [csv_input, last_year_slider, srb_slider, child_weight_slider, ya_peak_age_slider, ya_spread_slider]:
    w.on_change("value", update_projection)

# Connect scenario button group
scenario_buttons.on_change("active", on_scenario_change)

pyr_year_slider.on_change("value", update_pyramid)
tfr_anchors_source.on_change("data", on_tfr_manual_change)

# Initial projection
_update_projection()

# Mark initialization as complete to enable auto-switch to Custom
STATE["initializing"] = False

# ======================= LAYOUT =========================

sidebar = column(
    base_year_div,
    last_year_slider,
    Div(text="<hr><h2>Scenarios</h2>"),
    Div(text="<p style='margin-bottom: 12px;'>Select a scenario or customize TFR values.</p>"),
    scenario_buttons,
    Div(text="<p style='font-size: 0.9rem; margin-top: 12px; color: var(--dashboard-text-muted);'>💡 Dragging TFR anchors auto-switches to Custom</p>"),
    Div(text="<hr><h2>Fertility</h2>"),
    srb_slider,
    Div(text="<hr><h2>Net migration</h2>"),
    Div(text="<p>Totals follow PDF scenario summary; age pattern uses a configurable two-hump profile.</p>"),
    child_weight_slider,
    ya_peak_age_slider,
    ya_spread_slider,
    sizing_mode="fixed",
    width=360,
)
sidebar.css_classes = ["dashboard-sidebar"]

metrics_row = row(
    column(base_pop_div),
    column(last_pop_div),
    column(tfr_metric_div),
    column(mig_metric_div),
    sizing_mode="stretch_width",
)

# TFR figure
tfr_editor_block = column(
    Div(text="<h3>Total Fertility Rate (TFR)</h3>"),
    Div(text="<p>Drag the orange anchor points to adjust TFR values. The year positions are locked—if you drag horizontally, the points will snap back to their fixed years. The interpolated TFR path and projections update automatically.</p>"),
    tfr_fig,
    sizing_mode="stretch_width",
)

# Create 2-column rows for graphs
charts_row_1 = row(
    column(Div(text="<h3>Total population over time</h3>"), tot_pop_fig, sizing_mode="stretch_width"),
    column(Div(text="<h3>Dependency ratios</h3>"), dep_fig, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
)

charts_row_2 = row(
    column(tfr_editor_block, sizing_mode="stretch_width"),
    column(Div(text="<h3>Net International Migration (per year)</h3>"), mig_fig, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
)

# Wrap Age Pyramid in a container
pyramid_container = column(
    Div(text="<h2>Age pyramid (male left, female right)</h2>"),
    pyr_year_slider,
    pyramid_fig,
    sizing_mode="stretch_width",
)

main = column(
    title_div,
    metrics_row,
    scenario_caption_div,
    Div(text="<hr>"),
    charts_row_1,
    Div(text="<hr>"),
    Div(text="<h2>Assumptions (paths)</h2>"),
    charts_row_2,
    Div(text="<hr>"),
    pyramid_container,
    Div(text="<hr>"),
    Div(text="<h2>🔍 DIAGNOSTIC: Population Flow Analysis</h2>"),
    flow_diag_div,
    check_2070_div,
    Div(text="<hr>"),
    sizing_mode="stretch_width",     # FIXED (was stretch_both)
)
main.css_classes = ["dashboard-main"]
main.spacing = 12

root_layout = row(
    sidebar,
    main,
    sizing_mode="stretch_width",     # FIXED (was stretch_both)
)
root_layout.spacing = 18
root_layout.css_classes = ["dashboard-root"]
root_layout.stylesheets = [dashboard_stylesheet]

curdoc().add_root(root_layout)