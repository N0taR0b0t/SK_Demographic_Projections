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
    CustomJSTickFormatter, Span,
)
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral4

# ======================= THEME & CONSTANTS =======================

curdoc().title = "Population Projections for Korea - Bokeh Dashboard"
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
  padding: 16px 18px 16px 12px;
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
EDITABLE_YEARS = [BASE_YEAR, 2035, 2070, 2100]

SCENARIOS = ["Low", "Medium", "High", "Custom"]
PRESET_SCENARIOS = ["Low", "Medium", "High"]

TEST_ADDITIONAL = 500_000

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

# Track current scenario and last preset used (for Custom scenario reference)
CURRENT_SCENARIO = {"name": "Medium"}
LAST_PRESET_SCENARIO = "Medium"  # Track which preset was active before switching to Custom

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

    # Add TEST_ADDITIONAL people equally spread across ages (equally split male/female)
    if TEST_ADDITIONAL > 0:
        age_range = np.arange(6, 26) #Ages
        per_age_per_sex = TEST_ADDITIONAL / (2.0 * len(age_range))  # Split by sex, then by age
        for age in age_range:
            pop[MALE][age] += per_age_per_sex
            pop[FEMALE][age] += per_age_per_sex

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
<h1 style="margin-bottom:0.2rem;">Interactive Population Projections for South Korea</h1>
<p style="margin-top:0;">
</p>
""")

csv_input = TextInput(title="Base population CSV path", value="korea_population_by_age.csv")
base_year_div = Div(text=f"<b>Base year:</b> {BASE_YEAR}")

last_year_slider = Slider(title="Last year (max 2100)", start=FINAL_YEAR_MIN, end=FINAL_YEAR_MAX, step=1, value=FINAL_YEAR_DEFAULT)

# Scenario radio button group
scenario_buttons = RadioButtonGroup(labels=["Low", "Medium", "High", "Custom"], active=1, width=240)
scenario_buttons.css_classes = ["scenario-radio-group"]

srb_slider = Slider(title="Sex ratio at birth (M/F)", start=1.00, end=1.10, step=0.001, value=1.055)
#child_weight_slider = Slider(title="Child hump weight", start=0.0, end=1.0, step=0.05, value=0.35)
#ya_peak_age_slider = Slider(title="Young-adult peak age", start=22, end=40, step=1, value=28)
#ya_spread_slider = Slider(title="Young-adult spread (σ)", start=3, end=12, step=1, value=6)

base_pop_div = Div(text="<b>Base population (people)</b>: –")
last_pop_div = Div(text="<b>Last year population (people)</b>: –")
tfr_metric_div = Div(text="<b>TFR (base → last)</b>: –")
mig_metric_div = Div(text="<b>Net migration (base, M+F)</b>: –")
scenario_caption_div = Div(text="Active scenario: <b>Medium</b>")
flow_diag_div = Div(text="")
check_2070_div = Div(text="")

initial_scenario = CURRENT_SCENARIO["name"]
initial_tfr_table = TFR_TABLE_EDITABLE[initial_scenario]

# Always use the 4 editable years; fill values by interpolation
initial_tfr_years = EDITABLE_YEARS.copy()
initial_tfr_values = list(interp_from_table(initial_tfr_years, initial_tfr_table))
tfr_anchors_source = ColumnDataSource(data={"year": initial_tfr_years, "tfr": initial_tfr_values})

# CustomJS to lock x-axis and prevent data corruption
lock_x_callback = CustomJS(args=dict(source=tfr_anchors_source), code="""
// Prevent recursive calls
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

// SCENARIO 1: Check for scenario change flag in data
// Python on_scenario_change sets _scenario_change: [True] in the data
if (source.data['_scenario_change'] && source.data['_scenario_change'][0] === true) {
    console.log('Scenario change flag detected - accepting new years');
    source._fixed_years = years.slice();
    source._fixed_tfr = tfr.slice();
    // Remove the flag so normal drag behavior resumes
    source._locking = true;
    source.data = {
        year: years.slice(),
        tfr: tfr.slice()
    };
    source._locking = false;
    return;
}

// SCENARIO 2: Vertical drag only (TFR changed, years unchanged)
let any_year_changed = false;
for (let i = 0; i < years.length; i++) {
    if (Math.abs(years[i] - fixed_years[i]) > 0.001) {
        any_year_changed = true;
        break;
    }
}

if (!any_year_changed) {
    console.log('Vertical drag only - updating TFR values');
    source._fixed_tfr = tfr.slice();
    // Data is already correct (years locked, tfr updated by drag)
    // Python callback will handle the update automatically
    return;
}

// SCENARIO 3: User attempted horizontal/diagonal drag - BLOCK IT
console.log('Horizontal drag detected - snapping back to fixed years');
source._locking = true;
// Snap x back to fixed years but keep the new TFR values
// Flag array must match length of data arrays (Bokeh requirement)
const flag_array = new Array(tfr.length).fill(true);
source.data = {
    year: fixed_years.slice(),
    tfr: tfr.slice(),
    _user_drag: flag_array  // Flag to tell Python this is a user drag that needs updating
};
source._locking = false;
source._fixed_tfr = tfr.slice();
""")
tfr_anchors_source.js_on_change('data', lock_x_callback)

tfr_path_source = ColumnDataSource(data={"year": [], "tfr": []})
tot_pop_source = ColumnDataSource(data={"year": [], "total": []})
mig_source = ColumnDataSource(data={"year": [], "male": [], "female": []})
# Separate sources for Low, Medium, High preset overlays
mig_low_source = ColumnDataSource(data={"year": [], "total": []})
mig_medium_source = ColumnDataSource(data={"year": [], "total": []})
mig_high_source = ColumnDataSource(data={"year": [], "total": []})
dep_source = ColumnDataSource(data={"year": [], "young": [], "old": [], "total": []})
age_pyr_source = ColumnDataSource(data={"age": [], "sex": [], "value": [], "population": []})

# ======================= BOKEH FIGURES =========================

tot_pop_fig = figure(
    title="Total Population",
    x_axis_label="Year",
    y_axis_label="Population (millions)",
    height=405,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
tot_pop_fig.line("year", "total", source=tot_pop_source, line_width=2)
tot_pop_fig.yaxis.formatter = NumeralTickFormatter(format="0.0")
tot_pop_hover = HoverTool(tooltips=[("Year", "@year{0}"), ("Population (millions)", "@total{0.00}")], mode="vline")
tot_pop_fig.add_tools(tot_pop_hover)

tfr_fig = figure(
    x_axis_label="Year",
    y_axis_label="Children per woman",
    height=380,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    y_range=Range1d(0.25, 1.5),
)
tfr_fig.line("year", "tfr", source=tfr_path_source, line_width=2)
# Add invisible larger circles as hit targets for easier dragging
tfr_hit_renderer = tfr_fig.scatter("year", "tfr", source=tfr_anchors_source, size=25, color="orange", alpha=0.0)
# Add visible smaller circles on top
tfr_points_renderer = tfr_fig.scatter("year", "tfr", source=tfr_anchors_source, size=10, color="orange")
tfr_hover = HoverTool(tooltips=[("Year", "@year{0}"), ("TFR", "@tfr{0.000}")], mode="vline")
tfr_fig.add_tools(tfr_hover)

# Add PointDrawTool for dragging anchors - only use hit renderer to avoid double updates
tfr_draw_tool = PointDrawTool(renderers=[tfr_hit_renderer], add=False)
tfr_fig.add_tools(tfr_draw_tool)
tfr_fig.toolbar.active_tap = tfr_draw_tool



mig_fig = figure(
    x_axis_label="Year",
    y_axis_label="People",
    height=380,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    y_range=Range1d(-100000, 200000),
)
# Add preset scenario lines (Low, Medium, High) - these will be styled dynamically
mig_low_line = mig_fig.line("year", "total", source=mig_low_source, line_width=2, line_color="gray", line_alpha=0.3, legend_label="Low")
mig_medium_line = mig_fig.line("year", "total", source=mig_medium_source, line_width=2, line_color="gray", line_alpha=0.3, legend_label="Medium")
mig_high_line = mig_fig.line("year", "total", source=mig_high_source, line_width=2, line_color="gray", line_alpha=0.3, legend_label="High")

mig_fig.legend.location = "top_right"
mig_fig.yaxis.formatter = NumeralTickFormatter(format="0,0")
mig_hover = HoverTool(tooltips=[("Year", "@year{0}"), ("Total", "@total{0,0}")], mode="vline")
mig_fig.add_tools(mig_hover)

pyramid_fig = figure(
    title="Age Pyramid",
    x_axis_label="Male        Female",
    y_axis_label="Age (years)",
    height=495,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
pyramid_fig.y_range.range_padding = 0.05
pyramid_fig.x_range = Range1d(-1, 1)

# Custom tick formatter to show absolute values (no negative signs)
pyramid_fig.xaxis.formatter = CustomJSTickFormatter(code="""
    return Math.abs(tick).toLocaleString();
""")

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
pyr_play_button = Button(label="▶ Play", button_type="success", width=100)

# Track animation state
ANIMATION_STATE = {"playing": False, "callback_id": None}

dep_fig = figure(
    title="Dependency Ratios",
    x_axis_label="Year",
    y_axis_label="Ratio",
    height=389,
    sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
dep_fig.line("year", "young", source=dep_source, line_width=2, legend_label="Young/Wkg")
dep_fig.line("year", "old", source=dep_source, line_width=2, line_dash="dashed", legend_label="Old/Wkg")
dep_fig.line("year", "total", source=dep_source, line_width=2, line_dash="dotdash", legend_label="Total/Wkg")
dep_fig.legend.location = "top_left"
# ======================= YEAR INDICATOR (for animation) =========================

# Vertical line spans for each time-series figure
year_span_tot_pop = Span(location=BASE_YEAR, dimension='height', line_color='white', line_width=2, line_alpha=0.8, visible=False)
year_span_tfr = Span(location=BASE_YEAR, dimension='height', line_color='white', line_width=2, line_alpha=0.8, visible=False)
year_span_mig = Span(location=BASE_YEAR, dimension='height', line_color='white', line_width=2, line_alpha=0.8, visible=False)
year_span_dep = Span(location=BASE_YEAR, dimension='height', line_color='white', line_width=2, line_alpha=0.8, visible=False)

# Add spans to figures
tot_pop_fig.add_layout(year_span_tot_pop)
tfr_fig.add_layout(year_span_tfr)
mig_fig.add_layout(year_span_mig)
dep_fig.add_layout(year_span_dep)

# Data sources for intersection dots
year_dot_tot_pop_source = ColumnDataSource(data={"x": [], "y": []})
year_dot_tfr_source = ColumnDataSource(data={"x": [], "y": []})
year_dot_mig_male_source = ColumnDataSource(data={"x": [], "y": []})
year_dot_mig_female_source = ColumnDataSource(data={"x": [], "y": []})
year_dot_dep_young_source = ColumnDataSource(data={"x": [], "y": []})
year_dot_dep_old_source = ColumnDataSource(data={"x": [], "y": []})
year_dot_dep_total_source = ColumnDataSource(data={"x": [], "y": []})

# Add circle glyphs for intersection dots (store renderers for visibility control)
year_dot_tot_pop_renderer = tot_pop_fig.scatter("x", "y", source=year_dot_tot_pop_source, size=8, color="white", line_color="black", line_width=1)
year_dot_tfr_renderer = tfr_fig.scatter("x", "y", source=year_dot_tfr_source, size=8, color="white", line_color="black", line_width=1)
year_dot_mig_male_renderer = mig_fig.scatter("x", "y", source=year_dot_mig_male_source, size=8, color="white", line_color="black", line_width=1)
year_dot_mig_female_renderer = mig_fig.scatter("x", "y", source=year_dot_mig_female_source, size=8, color="white", line_color="black", line_width=1)
year_dot_dep_young_renderer = dep_fig.scatter("x", "y", source=year_dot_dep_young_source, size=8, color="white", line_color="black", line_width=1)
year_dot_dep_old_renderer = dep_fig.scatter("x", "y", source=year_dot_dep_old_source, size=8, color="white", line_color="black", line_width=1)
year_dot_dep_total_renderer = dep_fig.scatter("x", "y", source=year_dot_dep_total_source, size=8, color="white", line_color="black", line_width=1)

# Initially hide all renderers
year_dot_tot_pop_renderer.visible = False
year_dot_tfr_renderer.visible = False
year_dot_mig_male_renderer.visible = False
year_dot_mig_female_renderer.visible = False
year_dot_dep_young_renderer.visible = False
year_dot_dep_old_renderer.visible = False
year_dot_dep_total_renderer.visible = False

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

    # Use positive values for both male and female, store negatives only for positioning
    values_display = np.concatenate([-pop_m, pop_f])  # For positioning (male left, female right)
    pop = np.concatenate([pop_m, pop_f])  # For display (both positive)

    age_pyr_source.data = {"age": ages_cat, "sex": sex, "value": values_display, "population": pop}

    # Calculate max value but ensure it's at least 400,000 per side (don't zoom in tighter than this)
    xmax = float(np.max(np.abs(values_display))) if len(values_display) > 0 else 1.0
    xmax = max(xmax, 400000)  # Ensure minimum zoom level of 400,000 (prevents zooming in too far)

    pyramid_fig.x_range.start = -1.12 * xmax
    pyramid_fig.x_range.end = 1.12 * xmax
    pyramid_fig.title.text = f"Age Pyramid - {year_pick}"

def update_pyramid(attr, old, new):
    _update_pyramid()

def _update_projection():
    try:
        pop0 = load_base_population(csv_input.value)
        #flow_msg = f"<p style='color:#7FDB51;'>Base population loaded from {DATA_BASE_YEAR} (calculations include {DATA_BASE_YEAR}-{BASE_YEAR-1} as historical baseline, displayed from {BASE_YEAR}).</p>"
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

    # Start with full default table (keeps historical years intact)
    anchor_table = dict(TFR_TABLE_DEFAULT[scenario_name])

    # Overlay with user-editable anchors (restricted to EDITABLE_YEARS)
    if len(anchor_years) > 0:
        anchor_years = anchor_years.astype(int)
        order = np.argsort(anchor_years)
        anchor_years, anchor_tfr = anchor_years[order], anchor_tfr[order]
        for y, v in zip(anchor_years, anchor_tfr):
            y_int = int(y)
            if y_int in EDITABLE_YEARS:
                anchor_table[y_int] = float(v)

    # For projection years >= BASE_YEAR, keep ONLY the 4 editable years,
    # so the path becomes three straight line segments between them.
    for y in list(anchor_table.keys()):
        if y >= BASE_YEAR and y not in EDITABLE_YEARS:
            del anchor_table[y]

    TFR_TABLE_EDITABLE[scenario_name] = dict(anchor_table)


    asfr_shapes, tfr_path, asfr_calibration = fertility_module(years, scenario_name, anchor_table)
    survM_time, survF_time = mortality_module(years, scenario_name)

    #child_weight = float(child_weight_slider.value)
    child_weight = float(0.35)
    ya_peak = float(28)
    ya_spread = float(6)
    mig_profile = age_profile_two_hump(child_weight, ya_peak, ya_spread)

    # For Custom scenario, use migration from the last active preset scenario
    migration_scenario = LAST_PRESET_SCENARIO if scenario_name == "Custom" else scenario_name
    migration_params = PDF_PARAMS[migration_scenario]
    netM_path, netF_path, mig_profile_used = migration_module(years, migration_scenario, migration_params["sex_split"], profile=mig_profile)
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

    # Calculate migration for all three preset scenarios for overlay lines
    # (mig_source is no longer used - we display the three preset lines instead)
    for preset_name in ["Low", "Medium", "High"]:
        preset_params = PDF_PARAMS[preset_name]
        netM_preset, netF_preset, _ = migration_module(years, preset_name, preset_params["sex_split"], profile=mig_profile)
        mig_total_preset = netM_preset[display_mask] + netF_preset[display_mask]

        if preset_name == "Low":
            mig_low_source.data = {"year": display_years, "total": mig_total_preset}
        elif preset_name == "Medium":
            mig_medium_source.data = {"year": display_years, "total": mig_total_preset}
        elif preset_name == "High":
            mig_high_source.data = {"year": display_years, "total": mig_total_preset}

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
            #f"{flow_msg}"
            f"<p><b>Cumulative flows ({BASE_YEAR}-{display_years[-1]}):</b><br>"
            f"Total births: {total_births:,.0f}<br>"
            f"Total deaths (estimated): {total_deaths_estimated:,.0f}<br>"
            f"Total net migration: {total_migration:,.0f}</p>"
            f"<p><b>Final year ({display_years[-1]}) population:</b> {final_pop:,.0f} people<br>"
            #f"Accounting check: {base_pop_total:,.0f} + {total_births:,.0f} "
            f"- {total_deaths_estimated:,.0f} + {total_migration:,.0f} = {final_pop:,.0f}</p>"
        )
    else:
        flow_diag_div.text = flow_msg

    # Only show 3% verification for preset scenarios (not Custom)
    if scenario_name == "Custom":
        check_2070_div.text = (
            f"<p style='color:#FFFFFF;'>"
            f"Custom scenario, skipping population check<br>"
        )
    elif 2070 in years:
        idx_2070 = int(np.where(years == 2070)[0][0])
        model_2070 = float(total[idx_2070, :].sum())
        target_2070 = PDF_TOTAL_2070[scenario_name]
        diff_pct = (model_2070 - target_2070) / target_2070

        if abs(diff_pct) > 0.03: #3%
            check_2070_div.text = (
                f"<p style='color:#FF4136;'>"
                f"<b>2070 population deviates from PDF by {diff_pct * 100:.1f}%</b><br>"
                f"Model 2070 total ({scenario_name}): {model_2070 / 1_000_000:.2f}M<br>"
                f"2070 total ({scenario_name}): {target_2070 / 1_000_000:.2f}M<br>"
                f"</p>"
            )
        else:
            check_2070_div.text = (
                f"<p style='color:#7FDB51;'>"
                f"2070 population is within ±3% of offical projections:<br>"
                f"Model 2070 total: {model_2070 / 1_000_000:.2f}M<br>"
                f"2070 total: {target_2070 / 1_000_000:.2f}M "
                f"({diff_pct * 100:.1f}% difference)."
                f"</p>"
            )

    # Update migration line colors based on active scenario
    update_migration_line_colors()

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
        # Track which preset scenario is being selected
        global LAST_PRESET_SCENARIO
        LAST_PRESET_SCENARIO = scenario_name

        # For preset scenarios, reload from defaults
        tfr_table = dict(TFR_TABLE_DEFAULT[scenario_name])
        TFR_TABLE_EDITABLE[scenario_name] = tfr_table
        # Reset SRB slider to default value for preset scenarios
        srb_slider.value = 1.055
    else:
        # For Custom, use whatever is currently in EDITABLE
        tfr_table = TFR_TABLE_EDITABLE.get(scenario_name, TFR_TABLE_DEFAULT["Medium"])

    # Always create anchors at the 4 editable years, using interpolation
    years_list = EDITABLE_YEARS.copy()
    tfr_list = list(interp_from_table(years_list, tfr_table))

    # Mark this update as a scenario change by adding a flag to the data itself
    # The JS callback will check for this flag and allow x-axis changes
    # Flag array must match length of data arrays (Bokeh requirement)
    tfr_anchors_source.data = {
        "year": years_list,
        "tfr": tfr_list,
        "_scenario_change": [True] * len(years_list)  # Flag to tell JS this is intentional
    }

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

    # Skip if this is a scenario change (check for the flag)
    if new.get("_scenario_change"):
        return

    # Check for user drag flag from JS (happens after snap-back)
    is_user_drag = new.get("_user_drag", [False])[0] if new.get("_user_drag") else False

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
        # OR if this is explicitly flagged as a user drag
        if is_user_drag or not np.allclose(old_tfr, new_tfr, atol=1e-3):
            # Values changed - update projection
            # If in preset scenario, switch to Custom first
            if CURRENT_SCENARIO["name"] in PRESET_SCENARIOS:
                # Save the current edited values to Custom
                years = new.get("year", [])
                # Filter out the flag keys when saving
                clean_tfr = [v for v in new_tfr]
                TFR_TABLE_EDITABLE["Custom"] = {int(y): float(v) for y, v in zip(years, clean_tfr)}

                # Track which preset we're switching from
                global LAST_PRESET_SCENARIO
                LAST_PRESET_SCENARIO = CURRENT_SCENARIO["name"]

                # Switch to Custom scenario (index 3 in SCENARIOS list)
                CURRENT_SCENARIO["name"] = "Custom"
                scenario_buttons.active = 3

            # Always run projection update when TFR values change
            _update_projection()

def update_migration_line_colors():
    """Update migration line colors based on active scenario"""
    scenario_name = CURRENT_SCENARIO["name"]

    # For Custom scenario, use the last active preset scenario's migration
    if scenario_name == "Custom":
        active_scenario = LAST_PRESET_SCENARIO
    else:
        active_scenario = scenario_name

    # Define colors for active and inactive lines
    active_color = "#1f77b4"  # Bokeh blue
    inactive_color = "gray"
    active_alpha = 1.0
    inactive_alpha = 0.3

    # Update Low line
    if active_scenario == "Low":
        mig_low_line.glyph.line_color = active_color
        mig_low_line.glyph.line_alpha = active_alpha
        mig_low_line.glyph.line_width = 2
    else:
        mig_low_line.glyph.line_color = inactive_color
        mig_low_line.glyph.line_alpha = inactive_alpha
        mig_low_line.glyph.line_width = 2

    # Update Medium line
    if active_scenario == "Medium":
        mig_medium_line.glyph.line_color = active_color
        mig_medium_line.glyph.line_alpha = active_alpha
        mig_medium_line.glyph.line_width = 2
    else:
        mig_medium_line.glyph.line_color = inactive_color
        mig_medium_line.glyph.line_alpha = inactive_alpha
        mig_medium_line.glyph.line_width = 2

    # Update High line
    if active_scenario == "High":
        mig_high_line.glyph.line_color = active_color
        mig_high_line.glyph.line_alpha = active_alpha
        mig_high_line.glyph.line_width = 2
    else:
        mig_high_line.glyph.line_color = inactive_color
        mig_high_line.glyph.line_alpha = inactive_alpha
        mig_high_line.glyph.line_width = 2

def update_year_indicators(year):
    """Update the year indicator line and dots to show current animation year"""
    # Update span locations
    year_span_tot_pop.location = year
    year_span_tfr.location = year
    year_span_mig.location = year
    year_span_dep.location = year

    # Total population dot - use data source years to find index
    if len(tot_pop_source.data["year"]) > 0:
        display_years = np.array(tot_pop_source.data["year"])
        if year in display_years:
            idx = int(np.where(display_years == year)[0][0])
            total_pop = tot_pop_source.data["total"][idx]
            year_dot_tot_pop_source.data = {"x": [year], "y": [total_pop]}

    # TFR dot
    if len(tfr_path_source.data["year"]) > 0:
        display_years = np.array(tfr_path_source.data["year"])
        if year in display_years:
            idx = int(np.where(display_years == year)[0][0])
            tfr_val = tfr_path_source.data["tfr"][idx]
            year_dot_tfr_source.data = {"x": [year], "y": [tfr_val]}

    # Migration dot - show on the active preset line
    scenario_name = CURRENT_SCENARIO["name"]
    active_mig_scenario = LAST_PRESET_SCENARIO if scenario_name == "Custom" else scenario_name

    # Select the appropriate data source based on active scenario
    if active_mig_scenario == "Low":
        active_mig_source = mig_low_source
    elif active_mig_scenario == "Medium":
        active_mig_source = mig_medium_source
    elif active_mig_scenario == "High":
        active_mig_source = mig_high_source
    else:
        active_mig_source = mig_medium_source  # Fallback to Medium

    if len(active_mig_source.data["year"]) > 0:
        display_years = np.array(active_mig_source.data["year"])
        if year in display_years:
            idx = int(np.where(display_years == year)[0][0])
            mig_total = active_mig_source.data["total"][idx]
            year_dot_mig_male_source.data = {"x": [year], "y": [mig_total]}

    # Dependency ratio dots
    if len(dep_source.data["year"]) > 0:
        display_years = np.array(dep_source.data["year"])
        if year in display_years:
            idx = int(np.where(display_years == year)[0][0])
            dep_young = dep_source.data["young"][idx]
            dep_old = dep_source.data["old"][idx]
            dep_total = dep_source.data["total"][idx]
            year_dot_dep_young_source.data = {"x": [year], "y": [dep_young]}
            year_dot_dep_old_source.data = {"x": [year], "y": [dep_old]}
            year_dot_dep_total_source.data = {"x": [year], "y": [dep_total]}

def animate_pyramid():
    """Advance the pyramid year by 1 each time this is called"""
    current_year = int(pyr_year_slider.value)
    end_year = int(pyr_year_slider.end)

    if current_year < end_year:
        pyr_year_slider.value = current_year + 1
        # Update pyramid and year indicators
        _update_pyramid()
        update_year_indicators(current_year + 1)
    else:
        # Reached the end, stop animation
        stop_animation()

def start_animation():
    """Start the pyramid animation"""
    ANIMATION_STATE["playing"] = True
    pyr_play_button.label = "⏸ Pause"
    pyr_play_button.button_type = "warning"

    # Show year indicators
    year_span_tot_pop.visible = True
    year_span_tfr.visible = True
    year_span_mig.visible = True
    year_span_dep.visible = True

    year_dot_tot_pop_renderer.visible = True
    year_dot_tfr_renderer.visible = True
    year_dot_mig_male_renderer.visible = True  # Now showing total migration
    year_dot_dep_young_renderer.visible = True
    year_dot_dep_old_renderer.visible = True
    year_dot_dep_total_renderer.visible = True

    # Initialize indicators to current year
    update_year_indicators(int(pyr_year_slider.value))

    # Add periodic callback
    ANIMATION_STATE["callback_id"] = curdoc().add_periodic_callback(animate_pyramid, 140) #ms of delay

def stop_animation():
    """Stop the pyramid animation"""
    ANIMATION_STATE["playing"] = False
    pyr_play_button.label = "▶ Play"
    pyr_play_button.button_type = "success"

    # Hide year indicators
    year_span_tot_pop.visible = False
    year_span_tfr.visible = False
    year_span_mig.visible = False
    year_span_dep.visible = False

    year_dot_tot_pop_renderer.visible = False
    year_dot_tfr_renderer.visible = False
    year_dot_mig_male_renderer.visible = False  # Now showing total migration
    year_dot_dep_young_renderer.visible = False
    year_dot_dep_old_renderer.visible = False
    year_dot_dep_total_renderer.visible = False

    # Remove periodic callback if it exists
    if ANIMATION_STATE["callback_id"] is not None:
        curdoc().remove_periodic_callback(ANIMATION_STATE["callback_id"])
        ANIMATION_STATE["callback_id"] = None

def toggle_play_pause():
    """Toggle between play and pause states"""
    if ANIMATION_STATE["playing"]:
        stop_animation()
    else:
        # If at the end (year 100 or max year), reset to start
        current_year = int(pyr_year_slider.value)
        end_year = int(pyr_year_slider.end)

        if current_year >= end_year:
            pyr_year_slider.value = int(pyr_year_slider.start)

        start_animation()

def on_srb_change(attr, old, new):
    """Handle SRB slider change - auto-switch to Custom scenario"""
    # Skip during initialization
    if STATE.get("initializing", False):
        return

    # If in a preset scenario, switch to Custom
    if CURRENT_SCENARIO["name"] in PRESET_SCENARIOS:
        # Track which preset we're switching from
        global LAST_PRESET_SCENARIO
        LAST_PRESET_SCENARIO = CURRENT_SCENARIO["name"]

        CURRENT_SCENARIO["name"] = "Custom"
        scenario_buttons.active = 3

    # Update projection
    _update_projection()

# ======================= WIRING CALLBACKS =========================

# TextInput triggers on value (when user presses enter or loses focus)
csv_input.on_change("value", update_projection)

# Last year slider triggers on value_throttled (only when user releases)
last_year_slider.on_change("value_throttled", update_projection)

# SRB slider triggers on value_throttled and auto-switches to Custom
srb_slider.on_change("value_throttled", on_srb_change)

# Connect scenario button group
scenario_buttons.on_change("active", on_scenario_change)

pyr_year_slider.on_change("value_throttled", update_pyramid)
tfr_anchors_source.on_change("data", on_tfr_manual_change)

# Connect play/pause button
pyr_play_button.on_click(toggle_play_pause)

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
    Div(text="<p style='font-size: 0.9rem; margin-top: 12px; color: var(--dashboard-text-muted);'></p>"),
    Div(text="<hr><h2>Fertility</h2>"),
    srb_slider,
    Div(text="<hr><h2>Model Validity</h2>"),
    #flow_diag_div,
    check_2070_div,
    sizing_mode="fixed",
    width=270,
    height=600,
)
sidebar.css_classes = ["dashboard-sidebar"]

#metrics_row = row(
#    column(base_pop_div),
#    column(last_pop_div),
#    column(tfr_metric_div),
#    column(mig_metric_div),
#    sizing_mode="stretch_width",
#)

# TFR figure
tfr_editor_block = column(
    Div(text="<h3>Total Fertility Rate (TFR)</h3>"),
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
    column(Div(text="<h3>Net International Migration</h3>"), mig_fig, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
)

# Wrap Age Pyramid in a container
pyramid_controls = row(pyr_year_slider, pyr_play_button, sizing_mode="stretch_width")
pyramid_container = column(
    Div(text="<h2>Age pyramid (male left, female right)</h2>"),
    pyramid_controls,
    pyramid_fig,
    sizing_mode="stretch_width",
)

main = column(
    title_div,
    #metrics_row,
    scenario_caption_div,
    charts_row_2,
    #Div(text="<br>"),
    charts_row_1,
    #Div(text="<br>"),
    pyramid_container,
    Div(text="<br>"),
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