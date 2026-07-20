import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="MARSAD | AI Value Engineering Platform",

    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --lavender-main: #8F73B9;
            --lavender-mid: #7C61A8;
            --lavender-dark: #654A91;
            --lavender-deep: #4D356F;
            --lavender-soft: #BDA7D8;
            --lavender-card: #75599D;
            --lavender-card-light: #8B70B3;
            --saudi-green: #006C35;
            --saudi-green-dark: #00542A;
            --sand: #D8C19F;
            --sand-light: #E9D8BA;
            --sand-dark: #B89A70;
            --white: #FFFFFF;
        }

        /* Lavender stone / carved 3D background */
        .stApp {
            background-color: var(--lavender-main);
            background-image:
                radial-gradient(circle at 18% 22%, rgba(255,255,255,.16) 0 1px, transparent 2px),
                radial-gradient(circle at 72% 34%, rgba(61,39,91,.20) 0 1.5px, transparent 2.6px),
                radial-gradient(circle at 38% 78%, rgba(255,255,255,.10) 0 2px, transparent 3px),
                repeating-linear-gradient(118deg, rgba(255,255,255,.035) 0 2px, rgba(55,34,84,.035) 2px 5px),
                linear-gradient(145deg, #9A80C0 0%, #8569B0 42%, #76599F 72%, #8F73B9 100%);
            background-size: 38px 38px, 57px 57px, 83px 83px, 11px 11px, cover;
            background-attachment: fixed;
            color: var(--white);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            opacity: .38;
            background:
                linear-gradient(135deg, rgba(255,255,255,.08), transparent 38%),
                radial-gradient(ellipse at 50% 0%, rgba(255,255,255,.12), transparent 55%),
                radial-gradient(ellipse at 50% 100%, rgba(57,34,87,.18), transparent 60%);
            mix-blend-mode: soft-light;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
            position: relative;
            z-index: 1;
        }

        .main-title {
            font-size: 2.65rem;
            font-weight: 900;
            margin-bottom: .25rem;
            color: var(--white);
            letter-spacing: .3px;
            text-shadow: 0 3px 16px rgba(45,27,70,.35);
        }

        .sub-title {
            font-size: 1.05rem;
            color: var(--white);
            margin-bottom: 1.2rem;
            max-width: 1050px;
            line-height: 1.8;
        }

        h1, h2, h3, h4, h5, h6, p, label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"] {
            color: var(--white) !important;
        }

        [data-testid="stSidebar"] {
            background-color: #775A9F;
            background-image:
                radial-gradient(circle at 25% 15%, rgba(255,255,255,.11) 0 1px, transparent 2px),
                repeating-linear-gradient(125deg, rgba(255,255,255,.025) 0 2px, rgba(40,24,64,.04) 2px 6px),
                linear-gradient(180deg, #8D72B6 0%, #735798 58%, #654A91 100%);
            background-size: 43px 43px, 12px 12px, cover;
            border-right: 3px solid var(--saudi-green);
        }

        [data-testid="stSidebar"] * { color: var(--white) !important; }

        [data-testid="stMetric"] {
            text-align: center;
            background: linear-gradient(145deg, rgba(148,121,187,.96), rgba(109,79,151,.96));
            border: 1.5px solid var(--saudi-green);
            border-radius: 18px;
            padding: 16px 10px;
            box-shadow: inset 1px 1px 0 rgba(255,255,255,.18), inset -2px -2px 8px rgba(50,29,78,.18), 0 9px 24px rgba(55,34,84,.22);
        }

        [data-testid="stMetricValue"] {
            color: var(--saudi-green) !important;
            font-weight: 900;
        }
        [data-testid="stMetricLabel"] { color: var(--white) !important; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(101,74,145,.62);
            padding: 8px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,.28);
            flex-wrap: wrap;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.13);
        }

        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(145deg, rgba(148,121,187,.90), rgba(112,82,155,.90));
            border: 1px solid rgba(255,255,255,.30);
            border-radius: 12px;
            color: var(--white);
            padding: 8px 14px;
        }

        .stTabs [aria-selected="true"] {
            background: var(--saudi-green) !important;
            color: var(--white) !important;
            font-weight: 850;
            border-color: rgba(255,255,255,.72) !important;
        }

        div[data-testid="stExpander"] {
            background: linear-gradient(145deg, rgba(139,112,179,.96), rgba(101,74,145,.96));
            border: 1.5px solid rgba(255,255,255,.45);
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 10px;
            box-shadow: inset 1px 1px 0 rgba(255,255,255,.14), 0 8px 22px rgba(50,29,78,.18);
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * { color: var(--white) !important; font-weight: 750; }

        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, var(--saudi-green), var(--saudi-green-dark));
            color: var(--white) !important;
            border: 1px solid rgba(255,255,255,.52);
            border-radius: 12px;
            font-weight: 850;
            min-height: 44px;
            box-shadow: 0 7px 20px rgba(0,108,53,.28);
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover { filter: brightness(1.08); border: 1px solid var(--white); }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1.5px solid rgba(255,255,255,.45);
            border-radius: 14px;
            overflow: hidden;
            background: var(--lavender-card);
        }

        [data-testid="stAlert"] {
            background: linear-gradient(145deg, rgba(145,117,184,.94), rgba(105,77,148,.94));
            border: 1.5px solid rgba(255,255,255,.42);
            border-radius: 14px;
        }

        .value-card {
            background: linear-gradient(145deg, rgba(148,121,187,.97), rgba(105,77,148,.97));
            border: 1.5px solid var(--saudi-green);
            border-radius: 18px;
            padding: 18px;
            min-height: 155px;
            box-shadow: inset 1px 1px 0 rgba(255,255,255,.18), inset -2px -2px 8px rgba(50,29,78,.18), 0 10px 28px rgba(50,29,78,.20);
        }
        .value-card h4 { color: var(--saudi-green) !important; margin-top: 0; margin-bottom: 8px; font-weight: 900; }
        .value-card p, .value-card div, .value-card span { color: var(--white) !important; }

        .section-banner {
            background: linear-gradient(145deg, rgba(151,124,190,.95), rgba(111,82,154,.95));
            border-left: 5px solid var(--saudi-green);
            border-radius: 12px;
            padding: 14px 18px;
            margin: 8px 0 18px 0;
            box-shadow: inset 1px 1px 0 rgba(255,255,255,.15), 0 7px 20px rgba(50,29,78,.15);
        }

        .formula-box {
            background: linear-gradient(145deg, rgba(139,112,179,.96), rgba(101,74,145,.96));
            border: 1.5px solid rgba(255,255,255,.42);
            border-radius: 14px;
            padding: 14px 16px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            color: var(--white) !important;
        }

        .green-divider { height: 2px; border: 0; background: linear-gradient(90deg, transparent, var(--saudi-green), transparent); margin: 22px 0; }

        input, textarea, [data-baseweb="select"] > div {
            background-color: rgba(117,89,157,.98) !important;
            color: var(--white) !important;
            border-color: rgba(255,255,255,.52) !important;
        }
        input::placeholder, textarea::placeholder { color: rgba(255,255,255,.72) !important; }

        [data-baseweb="popover"], [role="listbox"] { background: #75599D !important; color: var(--white) !important; }
        [data-baseweb="popover"] *, [role="listbox"] * { color: var(--white) !important; }

        [data-testid="stSlider"] [role="slider"] { background-color: var(--saudi-green) !important; border-color: var(--white) !important; }
        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background-color: var(--sand-dark) !important; }

        /* Only the black chart surfaces are changed to lavender. Text styling stays unchanged. */
        [data-testid="stPlotlyChart"],
        [data-testid="stPlotlyChart"] > div,
        .js-plotly-plot,
        .plot-container,
        .plotly,
        .svg-container {
            border-radius: 16px;
            background: var(--lavender-card) !important;
        }

        [data-testid="stPlotlyChart"] {
            border: 2px solid rgba(255,255,255,.52);
            overflow: hidden;
            box-shadow: inset 1px 1px 0 rgba(255,255,255,.14), 0 9px 24px rgba(50,29,78,.20);
        }

        /* Excel uploader: replace its dark/black box with lavender only. */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] > div {
            background: var(--lavender-card) !important;
            border-color: rgba(255,255,255,.52) !important;
            border-radius: 14px !important;
        }

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, #8F73B9, #75599D) !important;
            border: 1px solid rgba(255,255,255,.60) !important;
        }

        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SAMPLE DATA
# =========================================================
@st.cache_data
def get_sample_data() -> Dict[str, pd.DataFrame]:
    project = pd.DataFrame(
        [{
            "Project Name": "Smart Office Building",
            "City": "Riyadh",
            "Building Area m2": 12000,
            "Analysis Period Years": 30,
            "Project Budget SAR": 45000000,
            "Planned Duration Days": 540,
            "Annual Discount Rate": 0.05,
            "Electricity Cost SAR per kWh": 0.18,
            "Carbon Factor kgCO2e per kWh": 0.45,
            "Risk Reserve Ratio": 0.08
        }]
    )

    environment = pd.DataFrame(
        [{
            "City": "Riyadh",
            "Average Summer Temperature C": 43,
            "Average Humidity Ratio": 0.22,
            "Annual Cooling Hours": 3200,
            "Dust Severity 1-10": 8,
            "Rain Severity 1-10": 2,
            "Climate Factor": 1.15
        }]
    )

    activities = pd.DataFrame(
        [
            ["A01", "Detailed Design", "", 35, 30, 45, 180000, 8, 0.25, "Design"],
            ["A02", "Material Approval", "A01", 20, 18, 28, 90000, 4, 0.30, "Procurement"],
            ["A03", "Earthworks", "A01", 30, 28, 40, 450000, 22, 0.18, "Construction"],
            ["A04", "Structural Works", "A03", 120, 110, 145, 6200000, 85, 0.22, "Construction"],
            ["A05", "Facade Procurement", "A02", 75, 70, 100, 3300000, 12, 0.40, "Supply"],
            ["A06", "Facade Installation", "A04,A05", 70, 65, 90, 2400000, 48, 0.32, "Construction"],
            ["A07", "MEP Works", "A04", 115, 105, 145, 5400000, 76, 0.28, "Construction"],
            ["A08", "Finishing Works", "A06,A07", 95, 90, 120, 4600000, 70, 0.25, "Construction"],
            ["A09", "Testing and Handover", "A08", 25, 22, 35, 650000, 18, 0.20, "Handover"],
        ],
        columns=[
            "Activity ID",
            "Activity Name",
            "Predecessors",
            "Planned Duration Days",
            "Optimistic Duration Days",
            "Pessimistic Duration Days",
            "Planned Cost SAR",
            "Workers",
            "Delay Probability",
            "Activity Type",
        ]
    )

    materials = pd.DataFrame(
        [
            ["M01", "Double Low-E Glass", "Facade", 520, 1.60, 30, 0.020, 12, 320, 3.8, 0.85, "S01"],
            ["M02", "Triple Glazing", "Facade", 690, 1.05, 35, 0.016, 15, 390, 4.6, 0.72, "S02"],
            ["M03", "PIR Insulation Panels", "Insulation", 155, 0.022, 25, 0.012, 10, 110, 3.2, 0.90, "S03"],
            ["M04", "High-Density Rock Wool", "Insulation", 175, 0.035, 35, 0.009, 12, 75, 4.1, 0.78, "S04"],
            ["M05", "Aluminium Composite Panels", "Cladding", 260, 0.38, 20, 0.018, 8, 210, 3.4, 0.82, "S01"],
            ["M06", "Enhanced GRC Panels", "Cladding", 230, 0.45, 30, 0.013, 10, 165, 3.9, 0.76, "S05"],
        ],
        columns=[
            "Material ID",
            "Material Name",
            "Category",
            "Initial Cost SAR per m2",
            "U Value",
            "Service Life Years",
            "Annual Maintenance Ratio",
            "Lead Time Weeks",
            "Embodied Carbon kgCO2e per m2",
            "Quality Rating 1-5",
            "Local Availability Ratio",
            "Supplier ID",
        ]
    )

    suppliers = pd.DataFrame(
        [
            ["S01", "Supplier One", 0.86, 0.12, 0.88, 0.82, 14, 0.70],
            ["S02", "Supplier Two", 0.79, 0.18, 0.93, 0.76, 18, 0.65],
            ["S03", "Supplier Three", 0.91, 0.08, 0.86, 0.90, 9, 0.82],
            ["S04", "Supplier Four", 0.84, 0.11, 0.90, 0.87, 11, 0.78],
            ["S05", "Supplier Five", 0.77, 0.21, 0.82, 0.71, 20, 0.55],
        ],
        columns=[
            "Supplier ID",
            "Supplier Name",
            "On-Time Delivery Ratio",
            "Default Probability",
            "Supply Quality Ratio",
            "Supply Flexibility Ratio",
            "Average Lead Time Days",
            "Local Content Ratio",
        ]
    )

    risks = pd.DataFrame(
        [
            ["R01", "Facade supplier delay", "Supply", 0.40, 4, 18, 650000, "A05", "S01"],
            ["R02", "Labor shortage", "Resources", 0.28, 3, 14, 320000, "A04", ""],
            ["R03", "Severe dust event", "Weather", 0.32, 3, 7, 140000, "A06", ""],
            ["R04", "Late design change", "Scope", 0.20, 5, 25, 950000, "A01", ""],
            ["R05", "Material price increase", "Cost", 0.35, 4, 0, 1200000, "A02", ""],
        ],
        columns=[
            "Risk ID",
            "Risk Description",
            "Category",
            "Probability",
            "Impact 1-5",
            "Expected Delay Days",
            "Cost Impact SAR",
            "Activity ID",
            "Supplier ID",
        ]
    )

    resources = pd.DataFrame(
        [
            ["RES01", "Civil Engineers", "Human", 8, 1800, 0.78, 4.5, 5],
            ["RES02", "Facade Technicians", "Human", 26, 950, 0.84, 4.2, 10],
            ["RES03", "Tower Crane", "Equipment", 2, 5400, 0.88, 4.0, 7],
            ["RES04", "MEP Team", "Human", 35, 1100, 0.82, 4.3, 8],
        ],
        columns=[
            "Resource ID",
            "Resource Name",
            "Resource Type",
            "Available Quantity",
            "Daily Cost SAR",
            "Utilization Ratio",
            "Skill or Reliability Rating",
            "Replacement Time Days",
        ]
    )

    return {
        "Project": project,
        "Environment": environment,
        "Activities": activities,
        "Materials": materials,
        "Suppliers": suppliers,
        "Risks": risks,
        "Resources": resources,
    }


# =========================================================
# EXCEL FUNCTIONS
# =========================================================
def create_excel_template(data: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()

    instructions = pd.DataFrame(
        {
            "Sheet": list(data.keys()),
            "Purpose": [
                "General project, budget, duration, and lifecycle settings.",
                "Climate and environmental inputs.",
                "Schedule, dependencies, duration, cost, and labor data.",
                "Material alternatives and lifecycle characteristics.",
                "Supplier reliability and supply-chain inputs.",
                "Project risk register.",
                "Human and equipment resource data.",
            ],
            "Important Note": [
                "Do not rename columns.",
                "Use ratios between 0 and 1 where requested.",
                "Write predecessors as A01,A02.",
                "Each row represents one material alternative.",
                "Supplier ID must match the Materials sheet.",
                "Probability must be between 0 and 1.",
                "Each row represents one resource.",
            ],
        }
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False)

        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    output.seek(0)
    return output.getvalue()


def load_excel(file) -> Dict[str, pd.DataFrame]:
    required = [
        "Project",
        "Environment",
        "Activities",
        "Materials",
        "Suppliers",
        "Risks",
        "Resources",
    ]

    workbook = pd.ExcelFile(file)
    missing = [sheet for sheet in required if sheet not in workbook.sheet_names]

    if missing:
        raise ValueError("Missing sheets: " + ", ".join(missing))

    return {
        sheet: pd.read_excel(workbook, sheet_name=sheet)
        for sheet in required
    }


def export_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    output.seek(0)
    return output.getvalue()


# =========================================================
# GENERAL CALCULATION HELPERS
# =========================================================
def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def present_value(amount: float, year: int, discount_rate: float) -> float:
    if year <= 0:
        return amount

    return amount / ((1 + discount_rate) ** year)


def normalize(series: pd.Series, inverse: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)

    if numeric.max() == numeric.min():
        normalized = pd.Series(np.full(len(numeric), 0.5), index=numeric.index)
    else:
        normalized = (numeric - numeric.min()) / (numeric.max() - numeric.min())

    return 1 - normalized if inverse else normalized


# =========================================================
# MATERIAL LIFECYCLE INTELLIGENCE
# =========================================================
def analyze_materials(
    materials: pd.DataFrame,
    suppliers: pd.DataFrame,
    project: pd.DataFrame,
    environment: pd.DataFrame,
    target_area: float,
    weights: Dict[str, float],
) -> pd.DataFrame:
    project_row = project.iloc[0]
    environment_row = environment.iloc[0]

    years = int(safe_float(project_row["Analysis Period Years"], 30))
    discount_rate = safe_float(project_row["Annual Discount Rate"], 0.05)
    electricity_price = safe_float(
        project_row["Electricity Cost SAR per kWh"],
        0.18,
    )
    carbon_factor = safe_float(
        project_row["Carbon Factor kgCO2e per kWh"],
        0.45,
    )
    cooling_hours = safe_float(
        environment_row["Annual Cooling Hours"],
        3000,
    )
    climate_factor = safe_float(
        environment_row["Climate Factor"],
        1.0,
    )
    dust_severity = safe_float(
        environment_row["Dust Severity 1-10"],
        5,
    )

    merged = materials.merge(
        suppliers,
        on="Supplier ID",
        how="left",
    )

    rows = []

    for _, row in merged.iterrows():
        unit_cost = safe_float(row["Initial Cost SAR per m2"])
        u_value = max(safe_float(row["U Value"], 1.0), 0.01)
        service_life = max(int(safe_float(row["Service Life Years"], 20)), 1)
        maintenance_ratio = safe_float(
            row["Annual Maintenance Ratio"],
            0.01,
        )
        embodied_carbon = safe_float(
            row["Embodied Carbon kgCO2e per m2"],
            0,
        )
        quality = safe_float(row["Quality Rating 1-5"], 3)
        local_availability = safe_float(
            row["Local Availability Ratio"],
            0.5,
        )
        supplier_default = safe_float(
            row["Default Probability"],
            0.2,
        )
        supplier_commitment = safe_float(
            row["On-Time Delivery Ratio"],
            0.7,
        )
        supplier_flexibility = safe_float(
            row["Supply Flexibility Ratio"],
            0.7,
        )
        lead_time_weeks = safe_float(
            row["Lead Time Weeks"],
            10,
        )

        initial_cost = unit_cost * target_area

        annual_energy_kwh = (
            u_value
            * target_area
            * cooling_hours
            * climate_factor
            * 0.12
        )

        annual_energy_cost = annual_energy_kwh * electricity_price

        dust_multiplier = 1 + (dust_severity / 10) * 0.25
        annual_maintenance_cost = (
            initial_cost
            * maintenance_ratio
            * dust_multiplier
        )

        replacement_years = list(
            range(service_life, years + 1, service_life)
        )

        replacement_present_value = sum(
            present_value(initial_cost * 0.85, year, discount_rate)
            for year in replacement_years
        )

        maintenance_present_value = sum(
            present_value(
                annual_maintenance_cost,
                year,
                discount_rate,
            )
            for year in range(1, years + 1)
        )

        energy_present_value = sum(
            present_value(
                annual_energy_cost,
                year,
                discount_rate,
            )
            for year in range(1, years + 1)
        )

        lifecycle_cost = (
            initial_cost
            + replacement_present_value
            + maintenance_present_value
            + energy_present_value
        )

        operational_carbon = (
            annual_energy_kwh
            * carbon_factor
            * years
        )

        total_carbon = (
            embodied_carbon * target_area
            + operational_carbon
        )

        supply_chain_risk = (
            0.40 * supplier_default
            + 0.25 * (1 - supplier_commitment)
            + 0.20 * (1 - supplier_flexibility)
            + 0.15 * min(lead_time_weeks / 30, 1)
        )

        resilience = (
            0.35 * (quality / 5)
            + 0.25 * local_availability
            + 0.20 * supplier_commitment
            + 0.20 * supplier_flexibility
        )

        rows.append(
            {
                **row.to_dict(),
                "Total Initial Cost SAR": initial_cost,
                "Annual Energy Cost SAR": annual_energy_cost,
                "Annual Maintenance Cost SAR": annual_maintenance_cost,
                "Replacement Present Value SAR": replacement_present_value,
                "Lifecycle Cost SAR": lifecycle_cost,
                "Lifecycle Carbon kgCO2e": total_carbon,
                "Supply Chain Risk": supply_chain_risk,
                "Material Resilience": resilience,
                "Replacement Years": (
                    ", ".join(map(str, replacement_years))
                    if replacement_years
                    else "No replacement"
                ),
            }
        )

    result = pd.DataFrame(rows)

    result["Cost Score"] = normalize(
        result["Lifecycle Cost SAR"],
        inverse=True,
    )
    result["Energy Score"] = normalize(
        result["Annual Energy Cost SAR"],
        inverse=True,
    )
    result["Carbon Score"] = normalize(
        result["Lifecycle Carbon kgCO2e"],
        inverse=True,
    )
    result["Resilience Score"] = normalize(
        result["Material Resilience"],
    )
    result["Supply Score"] = normalize(
        result["Supply Chain Risk"],
        inverse=True,
    )

    total_weight = max(sum(weights.values()), 1e-9)

    result["AI Value Index"] = 100 * (
        weights["Cost"] * result["Cost Score"]
        + weights["Energy"] * result["Energy Score"]
        + weights["Carbon"] * result["Carbon Score"]
        + weights["Resilience"] * result["Resilience Score"]
        + weights["Supply"] * result["Supply Score"]
    ) / total_weight

    regret_columns = {
        "Lifecycle Cost SAR": "Cost Regret",
        "Annual Energy Cost SAR": "Energy Regret",
        "Lifecycle Carbon kgCO2e": "Carbon Regret",
        "Supply Chain Risk": "Supply Regret",
    }

    for source_column, regret_column in regret_columns.items():
        best_value = result[source_column].min()
        denominator = max(
            result[source_column].max() - best_value,
            1e-9,
        )

        result[regret_column] = (
            result[source_column] - best_value
        ) / denominator

    result["Decision Regret Index"] = 100 * result[
        list(regret_columns.values())
    ].mean(axis=1)

    return result.sort_values(
        ["AI Value Index", "Decision Regret Index"],
        ascending=[False, True],
    ).reset_index(drop=True)


# =========================================================
# PROACTIVE RISK ENGINE
# =========================================================
def analyze_risks(
    risks: pd.DataFrame,
    activities: pd.DataFrame,
    suppliers: pd.DataFrame,
    environment: pd.DataFrame,
) -> pd.DataFrame:
    risk_data = risks.merge(
        activities[
            [
                "Activity ID",
                "Activity Name",
                "Delay Probability",
                "Workers",
            ]
        ],
        on="Activity ID",
        how="left",
    )

    risk_data = risk_data.merge(
        suppliers[
            [
                "Supplier ID",
                "Default Probability",
                "On-Time Delivery Ratio",
            ]
        ],
        on="Supplier ID",
        how="left",
    )

    climate_factor = safe_float(
        environment.iloc[0]["Climate Factor"],
        1.0,
    )
    dust_severity = safe_float(
        environment.iloc[0]["Dust Severity 1-10"],
        5,
    )

    base_probability = pd.to_numeric(
        risk_data["Probability"],
        errors="coerce",
    ).fillna(0.2)

    activity_probability = pd.to_numeric(
        risk_data["Delay Probability"],
        errors="coerce",
    ).fillna(0.15)

    supplier_default = pd.to_numeric(
        risk_data["Default Probability"],
        errors="coerce",
    ).fillna(0.0)

    climate_adjustment = np.where(
        risk_data["Category"]
        .astype(str)
        .str.contains("Weather|Climate|Dust", regex=True),
        min(
            0.20,
            (climate_factor - 1) * 0.4
            + dust_severity / 100,
        ),
        0.0,
    )

    risk_data["Adjusted Probability"] = np.clip(
        0.55 * base_probability
        + 0.25 * activity_probability
        + 0.20 * supplier_default
        + climate_adjustment,
        0,
        0.98,
    )

    impact = pd.to_numeric(
        risk_data["Impact 1-5"],
        errors="coerce",
    ).fillna(3)

    delay = pd.to_numeric(
        risk_data["Expected Delay Days"],
        errors="coerce",
    ).fillna(0)

    cost = pd.to_numeric(
        risk_data["Cost Impact SAR"],
        errors="coerce",
    ).fillna(0)

    delay_normalized = delay / max(delay.max(), 1)
    cost_normalized = cost / max(cost.max(), 1)

    risk_data["Composite Impact"] = (
        0.45 * (impact / 5)
        + 0.30 * delay_normalized
        + 0.25 * cost_normalized
    )

    risk_data["Proactive Risk Index"] = (
        100
        * risk_data["Adjusted Probability"]
        * risk_data["Composite Impact"]
    )

    def risk_level(value: float) -> str:
        if value >= 45:
            return "Critical"
        if value >= 25:
            return "High"
        if value >= 12:
            return "Medium"
        return "Low"

    risk_data["Risk Level"] = risk_data[
        "Proactive Risk Index"
    ].apply(risk_level)

    risk_data["Expected Monetary Loss SAR"] = (
        risk_data["Adjusted Probability"]
        * cost
    )

    risk_data["Weighted Expected Delay Days"] = (
        risk_data["Adjusted Probability"]
        * delay
    )

    def recommended_response(row: pd.Series) -> str:
        category = str(row["Category"])

        if "Supply" in category:
            return (
                "Activate backup supplier, split purchase order, "
                "and reserve early inventory."
            )

        if "Resource" in category:
            return (
                "Reallocate labor, use a targeted additional shift, "
                "and protect the critical activity."
            )

        if "Weather" in category:
            return (
                "Move the execution window, prepare indoor substitute work, "
                "and add a dynamic weather buffer."
            )

        if "Scope" in category:
            return (
                "Freeze the decision point, use staged approval, "
                "and activate a fast change-control board."
            )

        if "Cost" in category:
            return (
                "Partially lock prices, purchase early, "
                "and use lifecycle-value alternatives."
            )

        return "Create a tailored response plan and early-warning trigger."

    risk_data["Recommended Response"] = risk_data.apply(
        recommended_response,
        axis=1,
    )

    return risk_data.sort_values(
        "Proactive Risk Index",
        ascending=False,
    ).reset_index(drop=True)


# =========================================================
# SCHEDULE DIGITAL TWIN
# =========================================================
def parse_predecessors(value) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def analyze_schedule(activities: pd.DataFrame) -> pd.DataFrame:
    schedule = activities.copy()

    activity_ids = schedule["Activity ID"].astype(str).tolist()

    duration = dict(
        zip(
            activity_ids,
            pd.to_numeric(
                schedule["Planned Duration Days"],
                errors="coerce",
            ).fillna(0),
        )
    )

    predecessors = dict(
        zip(
            activity_ids,
            schedule["Predecessors"].apply(parse_predecessors),
        )
    )

    early_start = {}
    early_finish = {}
    unresolved = set(activity_ids)

    for _ in range(len(activity_ids) + 5):
        progressed = False

        for activity in list(unresolved):
            activity_predecessors = predecessors.get(activity, [])

            if all(
                predecessor in early_finish
                for predecessor in activity_predecessors
            ):
                early_start[activity] = max(
                    [
                        early_finish[predecessor]
                        for predecessor in activity_predecessors
                    ],
                    default=0,
                )

                early_finish[activity] = (
                    early_start[activity]
                    + duration[activity]
                )

                unresolved.remove(activity)
                progressed = True

        if not unresolved or not progressed:
            break

    for activity in unresolved:
        early_start[activity] = 0
        early_finish[activity] = duration[activity]

    project_finish = (
        max(early_finish.values())
        if early_finish
        else 0
    )

    successors = {
        activity: []
        for activity in activity_ids
    }

    for activity, activity_predecessors in predecessors.items():
        for predecessor in activity_predecessors:
            if predecessor in successors:
                successors[predecessor].append(activity)

    late_start = {}
    late_finish = {}

    for activity in sorted(
        activity_ids,
        key=lambda item: early_finish.get(item, 0),
        reverse=True,
    ):
        activity_successors = successors.get(activity, [])

        late_finish[activity] = min(
            [
                late_start[successor]
                for successor in activity_successors
            ],
            default=project_finish,
        )

        late_start[activity] = (
            late_finish[activity]
            - duration[activity]
        )

    schedule["Early Start"] = (
        schedule["Activity ID"]
        .map(early_start)
        .fillna(0)
    )

    schedule["Early Finish"] = (
        schedule["Activity ID"]
        .map(early_finish)
        .fillna(0)
    )

    schedule["Late Start"] = (
        schedule["Activity ID"]
        .map(late_start)
        .fillna(0)
    )

    schedule["Late Finish"] = (
        schedule["Activity ID"]
        .map(late_finish)
        .fillna(0)
    )

    schedule["Total Float Days"] = (
        schedule["Late Start"]
        - schedule["Early Start"]
    )

    schedule["Critical Activity"] = np.where(
        schedule["Total Float Days"].abs() < 1e-6,
        "Yes",
        "No",
    )

    delay_probability = pd.to_numeric(
        schedule["Delay Probability"],
        errors="coerce",
    ).fillna(0.2)

    planned = pd.to_numeric(
        schedule["Planned Duration Days"],
        errors="coerce",
    ).fillna(0)

    pessimistic = pd.to_numeric(
        schedule["Pessimistic Duration Days"],
        errors="coerce",
    ).fillna(planned)

    schedule["Risk-Adjusted Duration Days"] = (
        planned
        + delay_probability
        * np.maximum(
            pessimistic - planned,
            planned * 0.15,
        )
    )

    return schedule.sort_values(
        ["Critical Activity", "Early Start"],
        ascending=[False, True],
    ).reset_index(drop=True)


# =========================================================
# PROJECT HEALTH MODEL
# =========================================================
def calculate_project_health(
    project: pd.DataFrame,
    schedule: pd.DataFrame,
    risks: pd.DataFrame,
    material_results: pd.DataFrame,
) -> Dict[str, float]:
    budget = safe_float(
        project.iloc[0]["Project Budget SAR"],
        1,
    )

    planned_duration = safe_float(
        project.iloc[0]["Planned Duration Days"],
        1,
    )

    network_duration = safe_float(
        schedule["Early Finish"].max(),
        planned_duration,
    )

    expected_risk_delay = safe_float(
        risks["Weighted Expected Delay Days"].sum(),
        0,
    )

    expected_risk_cost = safe_float(
        risks["Expected Monetary Loss SAR"].sum(),
        0,
    )

    schedule_overrun_probability = np.clip(
        (
            network_duration
            + expected_risk_delay
            - planned_duration
        ) / planned_duration
        + 0.25,
        0,
        0.95,
    )

    budget_overrun_probability = np.clip(
        expected_risk_cost / budget + 0.12,
        0,
        0.95,
    )

    critical_ratio = (
        schedule["Critical Activity"]
        .eq("Yes")
        .mean()
        if len(schedule)
        else 0
    )

    average_supply_risk = safe_float(
        material_results["Supply Chain Risk"].mean(),
        0.2,
    )

    overall_risk = np.clip(
        0.30 * schedule_overrun_probability
        + 0.30 * budget_overrun_probability
        + 0.20 * critical_ratio
        + 0.20 * average_supply_risk,
        0,
        1,
    )

    success_probability = np.clip(
        1 - overall_risk,
        0.05,
        0.98,
    )

    return {
        "Project Success Probability": success_probability * 100,
        "Schedule Overrun Probability": schedule_overrun_probability * 100,
        "Budget Overrun Probability": budget_overrun_probability * 100,
        "Overall Risk Index": overall_risk * 100,
        "Expected Delay Days": expected_risk_delay,
        "Expected Loss SAR": expected_risk_cost,
        "Current Network Duration Days": network_duration,
    }


# =========================================================
# WHAT-IF SIMULATION
# =========================================================
def run_scenario(
    base_health: Dict[str, float],
    project: pd.DataFrame,
    supplier_delay_days: int,
    labor_change_percent: float,
    material_cost_change_percent: float,
    weather_delay_days: int,
    mitigation_strength: float,
) -> Dict[str, float]:
    budget = safe_float(
        project.iloc[0]["Project Budget SAR"],
        1,
    )

    planned_duration = safe_float(
        project.iloc[0]["Planned Duration Days"],
        1,
    )

    if labor_change_percent >= 0:
        labor_time_effect = (
            -0.20
            * labor_change_percent
            / 100
            * planned_duration
        )
    else:
        labor_time_effect = (
            -0.35
            * labor_change_percent
            / 100
            * planned_duration
        )

    gross_delay = (
        base_health["Expected Delay Days"]
        + supplier_delay_days * 0.85
        + weather_delay_days * 0.70
        + labor_time_effect
    )

    net_delay = max(
        0,
        gross_delay
        * (1 - 0.65 * mitigation_strength),
    )

    material_cost_change = (
        budget
        * material_cost_change_percent
        / 100
    )

    labor_cost_change = (
        budget
        * max(labor_change_percent, 0)
        / 100
        * 0.08
    )

    delay_cost = (
        net_delay
        * (budget / planned_duration)
        * 0.20
    )

    gross_additional_cost = (
        base_health["Expected Loss SAR"]
        + material_cost_change
        + labor_cost_change
        + delay_cost
    )

    net_additional_cost = max(
        0,
        gross_additional_cost
        * (1 - 0.45 * mitigation_strength),
    )

    schedule_probability = np.clip(
        20
        + 100 * net_delay / planned_duration,
        0,
        98,
    )

    budget_probability = np.clip(
        12
        + 100 * net_additional_cost / budget,
        0,
        98,
    )

    overall_risk = np.clip(
        0.52 * schedule_probability
        + 0.48 * budget_probability,
        0,
        100,
    )

    success_probability = np.clip(
        100 - overall_risk * 0.82,
        3,
        98,
    )

    return {
        "Expected Delay Days": net_delay,
        "Additional Cost SAR": net_additional_cost,
        "Schedule Overrun Probability": schedule_probability,
        "Budget Overrun Probability": budget_probability,
        "Overall Risk Index": overall_risk,
        "Project Success Probability": success_probability,
    }


# =========================================================
# DECISION DNA
# =========================================================
def calculate_decision_dna(
    health: Dict[str, float],
    best_material: pd.Series,
    risk_results: pd.DataFrame,
) -> Dict[str, float]:
    major_risk_ratio = (
        risk_results["Risk Level"]
        .isin(["Critical", "High"])
        .mean()
        * 100
    )

    return {
        "Economic Value": safe_float(
            best_material["Cost Score"]
        ) * 100,
        "Sustainability": safe_float(
            best_material["Carbon Score"]
        ) * 100,
        "Supply Resilience": safe_float(
            best_material["Supply Score"]
        ) * 100,
        "Project Resilience": (
            100 - health["Overall Risk Index"]
        ),
        "Schedule Safety": (
            100
            - health["Schedule Overrun Probability"]
        ),
        "Recovery Capacity": max(
            0,
            100 - major_risk_ratio * 0.65,
        ),
    }


def generate_recommendations(
    health: Dict[str, float],
    best_material: pd.Series,
    risk_results: pd.DataFrame,
    schedule_results: pd.DataFrame,
) -> List[str]:
    recommendations = []

    recommendations.append(
        f"Provisionally select {best_material['Material Name']} "
        f"with an AI Value Index of "
        f"{best_material['AI Value Index']:.1f}/100 "
        f"and a Decision Regret Index of "
        f"{best_material['Decision Regret Index']:.1f}/100."
    )

    highest_risk = risk_results.iloc[0]

    recommendations.append(
        f"Highest early-warning signal: "
        f"{highest_risk['Risk Description']}. "
        f"Recommended response: "
        f"{highest_risk['Recommended Response']}"
    )

    critical_activities = schedule_results[
        schedule_results["Critical Activity"] == "Yes"
    ]

    if not critical_activities.empty:
        names = ", ".join(
            critical_activities["Activity Name"]
            .head(4)
            .astype(str)
        )

        recommendations.append(
            "Protect the following critical activities "
            f"from uncontrolled changes: {names}."
        )

    if health["Schedule Overrun Probability"] >= 40:
        recommendations.append(
            "Create a time decision gate before procurement activities. "
            "No alternative should be approved before testing its effect "
            "on supplier lead time and the critical path."
        )

    if health["Budget Overrun Probability"] >= 35:
        recommendations.append(
            "Replace the fixed risk reserve with a dynamic contingency "
            "linked to active risks and their probability of occurrence."
        )

    recommendations.append(
        "Apply the reversible-decision rule: high-impact decisions "
        "with low reversibility require higher approval and at least "
        "two simulated alternatives."
    )

    return recommendations






# =========================================================
# SUPERVISED MACHINE-LEARNING DELAY PREDICTION
# =========================================================
ML_FEATURES = [
    "Progress %",
    "Time Elapsed %",
    "Cost Variance %",
    "Delayed Critical Activities",
    "Workforce Availability %",
    "Supplier Reliability %",
    "Open Risks",
    "Average Risk Probability %",
    "Weather Disruption Days",
]


@st.cache_data
def create_synthetic_schedule_training_data(
    rows: int = 700,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create reproducible synthetic project records for the hackathon MVP."""
    rng = np.random.default_rng(random_state)

    progress = rng.uniform(5, 100, rows)
    time_elapsed = np.clip(
        progress + rng.normal(6, 20, rows),
        1,
        115,
    )
    cost_variance = np.clip(rng.normal(3, 10, rows), -18, 38)
    delayed_critical = rng.integers(0, 9, rows)
    workforce = rng.uniform(55, 110, rows)
    supplier_reliability = rng.uniform(50, 100, rows)
    open_risks = rng.integers(0, 18, rows)
    average_risk_probability = rng.uniform(5, 75, rows)
    weather_days = rng.integers(0, 24, rows)

    schedule_gap = np.maximum(time_elapsed - progress, 0)
    risk_signal = (
        0.030 * schedule_gap
        + 0.055 * np.maximum(cost_variance, 0)
        + 0.38 * delayed_critical
        + 0.032 * np.maximum(85 - workforce, 0)
        + 0.030 * np.maximum(82 - supplier_reliability, 0)
        + 0.12 * open_risks
        + 0.022 * average_risk_probability
        + 0.075 * weather_days
        + rng.normal(0, 0.75, rows)
    )

    delay_probability = 1 / (1 + np.exp(-(risk_signal - 4.2)))
    delayed = (rng.random(rows) < delay_probability).astype(int)

    predicted_delay_days = np.maximum(
        0,
        0.45 * schedule_gap
        + 1.9 * delayed_critical
        + 0.28 * np.maximum(cost_variance, 0)
        + 0.20 * np.maximum(85 - workforce, 0)
        + 0.18 * np.maximum(82 - supplier_reliability, 0)
        + 0.60 * open_risks
        + 0.11 * average_risk_probability
        + 0.52 * weather_days
        + rng.normal(0, 3.0, rows),
    )
    predicted_delay_days = np.where(delayed == 1, predicted_delay_days, predicted_delay_days * 0.20)

    return pd.DataFrame(
        {
            "Progress %": progress,
            "Time Elapsed %": time_elapsed,
            "Cost Variance %": cost_variance,
            "Delayed Critical Activities": delayed_critical,
            "Workforce Availability %": workforce,
            "Supplier Reliability %": supplier_reliability,
            "Open Risks": open_risks,
            "Average Risk Probability %": average_risk_probability,
            "Weather Disruption Days": weather_days,
            "Project Delayed": delayed,
            "Delay Days": predicted_delay_days,
        }
    )


@st.cache_resource
def train_delay_models() -> Dict[str, object]:
    """Train lightweight supervised models and return validation evidence."""
    training_data = create_synthetic_schedule_training_data()
    X = training_data[ML_FEATURES]
    y_class = training_data["Project Delayed"]
    y_days = training_data["Delay Days"]

    (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_days_train,
        y_days_test,
    ) = train_test_split(
        X,
        y_class,
        y_days,
        test_size=0.22,
        random_state=42,
        stratify=y_class,
    )

    classifier = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )
    regressor = RandomForestRegressor(
        n_estimators=220,
        max_depth=11,
        min_samples_leaf=3,
        random_state=42,
    )

    classifier.fit(X_train, y_class_train)
    regressor.fit(X_train, y_days_train)

    class_prediction = classifier.predict(X_test)
    day_prediction = regressor.predict(X_test)

    return {
        "classifier": classifier,
        "regressor": regressor,
        "accuracy": accuracy_score(y_class_test, class_prediction),
        "mae": mean_absolute_error(y_days_test, day_prediction),
        "training_rows": len(training_data),
        "data": training_data,
    }


def predict_project_delay(input_values: Dict[str, float]) -> Dict[str, object]:
    """Generate ML predictions, feature importance, alerts, and recommendations."""
    bundle = train_delay_models()
    input_frame = pd.DataFrame(
        [[input_values[column] for column in ML_FEATURES]],
        columns=ML_FEATURES,
    )

    delay_probability = float(
        bundle["classifier"].predict_proba(input_frame)[0, 1]
    )
    delay_days = max(
        0.0,
        float(bundle["regressor"].predict(input_frame)[0]),
    )

    if delay_probability >= 0.75:
        risk_level = "Critical"
    elif delay_probability >= 0.55:
        risk_level = "High"
    elif delay_probability >= 0.30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    importance = pd.DataFrame(
        {
            "Factor": ML_FEATURES,
            "Importance": bundle["classifier"].feature_importances_,
            "Current Value": [input_values[column] for column in ML_FEATURES],
        }
    ).sort_values("Importance", ascending=False)

    recommendations = []
    if input_values["Delayed Critical Activities"] >= 2:
        recommendations.append(
            "Protect and recover critical-path activities through targeted crashing or re-sequencing."
        )
    if input_values["Supplier Reliability %"] < 80:
        recommendations.append(
            "Activate a qualified backup supplier and split critical procurement packages."
        )
    if input_values["Workforce Availability %"] < 85:
        recommendations.append(
            "Reallocate skilled labor to critical work and assess a controlled additional shift."
        )
    if input_values["Cost Variance %"] > 8:
        recommendations.append(
            "Freeze non-essential scope changes and connect contingency release to active risk exposure."
        )
    if input_values["Average Risk Probability %"] > 40 or input_values["Open Risks"] > 8:
        recommendations.append(
            "Escalate the highest-probability risks and assign owners, triggers, and response deadlines."
        )
    if input_values["Weather Disruption Days"] > 5:
        recommendations.append(
            "Move weather-sensitive work and prepare indoor substitute activities."
        )
    if input_values["Time Elapsed %"] - input_values["Progress %"] > 10:
        recommendations.append(
            "Launch a schedule recovery gate because time consumption is materially ahead of progress."
        )
    if not recommendations:
        recommendations.append(
            "Maintain weekly monitoring; the current inputs do not indicate an urgent intervention."
        )

    return {
        "Delay Probability": delay_probability * 100,
        "Predicted Delay Days": delay_days,
        "Risk Level": risk_level,
        "Feature Importance": importance,
        "Recommendations": recommendations,
        "Model Accuracy": bundle["accuracy"] * 100,
        "Model MAE": bundle["mae"],
        "Training Rows": bundle["training_rows"],
    }


# =========================================================
# MARSAD CHART THEME — LAVENDER STONE, SAND, AND SAUDI GREEN
# =========================================================
MARSAD_COLOR_SEQUENCE = [
    "#D8C19F",  # sand
    "#E9D8BA",  # light sand
    "#B89A70",  # deep sand
    "#C7AE87",  # warm sand
    "#006C35",  # Saudi flag green
    "#8F73B9",  # lavender
]

MARSAD_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#75599D",
        plot_bgcolor="#75599D",
        font={"color": "#FFFFFF", "family": "Arial, sans-serif"},
        title={"font": {"color": "#FFFFFF", "size": 18}},
        colorway=MARSAD_COLOR_SEQUENCE,
        legend={
            "bgcolor": "rgba(101,74,145,0.78)",
            "bordercolor": "rgba(255,255,255,0.42)",
            "borderwidth": 1,
            "font": {"color": "#FFFFFF"},
        },
        xaxis={
            "gridcolor": "rgba(255,255,255,0.16)",
            "zerolinecolor": "rgba(255,255,255,0.25)",
            "linecolor": "rgba(255,255,255,0.52)",
            "tickfont": {"color": "#FFFFFF"},
            "title": {"font": {"color": "#FFFFFF"}},
        },
        yaxis={
            "gridcolor": "rgba(255,255,255,0.16)",
            "zerolinecolor": "rgba(255,255,255,0.25)",
            "linecolor": "rgba(255,255,255,0.52)",
            "tickfont": {"color": "#FFFFFF"},
            "title": {"font": {"color": "#FFFFFF"}},
        },
        margin={"l": 55, "r": 28, "t": 65, "b": 48},
        hoverlabel={
            "bgcolor": "#654A91",
            "bordercolor": "#E58E0B",
            "font": {"color": "#FFFFFF"},
        },
    )
)

pio.templates["marsad_lavender"] = MARSAD_TEMPLATE
pio.templates.default = "marsad_lavender"
px.defaults.template = "marsad_lavender"
px.defaults.color_discrete_sequence = MARSAD_COLOR_SEQUENCE
px.defaults.color_continuous_scale = ["#DB8B22", "#EA9D2A", "#E08D18"]


def apply_marsad_chart_theme(fig):
    """Apply the lavender-stone visual identity to any Plotly figure."""
    fig.update_layout(template="marsad_lavender")
    return fig


# =========================================================
# AI VALUE ENGINEER — RULE-BASED RECOMMENDATION ENGINE
# =========================================================
def build_value_engineering_action_plan(
    issue_type: str,
    delay_days: int,
    cost_overrun_pct: float,
    supplier_reliability: float,
    function_score: float,
    lifecycle_impact: str,
    weather_days: int,
    labor_change_pct: float,
    critical_activity: bool,
    reversible_decision: bool,
) -> Dict[str, object]:
    """
    Create a transparent Value Engineering action plan.

    This hackathon MVP uses explainable rules instead of an external AI API.
    The recommendations preserve the required function first, then optimize
    lifecycle value, schedule, cost, risk, and implementation feasibility.
    """

    actions: List[Dict[str, object]] = []
    root_causes: List[str] = []
    warnings: List[str] = []

    issue_lower = issue_type.lower()

    # Root-cause indicators
    if delay_days > 0:
        root_causes.append(
            f"Current schedule delay is approximately {delay_days} days."
        )
    if supplier_reliability < 80:
        root_causes.append(
            f"Supplier reliability is low at {supplier_reliability:.0f}%."
        )
    if cost_overrun_pct > 5:
        root_causes.append(
            f"Forecast cost overrun is {cost_overrun_pct:.1f}%."
        )
    if weather_days > 3:
        root_causes.append(
            f"Weather disruption may remove {weather_days} productive days."
        )
    if labor_change_pct < 0:
        root_causes.append(
            f"Available labor is reduced by {abs(labor_change_pct):.0f}%."
        )
    elif labor_change_pct > 0:
        root_causes.append(
            f"Additional labor capacity of {labor_change_pct:.0f}% is available."
        )
    if critical_activity:
        root_causes.append(
            "The affected work is on or near the critical path."
        )

    # Function protection comes first.
    if function_score < 80:
        warnings.append(
            "The current option does not adequately achieve the required function."
        )
        actions.append(
            {
                "Priority": 1,
                "Action": "Reject or redesign the current alternative before approval.",
                "Value Engineering Reason": (
                    "Cost reduction is not value improvement when the required "
                    "function is not achieved."
                ),
                "Expected Delay Recovery Days": 0,
                "Expected Cost Impact %": 1.5,
                "Value Improvement %": 18,
                "Confidence %": 96,
                "Function Status": "Function protection required",
            }
        )
        actions.append(
            {
                "Priority": 2,
                "Action": (
                    "Add the missing performance layer or select a technically "
                    "equivalent alternative."
                ),
                "Value Engineering Reason": (
                    "Restore the required function while keeping lifecycle cost "
                    "and implementation impact controlled."
                ),
                "Expected Delay Recovery Days": 1,
                "Expected Cost Impact %": 2.0,
                "Value Improvement %": 15,
                "Confidence %": 89,
                "Function Status": "Function restored",
            }
        )

    # Delay-related actions
    if delay_days > 0 or "delay" in issue_lower or "schedule" in issue_lower:
        recovery_base = max(2, int(round(delay_days * 0.35)))
        actions.extend(
            [
                {
                    "Priority": 1,
                    "Action": (
                        "Re-sequence non-critical and indoor activities while "
                        "the delayed work is being resolved."
                    ),
                    "Value Engineering Reason": (
                        "Uses existing resources and protects schedule value "
                        "without changing the required function."
                    ),
                    "Expected Delay Recovery Days": min(delay_days, recovery_base),
                    "Expected Cost Impact %": 0.5,
                    "Value Improvement %": 12,
                    "Confidence %": 91,
                    "Function Status": "Preserved",
                },
                {
                    "Priority": 2,
                    "Action": (
                        "Apply controlled schedule crashing only to critical "
                        "activities with the best cost-per-day recovery."
                    ),
                    "Value Engineering Reason": (
                        "Avoids spending on activities that do not reduce the "
                        "project completion date."
                    ),
                    "Expected Delay Recovery Days": min(
                        delay_days, max(1, int(round(delay_days * 0.25)))
                    ),
                    "Expected Cost Impact %": 2.5,
                    "Value Improvement %": 9,
                    "Confidence %": 84,
                    "Function Status": "Preserved",
                },
            ]
        )

    # Supplier / procurement actions
    if supplier_reliability < 85 or "supplier" in issue_lower or "material" in issue_lower:
        supplier_recovery = min(delay_days, max(3, int(round(delay_days * 0.45))))
        actions.extend(
            [
                {
                    "Priority": 1,
                    "Action": (
                        "Activate a prequalified backup supplier that meets the "
                        "same technical specification."
                    ),
                    "Value Engineering Reason": (
                        "Protects the function and reduces procurement exposure "
                        "instead of accepting the lowest purchase price."
                    ),
                    "Expected Delay Recovery Days": supplier_recovery,
                    "Expected Cost Impact %": 1.8,
                    "Value Improvement %": 17,
                    "Confidence %": 93,
                    "Function Status": "Equivalent function required",
                },
                {
                    "Priority": 2,
                    "Action": (
                        "Split the remaining quantity between two suppliers and "
                        "prioritize the critical installation zones."
                    ),
                    "Value Engineering Reason": (
                        "Reduces single-source dependency and protects milestone value."
                    ),
                    "Expected Delay Recovery Days": min(
                        delay_days, max(2, int(round(delay_days * 0.30)))
                    ),
                    "Expected Cost Impact %": 1.2,
                    "Value Improvement %": 14,
                    "Confidence %": 88,
                    "Function Status": "Preserved",
                },
                {
                    "Priority": 3,
                    "Action": (
                        "Evaluate an equivalent local material with shorter lead "
                        "time, matching performance, warranty, and service life."
                    ),
                    "Value Engineering Reason": (
                        "Optimizes total value through availability, resilience, "
                        "local content, and lifecycle performance."
                    ),
                    "Expected Delay Recovery Days": min(
                        delay_days, max(2, int(round(delay_days * 0.40)))
                    ),
                    "Expected Cost Impact %": -1.5,
                    "Value Improvement %": 20,
                    "Confidence %": 82,
                    "Function Status": "Technical approval required",
                },
            ]
        )

    # Cost overrun actions
    if cost_overrun_pct > 0 or "cost" in issue_lower or "budget" in issue_lower:
        savings = min(12.0, max(2.0, cost_overrun_pct * 0.45))
        actions.extend(
            [
                {
                    "Priority": 1,
                    "Action": (
                        "Run function-based alternative analysis on the five "
                        "highest-cost packages."
                    ),
                    "Value Engineering Reason": (
                        "Targets cost concentration while protecting essential functions."
                    ),
                    "Expected Delay Recovery Days": 0,
                    "Expected Cost Impact %": -savings,
                    "Value Improvement %": 22,
                    "Confidence %": 90,
                    "Function Status": "Function must be verified",
                },
                {
                    "Priority": 2,
                    "Action": (
                        "Replace high-maintenance options with alternatives that "
                        "have lower lifecycle ownership cost."
                    ),
                    "Value Engineering Reason": (
                        "Prevents a short-term saving from becoming a long-term cost penalty."
                    ),
                    "Expected Delay Recovery Days": 0,
                    "Expected Cost Impact %": -max(1.5, savings * 0.55),
                    "Value Improvement %": 19,
                    "Confidence %": 86,
                    "Function Status": "Equivalent performance required",
                },
                {
                    "Priority": 3,
                    "Action": (
                        "Standardize repeated components and reduce unnecessary "
                        "specification variation."
                    ),
                    "Value Engineering Reason": (
                        "Improves procurement efficiency, constructability, "
                        "maintenance, and replacement value."
                    ),
                    "Expected Delay Recovery Days": 1,
                    "Expected Cost Impact %": -max(1.0, savings * 0.35),
                    "Value Improvement %": 13,
                    "Confidence %": 84,
                    "Function Status": "Preserved",
                },
            ]
        )

    # Weather actions
    if weather_days > 0 or "weather" in issue_lower:
        actions.extend(
            [
                {
                    "Priority": 1,
                    "Action": (
                        "Move weather-sensitive activities and advance protected "
                        "indoor work packages."
                    ),
                    "Value Engineering Reason": (
                        "Maintains productive flow without weakening quality or function."
                    ),
                    "Expected Delay Recovery Days": min(weather_days, max(1, weather_days - 1)),
                    "Expected Cost Impact %": 0.4,
                    "Value Improvement %": 10,
                    "Confidence %": 90,
                    "Function Status": "Preserved",
                },
                {
                    "Priority": 2,
                    "Action": (
                        "Use temporary protection only where its cost is lower "
                        "than the delay and rework exposure."
                    ),
                    "Value Engineering Reason": (
                        "Selects protection based on value, not on blanket spending."
                    ),
                    "Expected Delay Recovery Days": min(weather_days, 3),
                    "Expected Cost Impact %": 0.8,
                    "Value Improvement %": 8,
                    "Confidence %": 83,
                    "Function Status": "Quality protected",
                },
            ]
        )

    # Resource actions
    if labor_change_pct < 0 or "labor" in issue_lower or "resource" in issue_lower:
        actions.extend(
            [
                {
                    "Priority": 1,
                    "Action": (
                        "Move skilled resources from activities with available "
                        "float to critical activities."
                    ),
                    "Value Engineering Reason": (
                        "Improves schedule value without increasing total headcount."
                    ),
                    "Expected Delay Recovery Days": min(delay_days, max(2, int(delay_days * 0.25))),
                    "Expected Cost Impact %": 0.3,
                    "Value Improvement %": 11,
                    "Confidence %": 88,
                    "Function Status": "Preserved",
                },
                {
                    "Priority": 2,
                    "Action": (
                        "Add temporary labor only after confirming workspace, "
                        "supervision, and productivity capacity."
                    ),
                    "Value Engineering Reason": (
                        "Prevents congestion and cost increase with no real schedule recovery."
                    ),
                    "Expected Delay Recovery Days": min(delay_days, max(1, int(delay_days * 0.20))),
                    "Expected Cost Impact %": 1.7,
                    "Value Improvement %": 7,
                    "Confidence %": 78,
                    "Function Status": "Preserved",
                },
            ]
        )

    # Lifecycle warning
    if lifecycle_impact == "Negative":
        warnings.append(
            "The proposed decision may reduce initial cost but create a negative lifecycle impact."
        )
        actions.append(
            {
                "Priority": 1,
                "Action": (
                    "Recalculate energy, maintenance, replacement, and end-of-life "
                    "cost before approval."
                ),
                "Value Engineering Reason": (
                    "The preferred solution must create value over the full analysis period."
                ),
                "Expected Delay Recovery Days": 0,
                "Expected Cost Impact %": -3.0,
                "Value Improvement %": 16,
                "Confidence %": 94,
                "Function Status": "Lifecycle validation required",
            }
        )

    # Irreversibility increases governance requirements.
    if not reversible_decision:
        warnings.append(
            "This is a low-reversibility decision; approval should require stronger evidence."
        )
        actions.append(
            {
                "Priority": 1,
                "Action": (
                    "Require technical, commercial, lifecycle, and project-control "
                    "approval before implementation."
                ),
                "Value Engineering Reason": (
                    "Reduces future decision regret for expensive or difficult-to-reverse changes."
                ),
                "Expected Delay Recovery Days": 0,
                "Expected Cost Impact %": 0.2,
                "Value Improvement %": 8,
                "Confidence %": 97,
                "Function Status": "Governance protection",
            }
        )

    # Default recommendation if inputs are relatively healthy.
    if not actions:
        actions.append(
            {
                "Priority": 1,
                "Action": (
                    "Maintain the current option and continue weekly value monitoring."
                ),
                "Value Engineering Reason": (
                    "No major value loss trigger is currently detected."
                ),
                "Expected Delay Recovery Days": 0,
                "Expected Cost Impact %": 0.0,
                "Value Improvement %": 3,
                "Confidence %": 82,
                "Function Status": "Preserved",
            }
        )

    actions_df = pd.DataFrame(actions)

    # Rank based on value, confidence, recovery, and lower cost impact.
    actions_df["Recommendation Score"] = (
        actions_df["Value Improvement %"] * 0.40
        + actions_df["Confidence %"] * 0.25
        + actions_df["Expected Delay Recovery Days"] * 1.8
        - actions_df["Expected Cost Impact %"].clip(lower=0) * 1.4
    )

    actions_df = (
        actions_df.sort_values(
            ["Recommendation Score", "Confidence %"],
            ascending=[False, False],
        )
        .drop_duplicates(subset=["Action"])
        .reset_index(drop=True)
    )

    actions_df["Priority"] = range(1, len(actions_df) + 1)

    estimated_recovery = int(
        min(
            delay_days,
            actions_df.head(3)["Expected Delay Recovery Days"].sum(),
        )
    )

    estimated_cost_effect = float(
        actions_df.head(3)["Expected Cost Impact %"].sum()
    )

    estimated_value_improvement = float(
        actions_df.head(3)["Value Improvement %"].mean()
    )

    confidence = float(
        actions_df.head(3)["Confidence %"].mean()
    )

    function_status = (
        "At Risk"
        if function_score < 80
        else "Protected"
    )

    return {
        "Root Causes": root_causes or [
            "No major trigger was detected from the selected inputs."
        ],
        "Warnings": warnings,
        "Actions": actions_df,
        "Estimated Recovery Days": estimated_recovery,
        "Estimated Remaining Delay Days": max(0, delay_days - estimated_recovery),
        "Estimated Cost Effect %": estimated_cost_effect,
        "Estimated Value Improvement %": estimated_value_improvement,
        "Overall Confidence %": confidence,
        "Function Status": function_status,
    }


# =========================================================
# MAIN INTERFACE
# =========================================================
st.markdown(
    '<div class="main-title">🏗️ MARSAD — AI Value Engineering Platform</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
    Function analysis, lifecycle cost optimization, risk intelligence,
    alternative evaluation, and decision simulation in one platform.
    </div>
    """,
    unsafe_allow_html=True,
)

# Main platform banner displayed directly below the title and subtitle.
#st.image("marsad_banner.png", width="stretch")

default_data = get_sample_data()

with st.sidebar:
    st.header("Data Management")

    template_file = create_excel_template(default_data)

    st.download_button(
        "Download Excel Template",
        data=template_file,
        file_name="marsad_project_template.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

    uploaded_file = st.file_uploader(
        "Upload the completed Excel file",
        type=["xlsx"],
        help=(
            "Required sheets: Project, Environment, Activities, "
            "Materials, Suppliers, Risks, and Resources."
        ),
    )

    use_sample_data = st.toggle(
        "Use sample data",
        value=uploaded_file is None,
    )

    st.divider()
    st.subheader("Material Decision Weights")

    weight_cost = st.slider(
        "Lifecycle Cost",
        0,
        100,
        30,
    )

    weight_energy = st.slider(
        "Energy Performance",
        0,
        100,
        20,
    )

    weight_carbon = st.slider(
        "Environmental Impact",
        0,
        100,
        15,
    )

    weight_resilience = st.slider(
        "Quality and Resilience",
        0,
        100,
        20,
    )

    weight_supply = st.slider(
        "Supply Reliability",
        0,
        100,
        15,
    )

    comparison_area = st.number_input(
        "Comparison Area m2",
        min_value=100.0,
        value=4000.0,
        step=100.0,
    )

try:
    if uploaded_file is not None and not use_sample_data:
        data = load_excel(uploaded_file)
        st.success("Excel file loaded and validated successfully.")
    else:
        data = default_data

except Exception as error:
    st.error(f"Unable to read the Excel file: {error}")
    st.info("The application returned to sample data.")
    data = default_data

project = data["Project"]
environment = data["Environment"]
activities = data["Activities"]
materials = data["Materials"]
suppliers = data["Suppliers"]
risks = data["Risks"]
resources = data["Resources"]

weights = {
    "Cost": weight_cost,
    "Energy": weight_energy,
    "Carbon": weight_carbon,
    "Resilience": weight_resilience,
    "Supply": weight_supply,
}

material_results = analyze_materials(
    materials=materials,
    suppliers=suppliers,
    project=project,
    environment=environment,
    target_area=comparison_area,
    weights=weights,
)

risk_results = analyze_risks(
    risks=risks,
    activities=activities,
    suppliers=suppliers,
    environment=environment,
)

schedule_results = analyze_schedule(activities)

health = calculate_project_health(
    project=project,
    schedule=schedule_results,
    risks=risk_results,
    material_results=material_results,
)

best_material = material_results.iloc[0]

decision_dna = calculate_decision_dna(
    health=health,
    best_material=best_material,
    risk_results=risk_results,
)

recommendations = generate_recommendations(
    health=health,
    best_material=best_material,
    risk_results=risk_results,
    schedule_results=schedule_results,
)


st.markdown(
    """
    <div class="section-banner">
        <b>MARSAD transforms Value Engineering from a one-time workshop
        into a continuous, data-driven decision system.</b><br>
        It compares functions, alternatives, lifecycle cost, risks,
        schedule impact, sustainability, procurement reliability,
        and long-term project value before implementation.
    </div>
    """,
    unsafe_allow_html=True,
)

overview_col_1, overview_col_2, overview_col_3, overview_col_4 = st.columns(4)

with overview_col_1:
    st.markdown(
        """
        <div class="value-card">
            <h4>Function First</h4>
            Defines the required project function before comparing solutions.
            Prevents the team from selecting an expensive option that does not
            improve performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_col_2:
    st.markdown(
        """
        <div class="value-card">
            <h4>Lifecycle Value</h4>
            Measures initial cost, operation, energy, maintenance,
            replacement, supplier risk, and end-of-life impact.
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_col_3:
    st.markdown(
        """
        <div class="value-card">
            <h4>Alternative Intelligence</h4>
            Ranks alternatives using transparent weighted criteria,
            Value Index, and Decision Regret Index.
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_col_4:
    st.markdown(
        """
        <div class="value-card">
            <h4>Value Protection</h4>
            Detects risks, critical activities, procurement weaknesses,
            and cost leakage before value is lost.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

tabs = st.tabs(
    [
        "Executive Dashboard",
        "Data Center",
        "Material Intelligence",
        "Proactive Risk Engine",
        "Value Intelligence",
        "What-If Lab",
        "Value DNA",
        "AI Value Engineer",
        "ML Delay Predictor",
        "Reports",
    ]
)


# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================
with tabs[0]:
    st.markdown(
        """
        <div class="section-banner">
            <b>Executive Dashboard</b><br>
            A single management view for project value, success probability,
            overrun exposure, major risks, recommended alternative,
            and immediate executive actions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What this dashboard answers", expanded=True):
        st.markdown(
            """
            - Are we preserving the required project functions?
            - Which decision currently creates the highest value?
            - What is the probability of schedule or budget overrun?
            - Which risks require executive intervention?
            - Which alternative gives the best lifecycle value?
            - What action should be approved today?
            """
        )

    with st.expander("How the indicators should be interpreted"):
        st.markdown(
            """
            **Project Success Probability:** combined readiness based on schedule,
            cost, risk, supply chain, and material resilience.

            **Overall Risk Index:** the weighted level of exposure across active
            risks and project constraints.

            **Expected Monetary Loss:** probability-weighted financial impact,
            not the guaranteed final loss.

            **Weighted Expected Delay:** probability-weighted delay exposure
            across the risk register.
            """
        )
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Project Success Probability",
        f"{health['Project Success Probability']:.1f}%",
    )

    metric_2.metric(
        "Schedule Overrun Probability",
        f"{health['Schedule Overrun Probability']:.1f}%",
    )

    metric_3.metric(
        "Budget Overrun Probability",
        f"{health['Budget Overrun Probability']:.1f}%",
    )

    metric_4.metric(
        "Overall Risk Index",
        f"{health['Overall Risk Index']:.1f}/100",
    )

    metric_5, metric_6, metric_7 = st.columns(3)

    metric_5.metric(
        "Weighted Expected Delay",
        f"{health['Expected Delay Days']:.1f} days",
    )

    metric_6.metric(
        "Expected Monetary Loss",
        f"{health['Expected Loss SAR']:,.0f} SAR",
    )

    metric_7.metric(
        "Current Best Material",
        str(best_material["Material Name"]),
    )

    left, right = st.columns([1.15, 1])

    with left:
        highest_risks = risk_results.head(7).copy()

        risk_chart = px.bar(
            highest_risks.sort_values(
                "Proactive Risk Index"
            ),
            x="Proactive Risk Index",
            y="Risk Description",
            orientation="h",
            text_auto=".1f",
            title="Highest Proactive Risks",
        )

        st.plotly_chart(
            apply_marsad_chart_theme(risk_chart),
            use_container_width=True,
        )

    with right:
        success_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health[
                    "Project Success Probability"
                ],
                title={
                    "text": "Project Success Readiness"
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#FFFFFF",
                        "tickfont": {"color": "#FFFFFF"},
                    },
                    "bar": {"color": "#006C35", "thickness": 0.30},
                    "bgcolor": "#6D47A3",
                    "bordercolor": "rgba(255,255,255,0.55)",
                    "borderwidth": 2,
                    "steps": [
                        {"range": [0, 100], "color": "rgba(216,193,159,0.30)"}
                    ],
                },
            )
        )

        success_gauge.update_layout(height=360)

        st.plotly_chart(
            apply_marsad_chart_theme(success_gauge),
            use_container_width=True,
        )

    st.subheader("AI Decision Recommendations")

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        st.markdown(
            f"**{index}.** {recommendation}"
        )


# =========================================================
# DATA CENTER
# =========================================================
with tabs[1]:
    st.markdown(
        """
        <div class="section-banner">
            <b>Value Engineering Data Center</b><br>
            Enter or upload project assumptions, function requirements,
            alternatives, suppliers, risks, activities, and resources.
            These inputs feed every calculation in the platform.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Recommended data preparation steps", expanded=True):
        st.markdown(
            """
            1. Confirm the project scope and analysis period.
            2. Define the main functions and performance requirements.
            3. Add at least three alternatives for every major decision.
            4. Use realistic supplier, lead-time, maintenance, and service-life data.
            5. Record probability values as decimals between 0 and 1.
            6. Link risks to affected activities and suppliers whenever possible.
            7. Review assumptions with the project team before final ranking.
            """
        )

    with st.expander("Data governance and validation"):
        st.markdown(
            """
            - Keep column names unchanged when using the Excel template.
            - Record the source and date of every major assumption.
            - Separate verified data from estimated data.
            - Update supplier reliability after every delivery.
            - Review lifecycle assumptions whenever the project scope changes.
            - Do not use the platform output as a substitute for professional approval.
            """
        )
    st.info(
        "You can edit the tables for testing. "
        "The original uploaded Excel file will not be changed."
    )

    data_tabs = st.tabs(
        [
            "Project",
            "Environment",
            "Activities",
            "Materials",
            "Suppliers",
            "Risks",
            "Resources",
        ]
    )

    edited_project = data_tabs[0].data_editor(
        project,
        use_container_width=True,
        num_rows="dynamic",
    )

    edited_environment = data_tabs[1].data_editor(
        environment,
        use_container_width=True,
        num_rows="dynamic",
    )

    edited_activities = data_tabs[2].data_editor(
        activities,
        use_container_width=True,
        num_rows="dynamic",
    )

    edited_materials = data_tabs[3].data_editor(
        materials,
        use_container_width=True,
        num_rows="dynamic",
    )

    edited_suppliers = data_tabs[4].data_editor(
        suppliers,
        use_container_width=True,
        num_rows="dynamic",
    )

    edited_risks = data_tabs[5].data_editor(
        risks,
        use_container_width=True,
        num_rows="dynamic",
    )

    edited_resources = data_tabs[6].data_editor(
        resources,
        use_container_width=True,
        num_rows="dynamic",
    )

    edited_file = export_excel(
        {
            "Project": edited_project,
            "Environment": edited_environment,
            "Activities": edited_activities,
            "Materials": edited_materials,
            "Suppliers": edited_suppliers,
            "Risks": edited_risks,
            "Resources": edited_resources,
        }
    )

    st.download_button(
        "Download Edited Excel File",
        data=edited_file,
        file_name="marsad_edited_data.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# =========================================================
# MATERIAL INTELLIGENCE
# =========================================================
with tabs[2]:
    st.markdown(
        """
        <div class="section-banner">
            <b>Material Value Intelligence</b><br>
            Compare construction materials using function achievement,
            lifecycle cost, energy performance, maintenance,
            replacement, carbon, quality, and supply reliability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Value Engineering logic used in this section", expanded=True):
        st.markdown(
            """
            **Step 1 — Define the function:** What must the material achieve?
            Examples: thermal insulation, fire resistance, durability,
            acoustic control, visual quality, or local-content requirements.

            **Step 2 — Generate alternatives:** Compare technically acceptable
            alternatives only.

            **Step 3 — Calculate lifecycle cost:** Initial cost is combined with
            operation, maintenance, energy, and replacement present value.

            **Step 4 — Measure performance:** Quality, resilience,
            availability, and supply reliability are included.

            **Step 5 — Rank value:** The platform calculates an AI Value Index
            and a Decision Regret Index.
            """
        )

    with st.expander("How to interpret the material outputs"):
        st.markdown(
            """
            - **AI Value Index:** higher is better.
            - **Decision Regret Index:** lower is better.
            - **Lifecycle Cost:** total discounted ownership cost.
            - **Supply Chain Risk:** exposure to supplier failure and lead time.
            - **Material Resilience:** expected ability to maintain performance.
            - **Replacement Years:** when major replacement is expected.
            """
        )

    st.markdown(
        """
        <div class="formula-box">
        Value = Required Function Performance ÷ Lifecycle Cost
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader(
        "Material Alternatives Across the Building Lifecycle"
    )

    material_metric_1, material_metric_2, material_metric_3 = st.columns(3)

    material_metric_1.metric(
        "Recommended Alternative",
        str(best_material["Material Name"]),
    )

    material_metric_2.metric(
        "Lifecycle Cost",
        f"{best_material['Lifecycle Cost SAR']:,.0f} SAR",
    )

    material_metric_3.metric(
        "AI Value Index",
        f"{best_material['AI Value Index']:.1f}/100",
    )

    material_columns = [
        "Material Name",
        "Category",
        "Supplier Name",
        "Total Initial Cost SAR",
        "Annual Energy Cost SAR",
        "Annual Maintenance Cost SAR",
        "Lifecycle Cost SAR",
        "Lifecycle Carbon kgCO2e",
        "Supply Chain Risk",
        "AI Value Index",
        "Decision Regret Index",
        "Replacement Years",
    ]

    st.dataframe(
        material_results[material_columns].style.format(
            {
                "Total Initial Cost SAR": "{:,.0f}",
                "Annual Energy Cost SAR": "{:,.0f}",
                "Annual Maintenance Cost SAR": "{:,.0f}",
                "Lifecycle Cost SAR": "{:,.0f}",
                "Lifecycle Carbon kgCO2e": "{:,.0f}",
                "Supply Chain Risk": "{:.2f}",
                "AI Value Index": "{:.1f}",
                "Decision Regret Index": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:
        value_chart = px.scatter(
            material_results,
            x="Lifecycle Cost SAR",
            y="AI Value Index",
            size="Material Resilience",
            hover_name="Material Name",
            hover_data=[
                "Supplier Name",
                "Lifecycle Carbon kgCO2e",
            ],
            title="AI Value versus Lifecycle Cost",
        )

        st.plotly_chart(
            apply_marsad_chart_theme(value_chart),
            use_container_width=True,
        )

    with right:
        regret_chart = px.bar(
            material_results.head(6),
            x="Material Name",
            y="Decision Regret Index",
            title="Decision Regret Index — Lower is Better",
        )

        st.plotly_chart(
            apply_marsad_chart_theme(regret_chart),
            use_container_width=True,
        )

    st.markdown(
        """
        **Innovation:** The platform does not select the lowest initial price.
        It combines acquisition cost, energy, maintenance, replacement,
        carbon, supplier risk, and long-term resilience.

        The **Decision Regret Index** estimates the future penalty of selecting
        an option that appears attractive today but becomes weaker over time.
        """
    )


# =========================================================
# PROACTIVE RISK ENGINE
# =========================================================
with tabs[3]:
    st.markdown(
        """
        <div class="section-banner">
            <b>Proactive Risk and Value Protection Engine</b><br>
            Identifies where project value may be lost because of delay,
            cost escalation, supply disruption, resource constraints,
            weather, or scope change.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Risk analysis methodology", expanded=True):
        st.markdown(
            """
            The platform adjusts the original risk probability using:
            activity delay probability, supplier default probability,
            climate severity, cost impact, time impact, and project criticality.

            Each risk receives:
            - Adjusted Probability
            - Composite Impact
            - Proactive Risk Index
            - Expected Monetary Loss
            - Weighted Expected Delay
            - Recommended Response
            """
        )

    with st.expander("Value protection actions"):
        st.markdown(
            """
            - Protect functions that are essential to project success.
            - Avoid cost reduction that weakens required performance.
            - Prioritize risks connected to critical-path activities.
            - Create backup suppliers for high-value materials.
            - Use dynamic contingency instead of a fixed reserve only.
            - Re-run alternative analysis after major risk changes.
            """
        )
    st.subheader("AI-Enhanced Proactive Risk Register")

    risk_columns = [
        "Risk ID",
        "Risk Description",
        "Category",
        "Activity Name",
        "Adjusted Probability",
        "Composite Impact",
        "Proactive Risk Index",
        "Risk Level",
        "Expected Monetary Loss SAR",
        "Weighted Expected Delay Days",
        "Recommended Response",
    ]

    st.dataframe(
        risk_results[risk_columns].style.format(
            {
                "Adjusted Probability": "{:.1%}",
                "Composite Impact": "{:.2f}",
                "Proactive Risk Index": "{:.1f}",
                "Expected Monetary Loss SAR": "{:,.0f}",
                "Weighted Expected Delay Days": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:
        matrix = px.scatter(
            risk_results,
            x="Adjusted Probability",
            y="Composite Impact",
            size="Cost Impact SAR",
            hover_name="Risk Description",
            color="Risk Level",
            title="Dynamic Risk Matrix",
        )

        st.plotly_chart(
            apply_marsad_chart_theme(matrix),
            use_container_width=True,
        )

    with right:
        grouped_risks = risk_results.groupby(
            "Category",
            as_index=False,
        ).agg(
            Expected_Loss_SAR=(
                "Expected Monetary Loss SAR",
                "sum",
            ),
            Weighted_Delay_Days=(
                "Weighted Expected Delay Days",
                "sum",
            ),
        )

        category_chart = px.bar(
            grouped_risks,
            x="Category",
            y="Expected_Loss_SAR",
            hover_data=["Weighted_Delay_Days"],
            title="Expected Loss by Risk Category",
        )

        st.plotly_chart(
            apply_marsad_chart_theme(category_chart),
            use_container_width=True,
        )


# =========================================================
# DIGITAL TWIN
# =========================================================
with tabs[4]:
    st.markdown(
        """
        <div class="section-banner">
            <b>Integrated Project Value Intelligence</b><br>
            Connects schedule logic, activity criticality, duration uncertainty,
            and value impact so that the team can see how one decision
            affects the whole project.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What this section provides", expanded=True):
        st.markdown(
            """
            - Early start and finish dates
            - Late start and finish dates
            - Total float
            - Critical activities
            - Risk-adjusted duration
            - Network duration
            - Decision effect on procurement, labor, cost, and project value
            """
        )

    with st.expander("Value Ripple Engine"):
        st.markdown(
            """
            The engine follows the chain of impact created by a decision.

            **Example:** Change facade material → change supplier →
            increase lead time → pressure critical activity →
            idle installation team → increase indirect cost →
            change cash flow → reduce overall project value.
            """
        )
    st.subheader("Integrated Project Value Intelligence")

    schedule_metric_1, schedule_metric_2, schedule_metric_3 = st.columns(3)

    schedule_metric_1.metric(
        "Network Duration",
        f"{schedule_results['Early Finish'].max():.0f} days",
    )

    schedule_metric_2.metric(
        "Critical Activities",
        int(
            schedule_results[
                "Critical Activity"
            ].eq("Yes").sum()
        ),
    )

    schedule_metric_3.metric(
        "Total Activities",
        len(schedule_results),
    )

    gantt_data = schedule_results.copy()
    start_date = pd.Timestamp.today().normalize()

    gantt_data["Start"] = (
        start_date
        + pd.to_timedelta(
            gantt_data["Early Start"],
            unit="D",
        )
    )

    gantt_data["Finish"] = (
        start_date
        + pd.to_timedelta(
            gantt_data["Early Finish"],
            unit="D",
        )
    )

    gantt_chart = px.timeline(
        gantt_data,
        x_start="Start",
        x_end="Finish",
        y="Activity Name",
        color="Critical Activity",
        hover_data=[
            "Activity ID",
            "Total Float Days",
            "Delay Probability",
        ],
        title="Integrated Project Schedule and Value Timeline",
    )

    gantt_chart.update_yaxes(
        autorange="reversed"
    )

    st.plotly_chart(
        apply_marsad_chart_theme(gantt_chart),
        use_container_width=True,
    )

    st.dataframe(
        schedule_results[
            [
                "Activity ID",
                "Activity Name",
                "Predecessors",
                "Planned Duration Days",
                "Early Start",
                "Early Finish",
                "Total Float Days",
                "Critical Activity",
                "Risk-Adjusted Duration Days",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **Value Ripple Engine:** A decision does not remain inside one
        department. Its effect moves through the project network.

        Example:

        Material change → Supplier change → Longer lead time →
        Critical path pressure → Labor idle time → Cash-flow change →
        Carbon change → Lower project success probability.
        """
    )


# =========================================================
# WHAT-IF LAB
# =========================================================
with tabs[5]:
    st.markdown(
        """
        <div class="section-banner">
            <b>What-If Value Laboratory</b><br>
            Simulate uncertain events and compare their effect on delay,
            cost, risk, and project success before approving a decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Available scenario variables", expanded=True):
        st.markdown(
            """
            **Supplier Delay:** tests procurement disruption.

            **Labor Change:** tests productivity improvement or workforce shortage.

            **Material Price Change:** tests inflation and market volatility.

            **Weather Disruption:** tests loss of productive days.

            **Mitigation Strength:** estimates how much of the impact can be reduced
            by early action.
            """
        )

    with st.expander("How to use scenario results"):
        st.markdown(
            """
            1. Run the base case.
            2. Test the expected scenario.
            3. Test a worst-case scenario.
            4. Increase mitigation strength to assess recovery.
            5. Compare project success probability.
            6. Select the option that protects required functions
               at the best lifecycle value.
            """
        )
    st.subheader("What-If Decision Laboratory")

    st.caption(
        "Test a decision before implementation and measure "
        "its impact on time, cost, risk, and success probability."
    )

    scenario_column_1, scenario_column_2, scenario_column_3 = st.columns(3)

    supplier_delay = scenario_column_1.slider(
        "Supplier Delay Days",
        0,
        90,
        14,
    )

    labor_change = scenario_column_2.slider(
        "Labor Change Percent",
        -40,
        60,
        0,
    )

    material_cost_change = scenario_column_3.slider(
        "Material Price Change Percent",
        -15,
        40,
        5,
    )

    scenario_column_4, scenario_column_5 = st.columns(2)

    weather_delay = scenario_column_4.slider(
        "Weather Disruption Days",
        0,
        45,
        5,
    )

    mitigation_strength = scenario_column_5.slider(
        "Mitigation Strength",
        0.0,
        1.0,
        0.45,
        0.05,
    )

    scenario = run_scenario(
        base_health=health,
        project=project,
        supplier_delay_days=supplier_delay,
        labor_change_percent=labor_change,
        material_cost_change_percent=material_cost_change,
        weather_delay_days=weather_delay,
        mitigation_strength=mitigation_strength,
    )

    scenario_metric_1, scenario_metric_2, scenario_metric_3, scenario_metric_4 = st.columns(4)

    scenario_metric_1.metric(
        "Delay After Mitigation",
        f"{scenario['Expected Delay Days']:.1f} days",
        delta=(
            f"{scenario['Expected Delay Days'] - health['Expected Delay Days']:.1f}"
        ),
    )

    scenario_metric_2.metric(
        "Additional Cost",
        f"{scenario['Additional Cost SAR']:,.0f} SAR",
        delta=(
            f"{scenario['Additional Cost SAR'] - health['Expected Loss SAR']:,.0f}"
        ),
    )

    scenario_metric_3.metric(
        "Risk Index",
        f"{scenario['Overall Risk Index']:.1f}/100",
    )

    scenario_metric_4.metric(
        "Success Probability",
        f"{scenario['Project Success Probability']:.1f}%",
    )

    comparison = pd.DataFrame(
        {
            "Indicator": [
                "Schedule Overrun Probability",
                "Budget Overrun Probability",
                "Overall Risk Index",
                "Project Success Probability",
            ],
            "Current Condition": [
                health["Schedule Overrun Probability"],
                health["Budget Overrun Probability"],
                health["Overall Risk Index"],
                health["Project Success Probability"],
            ],
            "Scenario": [
                scenario["Schedule Overrun Probability"],
                scenario["Budget Overrun Probability"],
                scenario["Overall Risk Index"],
                scenario["Project Success Probability"],
            ],
        }
    )

    comparison_chart = px.bar(
        comparison.melt(
            id_vars="Indicator",
            var_name="Condition",
            value_name="Value",
        ),
        x="Indicator",
        y="Value",
        color="Condition",
        barmode="group",
        title="Current Condition versus Simulated Scenario",
    )

    st.plotly_chart(
        apply_marsad_chart_theme(comparison_chart),
        use_container_width=True,
    )


# =========================================================
# DECISION DNA
# =========================================================
with tabs[6]:
    st.markdown(
        """
        <div class="section-banner">
            <b>Value DNA</b><br>
            A multi-dimensional fingerprint for each major decision.
            It shows where the alternative is strong, weak,
            balanced, or exposed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Value DNA dimensions", expanded=True):
        st.markdown(
            """
            - **Economic Value:** lifecycle cost efficiency.
            - **Sustainability:** environmental performance.
            - **Supply Resilience:** supplier and availability strength.
            - **Project Resilience:** ability to absorb uncertainty.
            - **Schedule Safety:** protection from time overrun.
            - **Recovery Capacity:** ability to recover after disruption.
            """
        )

    with st.expander("Decision approval guidance"):
        st.markdown(
            """
            - Approve when required functions are fully achieved.
            - Review when one dimension falls below the acceptable threshold.
            - Reject when low cost is achieved by reducing an essential function.
            - Require more simulation for high-impact, low-reversibility decisions.
            - Record the final decision and actual outcome for future learning.
            """
        )
    st.subheader("Value DNA")

    st.write(
        "Instead of evaluating a decision with one number, "
        "MARSAD evaluates every value-engineering decision across six connected dimensions."
    )

    dimensions = list(decision_dna.keys())
    scores = list(decision_dna.values())

    radar_chart = go.Figure()

    radar_chart.add_trace(
        go.Scatterpolar(
            r=scores + [scores[0]],
            theta=dimensions + [dimensions[0]],
            fill="toself",
            name="Current Decision DNA",
        )
    )

    radar_chart.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
            }
        },
        showlegend=False,
        height=520,
    )

    st.plotly_chart(
        apply_marsad_chart_theme(radar_chart),
        use_container_width=True,
    )

    decision_dna_table = pd.DataFrame(
        {
            "Dimension": dimensions,
            "Score": scores,
        }
    )

    st.dataframe(
        decision_dna_table.style.format(
            {"Score": "{:.1f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        ### MARSAD Value Engineering Method

        **1. Integrated Value Model**  
        Represents functions, schedule, cost, materials, suppliers, resources,
        risks, and environmental conditions.

        **2. Value Ripple Engine**  
        Measures the direct and indirect effect of every project decision.

        **3. Decision Regret Index**  
        Estimates the future loss caused by selecting a decision that looks
        attractive now but becomes weaker over time.

        **4. AI Decision Memory**  
        Stores each scenario, decision reason, expected result, and later
        actual outcome so future recommendations become more accurate.

        **5. Reversibility Gate**  
        High-impact decisions that are difficult to reverse require stronger
        approval and more simulation before execution.

        This combination creates a differentiated Value Engineering hackathon concept.
        It should not be presented as guaranteed to be the first concept
        ever created worldwide without a formal prior-art review.
        """
    )




# =========================================================
# AI VALUE ENGINEER
# =========================================================
with tabs[7]:
    st.markdown(
        """
        <div class="section-banner">
            <b>AI Value Engineer</b><br>
            An explainable Value Engineering consultant that converts project
            problems into prioritized actions while protecting required function,
            lifecycle value, schedule, quality, and implementation feasibility.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("How this consultant supports Value Engineering", expanded=True):
        st.markdown(
            """
            The consultant does not simply recommend the cheapest or fastest action.
            It follows a Value Engineering sequence:

            1. Protect the required function.
            2. Identify the probable source of value loss.
            3. Generate technically acceptable alternatives.
            4. Compare schedule, cost, risk, lifecycle, and implementation impact.
            5. Rank the actions by value improvement and confidence.
            6. Show warnings when a decision weakens function or lifecycle performance.
            """
        )

    with st.expander("Example questions this section can answer"):
        st.markdown(
            """
            - What should we do if a critical supplier is delayed?
            - Can we recover the project without excessive overtime?
            - Which action reduces cost while preserving the required function?
            - Is an alternative material truly better over the full lifecycle?
            - Should we activate a backup supplier or re-sequence the schedule?
            - What is the expected delay recovery, cost effect, and value improvement?
            """
        )

    input_col_1, input_col_2, input_col_3 = st.columns(3)

    with input_col_1:
        issue_type = st.selectbox(
            "Primary Issue",
            [
                "Project Delay",
                "Supplier Delay",
                "Material Availability",
                "Cost Overrun",
                "Weather Disruption",
                "Labor Shortage",
                "Function Underperformance",
                "Lifecycle Cost Concern",
                "Combined Project Problem",
            ],
            key="ai_value_issue_type",
        )

        delay_days = st.number_input(
            "Current Delay (Days)",
            min_value=0,
            max_value=365,
            value=18,
            step=1,
            key="ai_value_delay_days",
        )

        cost_overrun_pct = st.number_input(
            "Forecast Cost Overrun (%)",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.5,
            key="ai_value_cost_overrun",
        )

    with input_col_2:
        supplier_reliability = st.slider(
            "Supplier Reliability (%)",
            min_value=0,
            max_value=100,
            value=72,
            key="ai_value_supplier_reliability",
        )

        function_score = st.slider(
            "Required Function Achievement (%)",
            min_value=0,
            max_value=100,
            value=92,
            key="ai_value_function_score",
        )

        lifecycle_impact = st.selectbox(
            "Expected Lifecycle Impact",
            ["Positive", "Neutral", "Negative"],
            index=1,
            key="ai_value_lifecycle_impact",
        )

    with input_col_3:
        weather_days = st.number_input(
            "Expected Weather Disruption (Days)",
            min_value=0,
            max_value=90,
            value=3,
            step=1,
            key="ai_value_weather_days",
        )

        labor_change_pct = st.slider(
            "Labor Capacity Change (%)",
            min_value=-50,
            max_value=50,
            value=-10,
            key="ai_value_labor_change",
        )

        critical_activity = st.checkbox(
            "Affected Activity Is Critical",
            value=True,
            key="ai_value_critical_activity",
        )

        reversible_decision = st.checkbox(
            "Decision Is Easily Reversible",
            value=False,
            key="ai_value_reversible",
        )

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    custom_problem = st.text_area(
        "Project Problem Description",
        value=(
            "The steel supplier is delayed by 18 days. The affected activity "
            "is critical, and replacing the supplier may increase initial cost."
        ),
        height=110,
        key="ai_value_problem_description",
    )

    run_consultant = st.button(
        "Generate AI Value Engineering Action Plan",
        use_container_width=True,
        key="run_ai_value_engineer",
    )

    if run_consultant:
        result = build_value_engineering_action_plan(
            issue_type=issue_type + " " + custom_problem,
            delay_days=int(delay_days),
            cost_overrun_pct=float(cost_overrun_pct),
            supplier_reliability=float(supplier_reliability),
            function_score=float(function_score),
            lifecycle_impact=lifecycle_impact,
            weather_days=int(weather_days),
            labor_change_pct=float(labor_change_pct),
            critical_activity=bool(critical_activity),
            reversible_decision=bool(reversible_decision),
        )

        st.session_state["ai_value_result"] = result

    if "ai_value_result" in st.session_state:
        result = st.session_state["ai_value_result"]
        actions_df = result["Actions"]

        st.subheader("AI Value Engineering Assessment")

        metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

        metric_1.metric(
            "Function Status",
            result["Function Status"],
        )
        metric_2.metric(
            "Expected Recovery",
            f'{result["Estimated Recovery Days"]} days',
        )
        metric_3.metric(
            "Remaining Delay",
            f'{result["Estimated Remaining Delay Days"]} days',
        )
        metric_4.metric(
            "Value Improvement",
            f'{result["Estimated Value Improvement %"]:.1f}%',
        )
        metric_5.metric(
            "Confidence",
            f'{result["Overall Confidence %"]:.1f}%',
        )

        if result["Function Status"] == "At Risk":
            st.error(
                "Function Warning: the current option may not achieve the "
                "required function. Cost or schedule savings should not be "
                "approved until performance is restored."
            )
        else:
            st.success(
                "Function Check: the required function is currently protected, "
                "subject to technical validation of the selected action."
            )

        if result["Warnings"]:
            with st.expander("Critical Decision Warnings", expanded=True):
                for warning in result["Warnings"]:
                    st.warning(warning)

        with st.expander("Probable Root Causes", expanded=True):
            for cause in result["Root Causes"]:
                st.markdown(f"- {cause}")

        st.subheader("Prioritized AI Action Plan")

        display_columns = [
            "Priority",
            "Action",
            "Value Engineering Reason",
            "Expected Delay Recovery Days",
            "Expected Cost Impact %",
            "Value Improvement %",
            "Confidence %",
            "Function Status",
        ]

        st.dataframe(
            actions_df[display_columns].style.format(
                {
                    "Expected Cost Impact %": "{:+.1f}%",
                    "Value Improvement %": "{:.1f}%",
                    "Confidence %": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        top_action = actions_df.iloc[0]

        st.markdown(
            f"""
            <div class="value-card">
                <h4>Recommended Decision</h4>
                <b>{top_action['Action']}</b><br><br>
                <b>Why:</b> {top_action['Value Engineering Reason']}<br>
                <b>Expected delay recovery:</b>
                {int(top_action['Expected Delay Recovery Days'])} days<br>
                <b>Expected cost effect:</b>
                {top_action['Expected Cost Impact %']:+.1f}%<br>
                <b>Value improvement:</b>
                {top_action['Value Improvement %']:.1f}%<br>
                <b>Confidence:</b>
                {top_action['Confidence %']:.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

        chart_col_1, chart_col_2 = st.columns(2)

        with chart_col_1:
            value_chart = px.bar(
                actions_df.head(8),
                x="Value Improvement %",
                y="Action",
                orientation="h",
                title="Expected Value Improvement by Action",
            )
            value_chart.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(apply_marsad_chart_theme(value_chart), use_container_width=True)

        with chart_col_2:
            recovery_chart = px.scatter(
                actions_df.head(8),
                x="Expected Cost Impact %",
                y="Expected Delay Recovery Days",
                size="Value Improvement %",
                hover_name="Action",
                title="Cost versus Schedule Recovery",
            )
            st.plotly_chart(apply_marsad_chart_theme(recovery_chart), use_container_width=True)

        st.subheader("Implementation Roadmap")

        immediate_actions = actions_df.head(2)["Action"].tolist()
        weekly_actions = actions_df.iloc[2:5]["Action"].tolist()
        monitoring_actions = [
            "Confirm technical equivalence and required function achievement.",
            "Update lifecycle cost and risk assumptions after approval.",
            "Track actual delay recovery and cost effect weekly.",
            "Record the final result in the project decision memory.",
        ]

        roadmap_col_1, roadmap_col_2, roadmap_col_3 = st.columns(3)

        with roadmap_col_1:
            st.markdown(
                """
                <div class="value-card">
                    <h4>Immediate — Today</h4>
                """,
                unsafe_allow_html=True,
            )
            for action in immediate_actions:
                st.markdown(f"- {action}")
            st.markdown("</div>", unsafe_allow_html=True)

        with roadmap_col_2:
            st.markdown(
                """
                <div class="value-card">
                    <h4>This Week</h4>
                """,
                unsafe_allow_html=True,
            )
            if weekly_actions:
                for action in weekly_actions:
                    st.markdown(f"- {action}")
            else:
                st.markdown("- Validate and implement the selected action.")
            st.markdown("</div>", unsafe_allow_html=True)

        with roadmap_col_3:
            st.markdown(
                """
                <div class="value-card">
                    <h4>Value Monitoring</h4>
                """,
                unsafe_allow_html=True,
            )
            for action in monitoring_actions:
                st.markdown(f"- {action}")
            st.markdown("</div>", unsafe_allow_html=True)

        action_plan_export = export_excel(
            {
                "AI Action Plan": actions_df,
                "Assessment": pd.DataFrame(
                    {
                        "Indicator": [
                            "Issue",
                            "Function Status",
                            "Expected Recovery Days",
                            "Remaining Delay Days",
                            "Estimated Cost Effect %",
                            "Estimated Value Improvement %",
                            "Overall Confidence %",
                        ],
                        "Result": [
                            issue_type,
                            result["Function Status"],
                            result["Estimated Recovery Days"],
                            result["Estimated Remaining Delay Days"],
                            result["Estimated Cost Effect %"],
                            result["Estimated Value Improvement %"],
                            result["Overall Confidence %"],
                        ],
                    }
                ),
            }
        )

        st.download_button(
            "Download AI Value Engineering Action Plan",
            data=action_plan_export,
            file_name="marsad_ai_value_engineering_action_plan.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
            key="download_ai_value_action_plan",
        )

        st.caption(
            "Hackathon MVP: recommendations are generated by transparent rules "
            "and should be validated by the project, engineering, commercial, "
            "and safety teams before implementation."
        )



# =========================================================
# ML DELAY PREDICTOR
# =========================================================
with tabs[8]:
    st.markdown(
        """
        <div class="section-banner">
            <b>Supervised Machine-Learning Delay Predictor</b><br>
            A real Random Forest model is trained on reproducible synthetic
            schedule data and predicts project delay probability and expected
            delay days from user inputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Hackathon evidence: Input → supervised ML model → prediction → "
        "risk alert → explanation → recommended action."
    )

    model_bundle = train_delay_models()
    evidence_1, evidence_2, evidence_3 = st.columns(3)
    evidence_1.metric(
        "Validation Accuracy",
        f"{model_bundle['accuracy'] * 100:.1f}%",
    )
    evidence_2.metric(
        "Delay-Day MAE",
        f"{model_bundle['mae']:.1f} days",
    )
    evidence_3.metric(
        "Synthetic Training Records",
        f"{model_bundle['training_rows']:,}",
    )

    with st.expander("How the AI model works", expanded=True):
        st.markdown(
            """
            - **Model type:** Random Forest classifier and regressor.
            - **Training data:** synthetic project schedule records generated
              with controlled delay, cost, workforce, supplier, risk, and
              weather relationships.
            - **Classifier output:** probability that the project will be delayed.
            - **Regressor output:** estimated number of delay days.
            - **Explainability:** feature importance identifies the factors that
              most influenced the model overall.
            - **Purpose:** demonstration-grade MVP, not a production prediction
              system or a substitute for professional project controls.
            """
        )

    st.subheader("Enter Current Project Conditions")
    ml_col_1, ml_col_2, ml_col_3 = st.columns(3)

    with ml_col_1:
        ml_progress = st.slider(
            "Progress %",
            0,
            100,
            42,
            key="ml_progress",
        )
        ml_time_elapsed = st.slider(
            "Time Elapsed %",
            0,
            120,
            58,
            key="ml_time_elapsed",
        )
        ml_cost_variance = st.slider(
            "Cost Variance %",
            -20,
            40,
            9,
            key="ml_cost_variance",
        )

    with ml_col_2:
        ml_delayed_critical = st.number_input(
            "Delayed Critical Activities",
            min_value=0,
            max_value=30,
            value=3,
            step=1,
            key="ml_delayed_critical",
        )
        ml_workforce = st.slider(
            "Workforce Availability %",
            40,
            120,
            78,
            key="ml_workforce",
        )
        ml_supplier_reliability = st.slider(
            "Supplier Reliability %",
            30,
            100,
            72,
            key="ml_supplier_reliability",
        )

    with ml_col_3:
        ml_open_risks = st.number_input(
            "Open Risks",
            min_value=0,
            max_value=50,
            value=9,
            step=1,
            key="ml_open_risks",
        )
        ml_average_risk = st.slider(
            "Average Risk Probability %",
            0,
            100,
            46,
            key="ml_average_risk",
        )
        ml_weather_days = st.number_input(
            "Weather Disruption Days",
            min_value=0,
            max_value=60,
            value=6,
            step=1,
            key="ml_weather_days",
        )

    run_ml_prediction = st.button(
        "Run Machine-Learning Prediction",
        use_container_width=True,
        key="run_ml_delay_prediction",
    )

    if run_ml_prediction:
        ml_inputs = {
            "Progress %": float(ml_progress),
            "Time Elapsed %": float(ml_time_elapsed),
            "Cost Variance %": float(ml_cost_variance),
            "Delayed Critical Activities": float(ml_delayed_critical),
            "Workforce Availability %": float(ml_workforce),
            "Supplier Reliability %": float(ml_supplier_reliability),
            "Open Risks": float(ml_open_risks),
            "Average Risk Probability %": float(ml_average_risk),
            "Weather Disruption Days": float(ml_weather_days),
        }

        ml_result = predict_project_delay(ml_inputs)

        output_1, output_2, output_3 = st.columns(3)
        output_1.metric(
            "Delay Probability",
            f"{ml_result['Delay Probability']:.1f}%",
        )
        output_2.metric(
            "Predicted Delay",
            f"{ml_result['Predicted Delay Days']:.1f} days",
        )
        output_3.metric(
            "ML Risk Level",
            ml_result["Risk Level"],
        )

        if ml_result["Risk Level"] in ["Critical", "High"]:
            st.error(
                f"{ml_result['Risk Level']} delay alert: immediate management "
                "intervention is recommended."
            )
        elif ml_result["Risk Level"] == "Medium":
            st.warning(
                "Medium delay exposure: activate preventive actions and monitor weekly."
            )
        else:
            st.success(
                "Low predicted delay exposure under the current inputs."
            )

        st.subheader("Model Explanation")
        importance_chart = px.bar(
            ml_result["Feature Importance"].sort_values("Importance"),
            x="Importance",
            y="Factor",
            orientation="h",
            title="Random Forest Feature Importance",
            hover_data=["Current Value"],
        )
        st.plotly_chart(
            apply_marsad_chart_theme(importance_chart),
            use_container_width=True,
        )

        st.subheader("AI-Generated Management Actions")
        for recommendation in ml_result["Recommendations"]:
            st.markdown(f"- {recommendation}")

        prediction_report = pd.DataFrame(
            {
                "Indicator": [
                    "Delay Probability",
                    "Predicted Delay Days",
                    "Risk Level",
                    "Validation Accuracy",
                    "Validation MAE Days",
                    "Synthetic Training Records",
                ],
                "Value": [
                    f"{ml_result['Delay Probability']:.2f}%",
                    f"{ml_result['Predicted Delay Days']:.2f}",
                    ml_result["Risk Level"],
                    f"{ml_result['Model Accuracy']:.2f}%",
                    f"{ml_result['Model MAE']:.2f}",
                    ml_result["Training Rows"],
                ],
            }
        )

        ml_export = export_excel(
            {
                "Prediction": prediction_report,
                "Input Data": pd.DataFrame([ml_inputs]),
                "Feature Importance": ml_result["Feature Importance"],
                "Recommendations": pd.DataFrame(
                    {"Recommended Action": ml_result["Recommendations"]}
                ),
            }
        )

        st.download_button(
            "Download ML Prediction Report",
            data=ml_export,
            file_name="marsad_ml_delay_prediction.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    with st.expander("Download the synthetic demonstration data"):
        synthetic_csv = model_bundle["data"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Synthetic Schedule Dataset",
            data=synthetic_csv,
            file_name="marsad_synthetic_schedule_training_data.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# REPORTS
# =========================================================
with tabs[9]:
    st.markdown(
        """
        <div class="section-banner">
            <b>Reports and Value Engineering Documentation</b><br>
            Export the executive summary, alternatives analysis,
            risk analysis, schedule analysis, Value DNA,
            and recommended actions into one Excel report.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Recommended report structure", expanded=True):
        st.markdown(
            """
            1. Project and workshop information
            2. Function definition
            3. Current design or baseline option
            4. Generated alternatives
            5. Technical feasibility screening
            6. Lifecycle cost analysis
            7. Risk and schedule impact
            8. Weighted evaluation
            9. Recommended alternative
            10. Implementation and monitoring plan
            """
        )

    with st.expander("What should be reviewed before submission"):
        st.markdown(
            """
            - Confirm all figures and units.
            - Verify that alternatives achieve the required function.
            - Record assumptions and data sources.
            - Explain why the recommended option creates higher value.
            - Add responsible owners and implementation dates.
            - Obtain technical, financial, and project-management approval.
            """
        )
    st.subheader("Downloadable Executive Report")

    executive_summary = pd.DataFrame(
        {
            "Indicator": [
                "Project Success Probability",
                "Schedule Overrun Probability",
                "Budget Overrun Probability",
                "Overall Risk Index",
                "Weighted Expected Delay",
                "Expected Monetary Loss",
                "Recommended Material",
                "Recommended Material Lifecycle Cost",
                "Recommended Material AI Value Index",
                "Decision Regret Index",
            ],
            "Value": [
                f"{health['Project Success Probability']:.1f}%",
                f"{health['Schedule Overrun Probability']:.1f}%",
                f"{health['Budget Overrun Probability']:.1f}%",
                f"{health['Overall Risk Index']:.1f}/100",
                f"{health['Expected Delay Days']:.1f} days",
                f"{health['Expected Loss SAR']:,.0f} SAR",
                str(best_material["Material Name"]),
                f"{best_material['Lifecycle Cost SAR']:,.0f} SAR",
                f"{best_material['AI Value Index']:.1f}/100",
                f"{best_material['Decision Regret Index']:.1f}/100",
            ],
        }
    )

    recommendation_table = pd.DataFrame(
        {
            "Priority": range(
                1,
                len(recommendations) + 1,
            ),
            "Recommendation": recommendations,
        }
    )

    st.dataframe(
        executive_summary,
        use_container_width=True,
        hide_index=True,
    )

    report_file = export_excel(
        {
            "Executive Summary": executive_summary,
            "Recommendations": recommendation_table,
            "Material Analysis": material_results,
            "Risk Analysis": risk_results,
            "Schedule Analysis": schedule_results,
            "Value DNA": pd.DataFrame(
                {
                    "Dimension": dimensions,
                    "Score": scores,
                }
            ),
            **(
                {
                    "AI Value Action Plan":
                        st.session_state["ai_value_result"]["Actions"]
                }
                if "ai_value_result" in st.session_state
                else {}
            ),
        }
    )

    st.download_button(
        "Download Analysis Report",
        data=report_file,
        file_name="marsad_analysis_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

    pitch_text = f"""
MARSAD is an AI-powered Value Engineering platform for construction projects.
It does not only show what has happened; it simulates what may happen
before a decision is approved.

The platform combines function, lifecycle cost, schedule, budget, materials, suppliers,
resources, risks, and environmental data in one connected value model.
It evaluates material lifecycle cost, predicts risk and delay,
identifies critical activities, tests scenarios such as supplier
delay, labor changes, weather disruption, and price increases, and
generates prioritized Value Engineering recovery actions.

Its main innovation is the Value Ripple Engine, the Decision
Regret Index, and the explainable AI Value Engineer. Every decision is evaluated not only by its immediate
benefit, but also by its future impact on time, cost, sustainability,
supply resilience, and project success.

The current model estimates a project success probability of
{health['Project Success Probability']:.1f}% and recommends
{best_material['Material Name']} as the current best material option.
"""

    st.text_area(
        "Hackathon Pitch",
        pitch_text.strip(),
        height=260,
    )


st.caption(
    "MARSAD Value Engineering MVP. Future integrations may include BIM, "
    "Primavera P6, Power BI, weather APIs, IoT sensors, ERP systems, "
    "cost databases, and supplier portals."
)
