import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import locale
import os
import re


def format_currency(value):
    if pd.isna(value):
        return 'N/A'
    return "${:,.2f}".format(value)

# MRK Partners Color Palette
MRK_COLORS = {
    'orange': '#F47C20',     # Primary accent color
    'navy': '#0C223A',       # Deep Navy - primary color for charts
    'charcoal': '#333333',   # Text color
    'white': '#FFFFFF',      # Background
    'gray': '#F4F4F4',       # Section dividers, cards, tables
}

# Additional blue shades to complement the navy
BLUE_PALETTE = [
    '#0C223A',  # Deep Navy (darkest)
    '#1A3A5F',  # Darker blue
    '#2A5684',  # Dark blue
    '#3973A9',  # Medium blue
    '#4B92D3',  # Light blue
    '#75ADD8',  # Lighter blue
]

# Status colors
STATUS_COLORS = {
    'good': '#43A047',       # Green for good status
    'warning': '#FFA000',    # Amber for warnings
    'alert': '#E53935'       # Red for alerts/problems
}

# Custom color sequences for charts
AREA_COLORS = [BLUE_PALETTE[2], BLUE_PALETTE[4], MRK_COLORS['orange']]
BAR_COLORS = [BLUE_PALETTE[1], BLUE_PALETTE[3], BLUE_PALETTE[5], MRK_COLORS['orange']]
LINE_COLORS = [MRK_COLORS['orange'], BLUE_PALETTE[2], BLUE_PALETTE[4]]
PIE_COLORS = [BLUE_PALETTE[1], BLUE_PALETTE[3], BLUE_PALETTE[5], MRK_COLORS['orange']]

# Set page configuration
st.set_page_config(
    page_title="MRK Portfolio Water Analytics",
    page_icon="mrk.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to style the dashboard with MRK colors
st.markdown(f"""
<style>
    /* Base styling */
    .reportview-container .main .block-container {{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }}
    
    /* Header styling */
    h1, h2, h3 {{
        color: {MRK_COLORS['navy']};
    }}
    
    /* Text styling */
    p, li, div {{
        color: {MRK_COLORS['charcoal']};
    }}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {{
        font-size: 2rem !important;
        color: {MRK_COLORS['navy']};
    }}
    
    [data-testid="stMetricDelta"] {{
        color: {MRK_COLORS['orange']} !important;
    }}
    
    /* Metric label styling */
    [data-testid="stMetricLabel"] {{
        font-weight: 600 !important;
        color: {MRK_COLORS['charcoal']};
    }}
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {{
        background-color: {MRK_COLORS['gray']};
    }}
    
    /* Sidebar title */
    .sidebar-title {{
        background-color: {MRK_COLORS['navy']};
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 15px;
    }}
    
    /* Highlight box styling */
    .highlight-box {{
        background-color: {MRK_COLORS['navy']};
        color: white;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 15px;
        text-align: center;
    }}
    
    .highlight-box h2, .highlight-box h3 {{
        color: white !important;
        margin-top: 0;
    }}
    
    .highlight-value {{
        font-size: 2.5rem;
        margin: 15px 0;
    }}
    
    /* Card styling */
    .card {{
        background-color: {MRK_COLORS['white']};
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }}
    
    /* Status card styling */
    .status-card {{
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        color: white;
    }}
    
    .status-card h4 {{
        margin-top: 0;
        color: white !important;
    }}
    
    .status-card.good {{
        background-color: {STATUS_COLORS['good']};
    }}
    
    .status-card.warning {{
        background-color: {STATUS_COLORS['warning']};
    }}
    
    .status-card.alert {{
        background-color: {STATUS_COLORS['alert']};
    }}
    
    /* Info box styling */
    .info-box {{
        background-color: {BLUE_PALETTE[5]};
        color: {MRK_COLORS['white']};
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }}
    
    /* Warning box styling */
    .warning-box {{
        background-color: {MRK_COLORS['orange']};
        color: {MRK_COLORS['white']};
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }}
    
    /* Footer styling */
    .footer {{
        background-color: {MRK_COLORS['navy']};
        color: {MRK_COLORS['white']};
        padding: 20px;
        text-align: center;
        border-radius: 5px;
        margin-top: 30px;
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: {MRK_COLORS['orange']};
        color: {MRK_COLORS['white']};
        border: none;
        font-weight: 600;
    }}
    
    .stButton>button:hover {{
        background-color: {MRK_COLORS['navy']};
        color: {MRK_COLORS['white']};
    }}
    
    /* Tab styling */
    div[data-baseweb="tab-list"] {{
        background-color: {MRK_COLORS['gray']};
        border-radius: 5px;
    }}
    
    button[data-baseweb="tab"] {{
        color: {MRK_COLORS['charcoal']};
        font-weight: 600;
    }}
    
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {MRK_COLORS['navy']};
        color: {MRK_COLORS['white']};
    }}
    
    /* Selectbox styling */
    div[data-baseweb="select"] {{
        background-color: {MRK_COLORS['white']};
    }}
    
    /* Dataframe styling */
    .dataframe {{
        font-family: Arial, sans-serif;
    }}
    
    .dataframe th {{
        background-color: {MRK_COLORS['navy']};
        color: {MRK_COLORS['white']};
        padding: 10px;
    }}
    
    .dataframe td {{
        padding: 10px;
    }}
    
    /* Contact info box */
    .contact-box {{
        background-color: {BLUE_PALETTE[4]};
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
        text-align: center;
    }}
    
    .contact-box h4 {{
        color: white !important;
        margin-top: 0;
    }}
    
    /* Feature box styling */
    .feature-box {{
        background-color: {MRK_COLORS['white']};
        border-left: 5px solid {MRK_COLORS['navy']};
        padding: 15px;
        height: 180px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }}
    
    .feature-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }}
    
    .feature-box h4 {{
        color: {MRK_COLORS['navy']};
        margin-top: 0;
    }}
    
    .feature-box p.accent {{
        color: {MRK_COLORS['orange']};
        font-weight: bold;
    }}
    
    /* Improve Streamlit elements */
    div.stTabs button {{
        font-size: 16px;
    }}
    
    /* Loading animation */
    div.stSpinner > div {{
        border-top-color: {MRK_COLORS['orange']} !important;
    }}
    
    /* Logo container */
    .logo-container {{
        text-align: center;
        margin-bottom: 20px;
    }}
    
    /* Enhanced tooltips */
    .tooltip {{
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted {MRK_COLORS['navy']};
    }}
    
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: {MRK_COLORS['navy']};
        color: {MRK_COLORS['white']};
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
</style>
""", unsafe_allow_html=True)

# Helper functions
def format_currency(value):
    if pd.isna(value):
        return 'N/A'
    return "${:,.2f}".format(value)

def format_percent(value):
    if pd.isna(value):
        return 'N/A'
    return f"{value:.1f}%"

def format_number(value, decimals=0):
    if pd.isna(value):
        return 'N/A'
    return f"{value:,.{decimals}f}"

# Enhanced data validation function
def validate_csv_data(df):
    """Validate the uploaded CSV data and return issues found"""
    issues = []
    
    # Check for required columns
    required_cols = ["month", "year", "totalUsage", "totalCost"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for data completeness
    for col in [c for c in required_cols if c in df.columns]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            issues.append(f"Column '{col}' contains {null_count} missing values")
    
    # Check for data types
    if "totalUsage" in df.columns and not pd.api.types.is_numeric_dtype(df["totalUsage"]):
        issues.append("Column 'totalUsage' should contain numeric values")
    
    if "totalCost" in df.columns and not pd.api.types.is_numeric_dtype(df["totalCost"]):
        issues.append("Column 'totalCost' should contain numeric values")
    
    # Check for negative values
    if "totalUsage" in df.columns and (df["totalUsage"] < 0).any():
        issues.append("Column 'totalUsage' contains negative values")
    
    if "totalCost" in df.columns and (df["totalCost"] < 0).any():
        issues.append("Column 'totalCost' contains negative values")
    
    # Check data range
    if "year" in df.columns:
        years = df["year"].unique()
        if len(years) == 0:
            issues.append("No year data found")
        elif min(years) < 2000 or max(years) > datetime.now().year + 1:
            issues.append(f"Year values outside reasonable range: {min(years)} - {max(years)}")
    
    return issues

# Enhanced error handling decorator
def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            if "monthlyData" in kwargs:
                st.warning("Please check your input data and try again.")
            return None
    return wrapper

# Create app version and build date
APP_VERSION = "1.2.0"
BUILD_DATE = "April 29, 2025"

# Configure sidebar with MRK branding
st.sidebar.markdown("""
<div class="sidebar-title">
    <h3 style="margin:0; color:white;">MRK Water Analytics</h3>
    <p style="margin:0; font-size:0.8rem; color:white;">v{}</p>
</div>
""".format(APP_VERSION), unsafe_allow_html=True)

# File upload section
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV data file", type=["csv"])

# Property selection for multi-property setups
st.sidebar.header("Properties")
property_options = ["All Properties", "KT1", "KT2", "KT3", "KT3", "KT4", "Add New Property..."]
selected_property = st.sidebar.selectbox("Select Property:", property_options)

# Tier structure definition (this could also be loaded from a separate CSV)
tier_structure = pd.DataFrame([
    {"tier": 1, "minGpd": 0, "maxGpd": 80.9999, "waterRate": 6.53, "sewerRate": 8.67},
    {"tier": 2, "minGpd": 81, "maxGpd": 165.9999, "waterRate": 7.38, "sewerRate": 9.63},
    {"tier": 3, "minGpd": 166, "maxGpd": 275.9999, "waterRate": 8.50, "sewerRate": 12.09},
    {"tier": 4, "minGpd": 276, "maxGpd": float('inf'), "waterRate": 9.96, "sewerRate": 15.97}
])

# Utility allowance settings
utility_allowance = {
    "average": 0,
    "recommended": 0,
    "min": 0,
    "max": 0,
    "stdDev": 0
}

# Add sidebar options for advanced settings
with st.sidebar.expander("Advanced Settings"):
    use_custom_tier_rates = st.checkbox("Use Custom Tier Rates", False)
    if use_custom_tier_rates:
        st.write("Configure custom tier rates:")
        for i, tier in tier_structure.iterrows():
            tier_num = int(tier['tier'])
            col1, col2 = st.columns(2)
            with col1:
                new_water_rate = st.number_input(f"Tier {tier_num} Water Rate", min_value=0.01, value=tier['waterRate'], step=0.01, format="%.2f")
                tier_structure.at[i, 'waterRate'] = new_water_rate
            with col2:
                new_sewer_rate = st.number_input(f"Tier {tier_num} Sewer Rate", min_value=0.01, value=tier['sewerRate'], step=0.01, format="%.2f")
                tier_structure.at[i, 'sewerRate'] = new_sewer_rate
    
    # Add configuration for utility allowance calculation
    st.write("Utility Allowance Configuration:")
    allowance_buffer = st.slider("Allowance Buffer (σ)", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                                help="Number of standard deviations to add to the average cost for the recommended allowance")

# Add help and resources in sidebar
with st.sidebar.expander("Help & Resources"):
    st.markdown("""
    ### Quick Tips
    - Upload a CSV with Month, Year, Total_Usage, and Total_Effective_Expense columns
    - Custom columns can be mapped after upload
    - Data will be validated automatically
    
    ### Documentation
    - [User Guide](idk yet)
    - [Data Format Requirements](idk yet)
    
    ### Troubleshooting
    Please contact the Portfolio Analysts for assistance:
    - Matthew Dixon (mdixon@mrkpartners.com)
    - Josh Blankstein (jblankstein@mrkpartners.com)
    """)

# Function to load and process CSV data
def load_data(file):
    # Load the data
    data = pd.read_csv(file)
    
    # Data preprocessing
    st.sidebar.subheader("Data Columns")
    
    # Display column information in the sidebar
    st.sidebar.write("Available columns:")
    st.sidebar.write(", ".join(data.columns))
    
    # Column mapping (with smart defaults or user selection)
    month_col = st.sidebar.selectbox("Month column:", data.columns, 
                                   index=data.columns.get_loc("Month") if "Month" in data.columns else 0)
    year_col = st.sidebar.selectbox("Year column:", data.columns, 
                                  index=data.columns.get_loc("Year") if "Year" in data.columns else 0)
    usage_col = st.sidebar.selectbox("Usage column:", data.columns, 
                                   index=data.columns.get_loc("Total_Usage") if "Total_Usage" in data.columns else 0)
    cost_col = st.sidebar.selectbox("Cost column:", data.columns, 
                                  index=data.columns.get_loc("Total_Effective_Expense") if "Total_Effective_Expense" in data.columns else 0)
    
    # Optional columns mapping
    with st.sidebar.expander("Additional Column Mapping (Optional)"):
        property_col = None
        if "Property" in data.columns or "Property_ID" in data.columns:
            property_options = [None] + list(data.columns)
            property_col = st.selectbox("Property column:", property_options, 
                                    index=property_options.index("Property") if "Property" in property_options else
                                    (property_options.index("Property_ID") if "Property_ID" in property_options else 0))
        
        days_col = None
        if "Days_In_Period" in data.columns:
            days_options = [None] + list(data.columns)
            days_col = st.selectbox("Days in period column:", days_options,
                                 index=days_options.index("Days_In_Period") if "Days_In_Period" in days_options else 0)
    
    # Rename columns for consistency
    data_mapped = data.rename(columns={
        month_col: "month",
        year_col: "year",
        usage_col: "totalUsage",
        cost_col: "totalCost"
    })
    
    # Map property column if specified
    if property_col:
        data_mapped["property"] = data[property_col]
    
    # Filter by selected property if applicable
    if "property" in data_mapped.columns and selected_property != "All Properties":
        # Create a fuzzy matching for property names
        property_names = data_mapped["property"].unique()
        matching_properties = [p for p in property_names if selected_property.lower() in p.lower()]
        
        if matching_properties:
            data_mapped = data_mapped[data_mapped["property"].isin(matching_properties)]
            st.sidebar.success(f"Filtered to show only: {', '.join(matching_properties)}")
        else:
            st.sidebar.warning(f"No properties found matching '{selected_property}'")
    
    # Calculate derived metrics if not present
    if "dailyUsage" not in data_mapped.columns and days_col:
        data_mapped["dailyUsage"] = data_mapped["totalUsage"] / data[days_col]
    
    if "averageRate" not in data_mapped.columns:
        # Handle division by zero
        data_mapped["averageRate"] = data_mapped.apply(
            lambda row: row["totalCost"] / row["totalUsage"] if row["totalUsage"] > 0 else 0, 
            axis=1
        )
    
    # If tier information is in the data, use it
    tier_col = None
    if "Tier" in data.columns:
        tier_col = "Tier"
    elif "tier" in data.columns:
        tier_col = "tier"
    
    if tier_col:
        data_mapped["tier"] = data[tier_col]
    else:
        # Assign tiers based on daily usage if available
        if "dailyUsage" in data_mapped.columns:
            data_mapped["tier"] = data_mapped["dailyUsage"].apply(lambda x: 
                1 if x < 81 else (
                2 if x < 166 else (
                3 if x < 276 else 4
                ))
            )
    
    # Validate the data after mapping
    issues = validate_csv_data(data_mapped)
    if issues:
        st.sidebar.warning("Data validation warnings:")
        for issue in issues:
            st.sidebar.warning(f"- {issue}")
    
    # Process monthly data
    try:
        monthly_data = data_mapped.groupby(["month", "year"]).agg({
            "totalUsage": "sum",
            "totalCost": "sum"
        }).reset_index()
        
        # Calculate average rate
        monthly_data["averageRate"] = monthly_data.apply(
            lambda row: row["totalCost"] / row["totalUsage"] if row["totalUsage"] > 0 else 0,
            axis=1
        )
        
        # Sort by year and month for proper ordering
        month_order = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        # Try to sort by month if it has text representation
        try:
            # First check for month format - could be text or number
            first_month = monthly_data['month'].iloc[0]
            if isinstance(first_month, str):
                # Handle text months
                monthly_data["month_num"] = monthly_data["month"].map(lambda x: month_order.get(x, 0))
                if monthly_data["month_num"].min() == 0:
                    # Try extract numeric part if month format is like "01-Jan"
                    monthly_data["month_num"] = monthly_data["month"].apply(
                        lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
                    )
            else:
                # Already numeric
                monthly_data["month_num"] = monthly_data["month"]
                
            monthly_data = monthly_data.sort_values(["year", "month_num"])
            monthly_data = monthly_data.drop("month_num", axis=1)
        except Exception as e:
            st.sidebar.warning(f"Couldn't auto-sort by month: {str(e)}")
            # If sorting fails, keep as is
            pass
        
        # Calculate YoY changes
        monthly_data_with_yoy = calculate_yoy_changes(monthly_data)
        
        # Prepare yearly data
        yearly_data = monthly_data.groupby("year").agg({
            "totalUsage": "sum",
            "totalCost": "sum"
        }).reset_index()
        yearly_data["averageRate"] = yearly_data.apply(
            lambda row: row["totalCost"] / row["totalUsage"] if row["totalUsage"] > 0 else 0,
            axis=1
        )
        
        # Calculate YoY for yearly data
        yearly_data = calculate_yearly_yoy(yearly_data)
        
        # Calculate cost drivers
        cost_driver_data = calculate_cost_drivers(monthly_data_with_yoy)
        
        # Prepare tier data
        if "dailyUsage" in data_mapped.columns and "tier" in data_mapped.columns:
            tier_data = data_mapped[["month", "year", "totalUsage", "dailyUsage", "tier", "totalCost"]].copy()
            
            # Add tier rate based on the tier structure
            tier_data["tierRate"] = tier_data["tier"].apply(lambda x: 
                tier_structure.loc[tier_structure["tier"] == x, "waterRate"].values[0]
                if x in tier_structure["tier"].values else 0
            )
            
            # Calculate tier-based charges
            tier_data["tierBasedCharge"] = tier_data["totalUsage"] * tier_data["tierRate"] / 1000  # Assuming rate is per kgal
            tier_data["actualCharge"] = tier_data["totalCost"]
            tier_data["rateDifference"] = tier_data["actualCharge"] - tier_data["tierBasedCharge"]
            tier_data["usageExpense"] = tier_data["tierBasedCharge"]
            tier_data["nonUsageExpense"] = tier_data["rateDifference"]
        else:
            # Create a simplified tier dataframe with available columns
            tier_data = monthly_data_with_yoy.copy()
            # Add placeholder data for missing columns
            if "tier" not in tier_data.columns:
                tier_data["tier"] = 1
            if "dailyUsage" not in tier_data.columns:
                tier_data["dailyUsage"] = tier_data["totalUsage"] / 30  # Assuming 30 days/month
            
            tier_data["tierRate"] = tier_data["tier"].apply(lambda x: 
                tier_structure.loc[tier_structure["tier"] == x, "waterRate"].values[0]
                if x in tier_structure["tier"].values else 0
            )
            tier_data["tierBasedCharge"] = tier_data["totalCost"] * 0.9  # Placeholder calculation
            tier_data["actualCharge"] = tier_data["totalCost"]
            tier_data["rateDifference"] = tier_data["actualCharge"] - tier_data["tierBasedCharge"]
            tier_data["usageExpense"] = tier_data["tierBasedCharge"]
            tier_data["nonUsageExpense"] = tier_data["rateDifference"]
        
        # Calculate utility allowance metrics using the configured buffer
        calc_utility_allowance(monthly_data, buffer_factor=allowance_buffer)
        
        return monthly_data_with_yoy, tier_data, yearly_data, cost_driver_data
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error("Please check your data format and try again.")
        return None, None, None, None

def calculate_yoy_changes(monthly_data):
    """Calculate year-over-year changes for monthly data"""
    # Create a copy of the DataFrame
    df = monthly_data.copy()
    
    # Ensure data is sorted by month and year
    df = df.sort_values(['month', 'year'])
    
    # Initialize YoY columns
    df['usageYoY'] = np.nan
    df['costYoY'] = np.nan
    df['rateYoY'] = np.nan
    
    # Group by month to calculate YoY changes
    for month in df['month'].unique():
        month_data = df[df['month'] == month].copy()
        
        if len(month_data) > 1:
            # Sort by year
            month_data = month_data.sort_values('year')
            
            # Calculate percentage changes
            for i in range(1, len(month_data)):
                prev_usage = month_data.iloc[i-1]['totalUsage']
                prev_cost = month_data.iloc[i-1]['totalCost']
                prev_rate = month_data.iloc[i-1]['averageRate']
                
                if prev_usage > 0:
                    usage_yoy = 100 * (month_data.iloc[i]['totalUsage'] - prev_usage) / prev_usage
                    df.loc[(df['month'] == month) & (df['year'] == month_data.iloc[i]['year']), 'usageYoY'] = usage_yoy
                
                if prev_cost > 0:
                    cost_yoy = 100 * (month_data.iloc[i]['totalCost'] - prev_cost) / prev_cost
                    df.loc[(df['month'] == month) & (df['year'] == month_data.iloc[i]['year']), 'costYoY'] = cost_yoy
                
                if prev_rate > 0:
                    rate_yoy = 100 * (month_data.iloc[i]['averageRate'] - prev_rate) / prev_rate
                    df.loc[(df['month'] == month) & (df['year'] == month_data.iloc[i]['year']), 'rateYoY'] = rate_yoy
    
    return df

def calculate_yearly_yoy(yearly_data):
    """Calculate year-over-year changes for yearly data"""
    # Create a copy and ensure it's sorted by year
    df = yearly_data.copy().sort_values('year')
    
    # Initialize YoY columns
    df['usageYoY'] = np.nan
    df['costYoY'] = np.nan
    df['rateYoY'] = np.nan
    
    # Calculate YoY changes
    for i in range(1, len(df)):
        prev_usage = df.iloc[i-1]['totalUsage']
        prev_cost = df.iloc[i-1]['totalCost']
        prev_rate = df.iloc[i-1]['averageRate']
        
        if prev_usage > 0:
            df.iloc[i, df.columns.get_loc('usageYoY')] = 100 * (df.iloc[i]['totalUsage'] - prev_usage) / prev_usage
        
        if prev_cost > 0:
            df.iloc[i, df.columns.get_loc('costYoY')] = 100 * (df.iloc[i]['totalCost'] - prev_cost) / prev_cost
        
        if prev_rate > 0:
            df.iloc[i, df.columns.get_loc('rateYoY')] = 100 * (df.iloc[i]['averageRate'] - prev_rate) / prev_rate
    
    return df

def calculate_cost_drivers(monthly_data):
    """Calculate cost drivers from monthly data with YoY changes"""
    # Filter to only rows with both YoY metrics
    df = monthly_data.dropna(subset=['usageYoY', 'costYoY']).copy()
    
    # Add driver analysis columns
    df['primaryDriver'] = df.apply(
        lambda row: 'Usage' if abs(row['usageYoY']) > abs(row['costYoY'] - row['usageYoY']) else 'Rate', 
        axis=1
    )
    
    # Calculate contribution percentages
    df['usageContribution'] = df.apply(
        lambda row: min(100, max(0, (row['usageYoY'] / row['costYoY']) * 100)) if row['costYoY'] != 0 else 0,
        axis=1
    )
    
    df['rateContribution'] = df.apply(
        lambda row: 100 - row['usageContribution'] if row['costYoY'] != 0 else 0,
        axis=1
    )
    
    return df

def calc_utility_allowance(monthly_data, buffer_factor=1.0):
    """
    Calculate utility allowance recommendations
    
    Parameters:
    - monthly_data: DataFrame with monthly cost data
    - buffer_factor: Number of standard deviations to add (default: 1.0)
    """
    global utility_allowance
    
    if len(monthly_data) > 0:
        # Get the most recent 12 months if available
        if len(monthly_data) > 12:
            # Sort by year and month
            if 'month_num' not in monthly_data.columns:
                month_order = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
                    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
                }
                # Try to create a sortable date
                try:
                    # If month is string and year is numeric
                    if isinstance(monthly_data['month'].iloc[0], str) and pd.api.types.is_numeric_dtype(monthly_data['year']):
                        monthly_data['temp_date'] = monthly_data.apply(
                            lambda row: f"{int(row['year'])}-{month_order.get(row['month'], 1):02d}-01", 
                            axis=1
                        )
                        monthly_data['temp_date'] = pd.to_datetime(monthly_data['temp_date'])
                        recent_data = monthly_data.sort_values('temp_date', ascending=False).head(12)
                    else:
                        # Default sort by year descending, then month
                        recent_data = monthly_data.sort_values(['year', 'month'], ascending=[False, True]).head(12)
                except:
                    # Fallback to just using all data
                    recent_data = monthly_data
            else:
                recent_data = monthly_data.sort_values(['year', 'month_num'], ascending=[False, True]).head(12)
        else:
            recent_data = monthly_data
        
        costs = recent_data['totalCost'].values
        
        utility_allowance["average"] = np.mean(costs)
        utility_allowance["min"] = np.min(costs)
        utility_allowance["max"] = np.max(costs)
        utility_allowance["stdDev"] = np.std(costs)
        
        # Recommended allowance: average + buffer_factor * standard deviation
        utility_allowance["recommended"] = utility_allowance["average"] + (buffer_factor * utility_allowance["stdDev"])


# Dashboard Header with MRK branding
st.markdown(f"""
    <div style='text-align: center;'>
        <h1 style='color: {MRK_COLORS["navy"]}; font-size: 2.5rem;'>MRK Portfolio Water Analytics</h1>
        <p style='color: {MRK_COLORS["charcoal"]}; font-size: 1.2rem; margin-bottom: 10px;'>Real Estate Investment Portfolio Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Load data or show sample options
if uploaded_file is not None:
    try:
        with st.spinner("Processing data..."):
            monthly_data, tier_data, yearly_data, cost_driver_data = load_data(uploaded_file)
        
        if monthly_data is not None:
            st.sidebar.success("✅ Data successfully loaded!")
            
            # Add date range information to title
            if not monthly_data.empty:
                min_date = f"{monthly_data['month'].iloc[0]} {monthly_data['year'].iloc[0]}"
                max_date = f"{monthly_data['month'].iloc[-1]} {monthly_data['year'].iloc[-1]}"
                st.markdown(f"""
                    <div style='text-align: center; margin-bottom: 20px;'>
                        <p style='color: {MRK_COLORS["navy"]}; font-size: 1.1rem;'>
                            <strong>Analysis Period:</strong> {min_date} to {max_date}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Overview", 
                "Monthly Analysis", 
                "Year-over-Year", 
                "Cost Drivers", 
                "Tier Analysis", 
                "Utility Allowance"
            ])
            
            # Tab 1: Overview
            with tab1:
                st.header("Water Utility Overview")
                
                # Top metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Monthly Water Cost", 
                        value=format_currency(utility_allowance["average"]),
                        help="Average (Last 12 Months)"
                    )
                
                with col2:
                    st.metric(
                        label="Recommended Allowance", 
                        value=format_currency(utility_allowance["recommended"]),
                        help="Based on historical usage + buffer"
                    )
                
                if not yearly_data.empty and 'costYoY' in yearly_data.columns:
                    recent_year = yearly_data.iloc[-1]
                    cost_yoy_value = recent_year["costYoY"] if not pd.isna(recent_year["costYoY"]) else 0
                    cost_yoy_delta = f"{cost_yoy_value:+.1f}%" if not pd.isna(cost_yoy_value) else None
                    
                    with col3:
                        st.metric(
                            label="Annual Cost Trend", 
                            value=format_percent(cost_yoy_value),
                            delta=cost_yoy_delta,
                            delta_color="inverse",
                            help="Year-over-Year Change"
                        )
                    
                    if 'usageYoY' in yearly_data.columns:
                        primary_driver = "Rates" if cost_yoy_value > recent_year["usageYoY"] else "Usage"
                        
                        with col4:
                            st.metric(
                                label="Primary Cost Driver", 
                                value=primary_driver,
                                help="Based on YoY Analysis"
                            )
                
                st.markdown("""---""")
                
                # Status summary with alerts
                st.subheader("Status Summary")
                
                # Determine alerts
                alert_col1, alert_col2, alert_col3 = st.columns(3)
                
                # Cost trend alert
                with alert_col1:
                    if not yearly_data.empty and 'costYoY' in yearly_data.columns:
                        recent_cost_yoy = recent_year["costYoY"] if not pd.isna(recent_year["costYoY"]) else 0
                        if recent_cost_yoy > 10:
                            status_class = "alert"
                            status_msg = f"Water costs increased by {format_percent(recent_cost_yoy)} year-over-year, significantly above inflation."
                        elif recent_cost_yoy > 5:
                            status_class = "warning"
                            status_msg = f"Water costs increased by {format_percent(recent_cost_yoy)} year-over-year."
                        else:
                            status_class = "good"
                            status_msg = f"Water cost changes are within normal range ({format_percent(recent_cost_yoy)})."
                    else:
                        status_class = "warning"
                        status_msg = "Insufficient data to determine cost trend."
                    
                    st.markdown(f"""
                        <div class="status-card {status_class}">
                            <h4>Cost Trend</h4>
                            <p>{status_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Utility allowance alert
                with alert_col2:
                    if utility_allowance["average"] > 0:
                        current_vs_recommended = (utility_allowance["max"] / utility_allowance["recommended"]) * 100
                        if current_vs_recommended > 120:
                            status_class = "alert"
                            status_msg = f"Maximum monthly cost exceeds recommended allowance by {current_vs_recommended - 100:.1f}%."
                        elif current_vs_recommended > 100:
                            status_class = "warning"
                            status_msg = f"Maximum monthly cost exceeds recommended allowance by {current_vs_recommended - 100:.1f}%."
                        else:
                            status_class = "good"
                            status_msg = "Current recommended allowance covers all historical costs."
                    else:
                        status_class = "warning"
                        status_msg = "Insufficient data to calculate utility allowance."
                    
                    st.markdown(f"""
                        <div class="status-card {status_class}">
                            <h4>Utility Allowance</h4>
                            <p>{status_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Tier distribution alert
                with alert_col3:
                    if 'tier' in tier_data.columns:
                        tier_counts = tier_data["tier"].value_counts().to_dict()
                        highest_tier = max(tier_counts.keys()) if tier_counts else 0
                        if highest_tier >= 4:
                            high_tier_pct = tier_counts.get(4, 0) / sum(tier_counts.values()) * 100
                            if high_tier_pct > 20:
                                status_class = "alert"
                                status_msg = f"{high_tier_pct:.1f}% of usage falls in the highest rate tier (Tier 4), increasing costs significantly."
                            elif high_tier_pct > 10:
                                status_class = "warning"
                                status_msg = f"{high_tier_pct:.1f}% of usage falls in the highest rate tier (Tier 4)."
                            else:
                                status_class = "good"
                                status_msg = "Most usage falls within optimal rate tiers."
                        else:
                            status_class = "good"
                            status_msg = "All usage falls within optimal rate tiers."
                    else:
                        status_class = "warning"
                        status_msg = "Insufficient tier data available."
                    
                    st.markdown(f"""
                        <div class="status-card {status_class}">
                            <h4>Tier Distribution</h4>
                            <p>{status_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""---""")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Monthly Water Costs")
                    fig = px.bar(
                        monthly_data.tail(12), 
                        x="month", 
                        y=["totalCost", "totalUsage"],
                        barmode="group",
                        labels={
                            "month": "Month",
                            "value": "Amount",
                            "variable": "Metric"
                        },
                        title="Monthly Cost & Usage (Last 12 Months)",
                        color_discrete_sequence=[MRK_COLORS["navy"], BLUE_PALETTE[4]]
                    )
                    
                    fig.update_layout(
                        plot_bgcolor=MRK_COLORS["white"],
                        paper_bgcolor=MRK_COLORS["white"],
                        font=dict(color=MRK_COLORS["charcoal"]),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Cost vs. Usage Trends")
                    trend_vars = ["averageRate"]
                    if "costYoY" in monthly_data.columns:
                        trend_vars.append("costYoY")
                    if "usageYoY" in monthly_data.columns:
                        trend_vars.append("usageYoY")
                    
                    fig = px.line(
                        monthly_data.tail(12),
                        x="month",
                        y=trend_vars,
                        labels={
                            "month": "Month",
                            "value": "Value",
                            "variable": "Metric"
                        },
                        title="Rate, Cost & Usage Trends (Last 12 Months)",
                        color_discrete_sequence=LINE_COLORS
                    )
                    
                    fig.update_layout(
                        plot_bgcolor=MRK_COLORS["white"],
                        paper_bgcolor=MRK_COLORS["white"],
                        font=dict(color=MRK_COLORS["charcoal"]),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=400
                    )
                    
                    # Add reference line at y=0
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Key insights box
                st.markdown(f"""
                    <div class="card" style="margin-top: 20px; padding: 20px;">
                        <h3 style="color: {MRK_COLORS['navy']}; margin-top: 0;">Key Insights</h3>
                        <ul>
                            <li>The recommended utility allowance for water is <strong>{format_currency(utility_allowance["recommended"])}</strong>, which accounts for seasonal variations.</li>
                            <li>Average monthly water cost is <strong>{format_currency(utility_allowance["average"])}</strong> based on historical data.</li>
                            <li>Water costs show a {'<span style="color:' + MRK_COLORS['orange'] + '">rising</span>' if cost_yoy_value > 0 else '<span style="color:green">declining</span>'} trend year-over-year by {format_percent(abs(cost_yoy_value))} compared to previous year.</li>
                            <li>The primary driver of cost changes appears to be <strong>{primary_driver}</strong>, which should inform management strategies.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                # Quick action buttons
                st.subheader("Quick Actions")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📊 Generate Full Report"):
                        st.info("Full report generation is being implemented. Coming soon!")
                
                with col2:
                    if st.button("📩 Share Insights"):
                        st.info("Email sharing functionality coming soon!")
                        
                with col3:
                    if st.button("💾 Export Data"):
                        st.info("Data export functionality coming soon!")
            
            # Tab 2: Monthly Analysis
            with tab2:
                st.header("Monthly Water Usage Analysis")
                
                # Year selector
                if 'year' in monthly_data.columns:
                    years = sorted(monthly_data['year'].unique())
                    if years:
                        year_select = st.selectbox(
                            "Select Year:", 
                            options=years, 
                            index=len(years)-1,  # Default to most recent year
                            key="monthly_year_select"
                        )
                        filtered_data = monthly_data[monthly_data['year'] == year_select]
                    else:
                        filtered_data = monthly_data
                else:
                    filtered_data = monthly_data
                
                st.markdown("""---""")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Monthly Water Usage")
                    fig = px.bar(
                        filtered_data,
                        x="month",
                        y="totalUsage",
                        labels={
                            "month": "Month",
                            "totalUsage": "Usage (units)"
                        },
                        title=f"Monthly Water Usage ({year_select if 'year_select' in locals() else 'All Years'})",
                        color_discrete_sequence=[BLUE_PALETTE[2]]
                    )
                    
                    fig.update_layout(
                        plot_bgcolor=MRK_COLORS["white"],
                        paper_bgcolor=MRK_COLORS["white"],
                        font=dict(color=MRK_COLORS["charcoal"]),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Monthly Water Costs")
                    fig = px.bar(
                        filtered_data,
                        x="month",
                        y="totalCost",
                        labels={
                            "month": "Month",
                            "totalCost": "Total Cost ($)"
                        },
                        title=f"Monthly Water Costs ({year_select if 'year_select' in locals() else 'All Years'})",
                        color_discrete_sequence=[MRK_COLORS["navy"]]
                    )
                    
                    fig.update_layout(
                        plot_bgcolor=MRK_COLORS["white"],
                        paper_bgcolor=MRK_COLORS["white"],
                        font=dict(color=MRK_COLORS["charcoal"]),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Year over year comparison if available
                if "usageYoY" in filtered_data.columns or "costYoY" in filtered_data.columns:
                    yoy_cols = []
                    if "usageYoY" in filtered_data.columns:
                        yoy_cols.append("usageYoY")
                    if "costYoY" in filtered_data.columns:
                        yoy_cols.append("costYoY")
                    if "rateYoY" in filtered_data.columns:
                        yoy_cols.append("rateYoY")
                    
                    if yoy_cols:
                        st.subheader("Year-over-Year Monthly Comparisons")
                        
                        fig = px.line(
                            filtered_data.dropna(subset=yoy_cols),
                            x="month",
                            y=yoy_cols,
                            labels={
                                "month": "Month",
                                "value": "YoY Change (%)",
                                "variable": "Metric"
                            },
                            title=f"Monthly YoY Changes ({year_select if 'year_select' in locals() else 'All Years'})",
                            color_discrete_sequence=[BLUE_PALETTE[3], MRK_COLORS["orange"], BLUE_PALETTE[5]]
                        )
                        
                        fig.update_layout(
                            plot_bgcolor=MRK_COLORS["white"],
                            paper_bgcolor=MRK_COLORS["white"],
                            font=dict(color=MRK_COLORS["charcoal"]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=400
                        )
                        
                        # Add reference line at y=0
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.subheader("Monthly Data")
                
                # Format the dataframe for display
                display_df = filtered_data.copy()
                if "usageYoY" in display_df.columns:
                    display_df["usageYoY"] = display_df["usageYoY"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                if "costYoY" in display_df.columns:
                    display_df["costYoY"] = display_df["costYoY"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                if "rateYoY" in display_df.columns:
                    display_df["rateYoY"] = display_df["rateYoY"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                display_df["totalCost"] = display_df["totalCost"].apply(format_currency)
                display_df["averageRate"] = display_df["averageRate"].apply(lambda x: f"${x:.2f}")
                display_df["totalUsage"] = display_df["totalUsage"].apply(lambda x: f"{int(x)}")
                
                cols_to_display = ["month", "totalUsage"]
                if "usageYoY" in display_df.columns:
                    cols_to_display.append("usageYoY")
                cols_to_display.extend(["totalCost"])
                if "costYoY" in display_df.columns:
                    cols_to_display.append("costYoY")
                cols_to_display.append("averageRate")
                if "rateYoY" in display_df.columns:
                    cols_to_display.append("rateYoY")
                
                st.dataframe(
                    display_df[cols_to_display],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button for this data
                csv_data = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download Monthly Data as CSV",
                    data=csv_data,
                    file_name=f"monthly_water_data_{year_select if 'year_select' in locals() else 'all_years'}.csv",
                    mime="text/csv"
                )
            
            # Tab 3: Year-over-Year Analysis
            with tab3:
                st.header("Year-over-Year Analysis")
                st.write("Comparing annual trends and identifying long-term patterns")
                
                st.markdown("""---""")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Annual Water Usage & Cost")
                    fig = px.bar(
                        yearly_data,
                        x="year",
                        y=["totalUsage", "totalCost"],
                        barmode="group",
                        labels={
                            "year": "Year",
                            "value": "Amount",
                            "variable": "Metric"
                        },
                        title="Annual Usage & Cost Comparison",
                        color_discrete_sequence=[BLUE_PALETTE[3], MRK_COLORS["navy"]]
                    )
                    
                    fig.update_layout(
                        plot_bgcolor=MRK_COLORS["white"],
                        paper_bgcolor=MRK_COLORS["white"],
                        font=dict(color=MRK_COLORS["charcoal"]),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Annual Year-over-Year Changes")
                    yoy_cols = []
                    if "usageYoY" in yearly_data.columns:
                        yoy_cols.append("usageYoY")
                    if "costYoY" in yearly_data.columns:
                        yoy_cols.append("costYoY")
                    if "rateYoY" in yearly_data.columns:
                        yoy_cols.append("rateYoY")
                    
                    if yoy_cols:
                        yoy_data = yearly_data.dropna(subset=yoy_cols)
                        fig = px.line(
                            yoy_data,
                            x="year",
                            y=yoy_cols,
                            labels={
                                "year": "Year",
                                "value": "YoY Change (%)",
                                "variable": "Metric"
                            },
                            title="Year-over-Year (YoY) Changes",
                            color_discrete_sequence=LINE_COLORS
                        )
                        
                        fig.update_layout(
                            plot_bgcolor=MRK_COLORS["white"],
                            paper_bgcolor=MRK_COLORS["white"],
                            font=dict(color=MRK_COLORS["charcoal"]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=400
                        )
                        
                        # Add reference line at y=0
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Year-over-year data not available. Need at least two years of data.")
                
                # Average Rate Trend
                if len(yearly_data) > 1:
                    st.subheader("Water Rate Trends")
                    
                    fig = px.line(
                        yearly_data,
                        x="year",
                        y="averageRate",
                        labels={
                            "year": "Year",
                            "averageRate": "Average Rate ($/unit)"
                        },
                        title="Average Water Rate Trend by Year",
                        markers=True,
                        color_discrete_sequence=[MRK_COLORS["orange"]]
                    )
                    
                    fig.update_layout(
                        plot_bgcolor=MRK_COLORS["white"],
                        paper_bgcolor=MRK_COLORS["white"],
                        font=dict(color=MRK_COLORS["charcoal"]),
                        height=400
                    )
                    
                    # Add data labels to the points
                    fig.update_traces(texttemplate='$%{y:.2f}', textposition='top center')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.subheader("Yearly Summary Data")
                
                # Format the dataframe for display
                display_df = yearly_data.copy()
                if "usageYoY" in display_df.columns:
                    display_df["usageYoY"] = display_df["usageYoY"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                if "costYoY" in display_df.columns:
                    display_df["costYoY"] = display_df["costYoY"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                if "rateYoY" in display_df.columns:
                    display_df["rateYoY"] = display_df["rateYoY"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                display_df["totalCost"] = display_df["totalCost"].apply(format_currency)
                display_df["averageRate"] = display_df["averageRate"].apply(lambda x: f"${x:.2f}")
                display_df["totalUsage"] = display_df["totalUsage"].apply(lambda x: f"{int(x)}")
                
                cols_to_display = ["year", "totalUsage"]
                if "usageYoY" in display_df.columns:
                    cols_to_display.append("usageYoY")
                cols_to_display.append("totalCost")
                if "costYoY" in display_df.columns:
                    cols_to_display.append("costYoY")
                cols_to_display.append("averageRate")
                if "rateYoY" in display_df.columns:
                    cols_to_display.append("rateYoY")
                
                st.dataframe(
                    display_df[cols_to_display],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Analysis summary
                if len(yearly_data) > 1:
                    avg_usage_change = yearly_data["usageYoY"].mean() if "usageYoY" in yearly_data.columns else 0
                    avg_cost_change = yearly_data["costYoY"].mean() if "costYoY" in yearly_data.columns else 0
                    avg_rate_change = yearly_data["rateYoY"].mean() if "rateYoY" in yearly_data.columns else 0
                    
                    st.markdown(f"""
                        <div class="card" style="margin-top: 20px; padding: 20px;">
                            <h3 style="color: {MRK_COLORS['navy']}; margin-top: 0;">Multi-Year Trend Analysis</h3>
                            <p>Based on the available historical data:</p>
                            <ul>
                                <li>Water <strong>usage</strong> has shown an average annual change of <span style="color: {'red' if avg_usage_change > 0 else 'green'}">{format_percent(avg_usage_change)}</span>.</li>
                                <li>Water <strong>costs</strong> have shown an average annual change of <span style="color: {'red' if avg_cost_change > 0 else 'green'}">{format_percent(avg_cost_change)}</span>.</li>
                                <li>Water <strong>rates</strong> have shown an average annual change of <span style="color: {'red' if avg_rate_change > 0 else 'green'}">{format_percent(avg_rate_change)}</span>.</li>
                                <li>The data indicates a {'<strong>growing</strong>' if avg_cost_change > 0 else '<strong>declining</strong>'} trend in water expenses over time.</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Tab 4: Cost Driver Analysis
            with tab4:
                st.header("Cost Driver Analysis")
                
                # Year selector
                if 'year' in cost_driver_data.columns:
                    years = sorted(cost_driver_data['year'].unique())
                    if years:
                        year_select = st.selectbox(
                            "Select Year:", 
                            options=years, 
                            index=len(years)-1,
                            key="cost_driver_year_select"
                        )
                        filtered_data = cost_driver_data[cost_driver_data['year'] == year_select]
                    else:
                        filtered_data = cost_driver_data
                else:
                    filtered_data = cost_driver_data
                
                st.markdown("""---""")
                
                # Check if we have sufficient data for cost driver analysis
                if len(filtered_data) < 2 or 'primaryDriver' not in filtered_data.columns:
                    st.info("Insufficient data for cost driver analysis. Need at least two periods with year-over-year data.")
                else:
                    # Cost driver summary
                    st.subheader("Cost Driver Summary")
                    
                    # Calculate summary metrics
                    usage_driven = filtered_data[filtered_data['primaryDriver'] == 'Usage'].shape[0]
                    rate_driven = filtered_data[filtered_data['primaryDriver'] == 'Rate'].shape[0]
                    
                    avg_usage_contribution = filtered_data['usageContribution'].mean()
                    avg_rate_contribution = filtered_data['rateContribution'].mean()
                    
                    st.markdown(f"""
                        <div class="card" style="margin-bottom: 20px; padding: 20px;">
                            <h3 style="color: {MRK_COLORS['navy']}; margin-top: 0;">Understanding Cost Drivers</h3>
                            <p>This analysis examines whether cost increases are primarily driven by <strong>usage increases</strong> (which you can control) 
                            or <strong>rate increases</strong> (which you cannot control).</p>
                            
                            <p>For the selected period, <strong>{avg_usage_contribution:.1f}%</strong> of cost changes were driven by usage, 
                            while <strong>{avg_rate_contribution:.1f}%</strong> were driven by rate changes.</p>
                            
                            <p>Cost increases were primarily rate-driven in <strong>{rate_driven} months</strong> and 
                            usage-driven in <strong>{usage_driven} months</strong>.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Primary Cost Drivers")
                        if 'primaryDriver' in filtered_data.columns:
                            driver_counts = filtered_data["primaryDriver"].value_counts().reset_index()
                            driver_counts.columns = ["Driver", "Count"]
                            
                            fig = px.pie(
                                driver_counts,
                                values="Count",
                                names="Driver",
                                title="Primary Cost Drivers Distribution",
                                color_discrete_sequence=[BLUE_PALETTE[2], MRK_COLORS["orange"]]
                            )
                            
                            fig.update_layout(
                                plot_bgcolor=MRK_COLORS["white"],
                                paper_bgcolor=MRK_COLORS["white"],
                                font=dict(color=MRK_COLORS["charcoal"]),
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Primary driver data not available.")
                    
                    with col2:
                        st.subheader("Cost vs. Usage YoY Comparison")
                        if 'costYoY' in filtered_data.columns and 'usageYoY' in filtered_data.columns:
                            fig = px.line(
                                filtered_data,
                                x="month",
                                y=["costYoY", "usageYoY"],
                                labels={
                                    "month": "Month",
                                    "value": "YoY Change (%)",
                                    "variable": "Metric"
                                },
                                title="Cost vs. Usage YoY Comparison",
                                color_discrete_sequence=[MRK_COLORS["navy"], BLUE_PALETTE[4]]
                            )
                            
                            fig.update_layout(
                                plot_bgcolor=MRK_COLORS["white"],
                                paper_bgcolor=MRK_COLORS["white"],
                                font=dict(color=MRK_COLORS["charcoal"]),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                height=400
                            )
                            
                            # Add reference line at y=0
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Year-over-year comparison data not available.")
                    
                    # Additional chart for contribution analysis
                    if 'usageContribution' in filtered_data.columns and 'rateContribution' in filtered_data.columns:
                        st.subheader("Monthly Cost Change Attribution")
                        
                        fig = px.bar(
                            filtered_data,
                            x="month",
                            y=["usageContribution", "rateContribution"],
                            barmode="stack",
                            labels={
                                "month": "Month",
                                "value": "Contribution (%)",
                                "variable": "Driver"
                            },
                            title="Monthly Cost Change Attribution",
                            color_discrete_sequence=[BLUE_PALETTE[3], MRK_COLORS["orange"]]
                        )
                        
                        fig.update_layout(
                            plot_bgcolor=MRK_COLORS["white"],
                            paper_bgcolor=MRK_COLORS["white"],
                            font=dict(color=MRK_COLORS["charcoal"]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.subheader("Cost Driver Details")
                    
                    # Format the dataframe for display
                    display_df = filtered_data.copy()
                    if "costYoY" in display_df.columns:
                        display_df["costYoY"] = display_df["costYoY"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                    if "usageYoY" in display_df.columns:
                        display_df["usageYoY"] = display_df["usageYoY"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                    if "usageContribution" in display_df.columns:
                        display_df["usageContribution"] = display_df["usageContribution"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                    if "rateContribution" in display_df.columns:
                        display_df["rateContribution"] = display_df["rateContribution"].apply(lambda x: format_percent(x) if not pd.isna(x) else "N/A")
                    
                    cols_to_display = ["month"]
                    if "costYoY" in display_df.columns:
                        cols_to_display.append("costYoY")
                    if "usageYoY" in display_df.columns:
                        cols_to_display.append("usageYoY")
                    if "primaryDriver" in display_df.columns:
                        cols_to_display.append("primaryDriver")
                    if "usageContribution" in display_df.columns:
                        cols_to_display.append("usageContribution")
                    if "rateContribution" in display_df.columns:
                        cols_to_display.append("rateContribution")
                    
                    st.dataframe(
                        display_df[cols_to_display],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Management recommendations based on cost drivers
                    driver_recommendations = {
                        "Usage": [
                            "Implement water conservation measures",
                            "Install low-flow fixtures and appliances",
                            "Conduct regular maintenance to check for leaks",
                            "Educate residents on water-saving practices",
                            "Consider smart irrigation systems for landscaping"
                        ],
                        "Rate": [
                            "Monitor utility rate changes and budget accordingly",
                            "Evaluate potential for utility rate negotiation",
                            "Compare rates across similar properties to ensure competitiveness",
                            "Consider water reclamation or recycling systems for long-term savings",
                            "Calculate ROI on water-saving capital improvements to offset rising rates"
                        ]
                    }
                    
                    primary_driver = "Usage" if usage_driven > rate_driven else "Rate"
                    
                    st.markdown(f"""
                        <div class="card" style="margin-top: 20px; padding: 20px;">
                            <h3 style="color: {MRK_COLORS['navy']}; margin-top: 0;">Management Recommendations</h3>
                            <p>Based on the analysis, the primary cost driver is <strong>{primary_driver}</strong>. Here are recommended actions:</p>
                            <ul>
                                {"".join([f"<li>{item}</li>" for item in driver_recommendations[primary_driver]])}
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Tab 5: Tier Analysis
            with tab5:
                st.header("Water Rate Tier Analysis")
                st.write("Analysis of tiered water pricing structure")
                
                st.markdown("""---""")
                
                # Tier structure overview
                st.subheader("Tier Structure Overview")
                
                # Format the tier structure dataframe for display
                tier_display = tier_structure.copy()
                tier_display["Daily Usage (gallons)"] = tier_display.apply(
                    lambda row: f"{row['minGpd']} - {row['maxGpd']}" if not pd.isna(row['maxGpd']) and row['maxGpd'] != float('inf') else f"{row['minGpd']}+", 
                    axis=1
                )
                tier_display["Water Rate ($/kgal)"] = tier_display["waterRate"].apply(lambda x: f"${x:.2f}")
                tier_display["Sewer Rate ($/kgal)"] = tier_display["sewerRate"].apply(lambda x: f"${x:.2f}")
                tier_display["Tier"] = tier_display["tier"].apply(lambda x: f"Tier {int(x)}")
                
                # Create a styled table with MRK colors
                st.markdown(f"""
                    <table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
                        <thead>
                            <tr style="background-color: {MRK_COLORS['navy']}; color: white;">
                                <th style="padding: 10px; text-align: left;">Tier</th>
                                <th style="padding: 10px; text-align: left;">Daily Usage (gallons)</th>
                                <th style="padding: 10px; text-align: left;">Water Rate ($/kgal)</th>
                                <th style="padding: 10px; text-align: left;">Sewer Rate ($/kgal)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([f'<tr style="background-color: {MRK_COLORS["white"] if i % 2 == 0 else MRK_COLORS["gray"]}; border-bottom: 1px solid #ddd;"><td style="padding: 10px;">{row["Tier"]}</td><td style="padding: 10px;">{row["Daily Usage (gallons)"]}</td><td style="padding: 10px;">{row["Water Rate ($/kgal)"]}</td><td style="padding: 10px;">{row["Sewer Rate ($/kgal)"]}</td></tr>' for i, (_, row) in enumerate(tier_display.iterrows())])}
                        </tbody>
                    </table>
                """, unsafe_allow_html=True)
                
                st.markdown("""---""")
                
                # Year selector for tier data
                if 'year' in tier_data.columns:
                    years = sorted(tier_data['year'].unique())
                    if years:
                        year_select = st.selectbox(
                            "Select Year:", 
                            options=years, 
                            index=len(years)-1,
                            key="tier_year_select"
                        )
                        filtered_tier_data = tier_data[tier_data['year'] == year_select]
                    else:
                        filtered_tier_data = tier_data
                else:
                    filtered_tier_data = tier_data
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Tier Distribution")
                    if 'tier' in filtered_tier_data.columns:
                        tier_counts = filtered_tier_data["tier"].value_counts().reset_index()
                        tier_counts.columns = ["Tier", "Count"]
                        tier_counts["Tier"] = tier_counts["Tier"].apply(lambda x: f"Tier {int(x)}")
                        
                        # Sort by tier number
                        tier_counts["TierNum"] = tier_counts["Tier"].str.extract(r'(\d+)').astype(int)
                        tier_counts = tier_counts.sort_values("TierNum").drop("TierNum", axis=1)
                        
                        fig = px.pie(
                            tier_counts,
                            values="Count",
                            names="Tier",
                            title="Tier Distribution",
                            color_discrete_sequence=BLUE_PALETTE
                        )
                        
                        fig.update_layout(
                            plot_bgcolor=MRK_COLORS["white"],
                            paper_bgcolor=MRK_COLORS["white"],
                            font=dict(color=MRK_COLORS["charcoal"]),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Tier distribution data not available.")
                
                with col2:
                    st.subheader("Daily Usage vs. Tier Threshold")
                    if 'dailyUsage' in filtered_tier_data.columns:
                        fig = px.bar(
                            filtered_tier_data,
                            x="month",
                            y="dailyUsage",
                            labels={
                                "month": "Month",
                                "dailyUsage": "Gallons per Day"
                            },
                            title="Daily Usage",
                            color_discrete_sequence=[MRK_COLORS["navy"]]
                        )
                        
                        # Add tier threshold lines
                        for idx, tier in tier_structure.iterrows():
                            if not pd.isna(tier["maxGpd"]) and tier["maxGpd"] != float('inf'):
                                fig.add_hline(
                                    y=tier["maxGpd"],
                                    line_dash="dash",
                                    line_color=MRK_COLORS["orange"],
                                    annotation_text=f"Tier {int(tier['tier'])} Max",
                                    annotation_font=dict(color=MRK_COLORS["charcoal"])
                                )
                        
                        fig.update_layout(
                            plot_bgcolor=MRK_COLORS["white"],
                            paper_bgcolor=MRK_COLORS["white"],
                            font=dict(color=MRK_COLORS["charcoal"]),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Daily usage data not available.")
                
                # Additional charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Tier-Based vs. Actual Charges")
                    if 'tierBasedCharge' in filtered_tier_data.columns and 'actualCharge' in filtered_tier_data.columns:
                        fig = px.bar(
                            filtered_tier_data,
                            x="month",
                            y=["tierBasedCharge", "actualCharge"],
                            barmode="group",
                            labels={
                                "month": "Month",
                                "value": "Cost ($)",
                                "variable": "Charge Type"
                            },
                            title="Tier-Based vs. Actual Charges",
                            color_discrete_sequence=[BLUE_PALETTE[3], MRK_COLORS["orange"]]
                        )
                        
                        fig.update_layout(
                            plot_bgcolor=MRK_COLORS["white"],
                            paper_bgcolor=MRK_COLORS["white"],
                            font=dict(color=MRK_COLORS["charcoal"]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Tier-based charge comparison data not available.")
                
                with col2:
                    st.subheader("Usage vs. Non-Usage Expenses")
                    if 'usageExpense' in filtered_tier_data.columns and 'nonUsageExpense' in filtered_tier_data.columns:
                        fig = px.bar(
                            filtered_tier_data,
                            x="month",
                            y=["usageExpense", "nonUsageExpense"],
                            barmode="stack",
                            labels={
                                "month": "Month",
                                "value": "Cost ($)",
                                "variable": "Expense Type"
                            },
                            title="Usage vs. Non-Usage Expenses",
                            color_discrete_sequence=[BLUE_PALETTE[2], MRK_COLORS["orange"]]
                        )
                        
                        fig.update_layout(
                            plot_bgcolor=MRK_COLORS["white"],
                            paper_bgcolor=MRK_COLORS["white"],
                            font=dict(color=MRK_COLORS["charcoal"]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Usage vs. non-usage expense breakdown not available.")
                
                # Data table
                st.subheader("Tier Analysis Details")
                
                # Format the dataframe for display
                display_df = filtered_tier_data.copy()
                
                # Format columns that might exist
                if "dailyUsage" in display_df.columns:
                    display_df["dailyUsage"] = display_df["dailyUsage"].apply(lambda x: f"{x:.1f} gal/day" if not pd.isna(x) else "N/A")
                if "tier" in display_df.columns:
                    display_df["tier"] = display_df["tier"].apply(lambda x: f"Tier {int(x)}" if not pd.isna(x) else "N/A")
                if "tierRate" in display_df.columns:
                    display_df["tierRate"] = display_df["tierRate"].apply(lambda x: f"${x:.2f}/kgal" if not pd.isna(x) else "N/A")
                if "tierBasedCharge" in display_df.columns:
                    display_df["tierBasedCharge"] = display_df["tierBasedCharge"].apply(lambda x: format_currency(x) if not pd.isna(x) else "N/A")
                if "actualCharge" in display_df.columns:
                    display_df["actualCharge"] = display_df["actualCharge"].apply(lambda x: format_currency(x) if not pd.isna(x) else "N/A")
                if "rateDifference" in display_df.columns:
                    display_df["rateDifference"] = display_df["rateDifference"].apply(lambda x: format_currency(x) if not pd.isna(x) else "N/A")
                
                # Determine which columns to display
                cols_to_display = ["month"]
                if "dailyUsage" in display_df.columns:
                    cols_to_display.append("dailyUsage")
                if "tier" in display_df.columns:
                    cols_to_display.append("tier")
                if "tierRate" in display_df.columns:
                    cols_to_display.append("tierRate")
                if "tierBasedCharge" in display_df.columns:
                    cols_to_display.append("tierBasedCharge")
                if "actualCharge" in display_df.columns:
                    cols_to_display.append("actualCharge")
                if "rateDifference" in display_df.columns:
                    cols_to_display.append("rateDifference")
                
                st.dataframe(
                    display_df[cols_to_display],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Tier optimization tips
                if 'tier' in filtered_tier_data.columns:
                    # Get counts by tier
                    tier_counts = filtered_tier_data["tier"].value_counts().to_dict()
                    most_common_tier = max(tier_counts, key=tier_counts.get) if tier_counts else None
                    
                    if most_common_tier:
                        # Check if there are higher tiers
                        higher_tier_exists = any(t > most_common_tier for t in tier_counts.keys())
                        
                        st.markdown(f"""
                            <div class="card" style="margin-top: 20px; padding: 20px;">
                                <h3 style="color: {MRK_COLORS['navy']}; margin-top: 0;">Tier Optimization Analysis</h3>
                                <p>The most common tier for this property is <strong>Tier {int(most_common_tier)}</strong>, which accounts for 
                                approximately <strong>{tier_counts[most_common_tier]/sum(tier_counts.values())*100:.1f}%</strong> of all usage periods.</p>
                                
                                {'<p>Since usage occasionally enters higher tiers, there is potential for cost savings by managing daily usage.</p>' if higher_tier_exists else ''}
                                
                                <h4 style="color: {MRK_COLORS['navy']};">Recommendations:</h4>
                                <ul>
                                    <li>Monitor daily usage to stay within lower tier thresholds</li>
                                    <li>Consider automated leak detection systems to prevent excess usage</li>
                                    <li>Implement water conservation efforts during high-usage months</li>
                                    <li>Regularly check for billing errors that might incorrectly assign higher tiers</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Tab 6: Utility Allowance
            with tab6:
                st.header("Utility Allowance Analysis")
                st.write("Assessment of appropriate utility allowances based on historical data")
                
                st.markdown("""---""")
                
                # Recommended allowance
                st.markdown(f"""
                    <div style="background-color:{MRK_COLORS['navy']}; padding:20px; border-radius:5px; text-align:center; color:white; margin-bottom:20px;">
                        <h3 style="color:white; margin-top:0;">Recommended Monthly Allowance</h3>
                        <div style="font-size:2.5rem; margin:15px 0; font-weight:bold;">{format_currency(utility_allowance["recommended"])}</div>
                        <p style="margin-bottom:0;">Based on historical usage + buffer for seasonal variations</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Allowance details
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Average Monthly Cost", 
                        value=format_currency(utility_allowance["average"]),
                        help="Based on last 12 months"
                    )
                
                with col2:
                    st.metric(
                        label="Minimum Monthly Cost", 
                        value=format_currency(utility_allowance["min"]),
                        help="Lowest in last 12 months"
                    )
                
                with col3:
                    st.metric(
                        label="Maximum Monthly Cost", 
                        value=format_currency(utility_allowance["max"]),
                        help="Highest in last 12 months"
                    )
                
                with col4:
                    st.metric(
                        label="Standard Deviation", 
                        value=format_currency(utility_allowance["stdDev"]),
                        help="Typical variation from average"
                    )
                
                st.markdown("""---""")
                
                # Cost distribution chart
                if len(monthly_data) > 0:
                    st.subheader("Distribution of Monthly Costs")
                    
                    # Create histogram of costs
                    fig = px.histogram(
                        monthly_data,
                        x="totalCost",
                        nbins=10,
                        labels={"totalCost": "Monthly Cost ($)"},
                        title="Frequency Distribution of Monthly Water Costs",
                        color_discrete_sequence=[BLUE_PALETTE[2]]
                    )
                    
                    # Add vertical lines for key metrics
                    fig.add_vline(x=utility_allowance["average"], line_dash="solid", line_color=BLUE_PALETTE[4],
                                 annotation_text="Average", annotation_position="top right")
                    fig.add_vline(x=utility_allowance["recommended"], line_dash="solid", line_color=MRK_COLORS["orange"],
                                 annotation_text="Recommended", annotation_position="top right")
                    
                    fig.update_layout(
                        plot_bgcolor=MRK_COLORS["white"],
                        paper_bgcolor=MRK_COLORS["white"],
                        font=dict(color=MRK_COLORS["charcoal"]),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Guidance
                st.subheader("Utility Allowance Guidance")
                
                st.markdown(f"""
                    <div class="card" style="margin-bottom: 20px; padding: 20px;">
                        <h3 style="color: {MRK_COLORS['navy']}; margin-top: 0;">Understanding This Recommendation</h3>
                        <p>The recommended allowance is calculated based on average monthly costs plus a buffer (standard deviation) 
                        to account for normal fluctuations in usage and rates.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="background-color: {BLUE_PALETTE[5]}; color: white; padding: 20px; border-radius: 5px; margin-top: 15px;">
                        <h4 style="color: white; margin-top: 0;">Considerations for Setting Allowances:</h4>
                        <ul style="margin-bottom: 0;">
                            <li>The current historical average is {format_currency(utility_allowance["average"])} per month</li>
                            <li>Adding a standard deviation ({format_currency(utility_allowance["stdDev"])}) accounts for ~68% of normal variations</li>
                            <li>Recommended allowance of {format_currency(utility_allowance["recommended"])} should cover most monthly bills</li>
                            <li>Consider seasonal patterns - costs typically increase in summer months</li>
                            <li>Review annually to account for utility rate increases</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                # Seasonal trend analysis
                if len(monthly_data) >= 12:
                    st.subheader("Seasonal Variations Analysis")
                    
                    # Attempt to identify seasonal patterns
                    try:
                        # Convert month names to month numbers if needed
                        month_to_num = {
                            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12,
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        
                        season_data = monthly_data.copy()
                        if isinstance(season_data["month"].iloc[0], str):
                            # Try to convert string months to numbers
                            season_data["month_num"] = season_data["month"].map(lambda x: month_to_num.get(x, 0))
                        else:
                            # Already numeric
                            season_data["month_num"] = season_data["month"]
                        
                        # Group by month number
                        seasonal_avg = season_data.groupby("month_num")["totalCost"].mean().reset_index()
                        seasonal_avg["month_name"] = seasonal_avg["month_num"].apply(lambda x: 
                            {v: k for k, v in month_to_num.items() if len(k) > 3}.get(x, f"Month {x}")
                        )
                        
                        # Sort by month
                        seasonal_avg = seasonal_avg.sort_values("month_num")
                        
                        # Calculate percent difference from annual average
                        annual_avg = seasonal_avg["totalCost"].mean()
                        seasonal_avg["pct_diff"] = (seasonal_avg["totalCost"] - annual_avg) / annual_avg * 100
                        
                        # Create chart
                        fig = px.line(
                            seasonal_avg,
                            x="month_name",
                            y="totalCost",
                            labels={
                                "month_name": "Month",
                                "totalCost": "Average Cost ($)"
                            },
                            title="Seasonal Cost Patterns",
                            markers=True,
                            color_discrete_sequence=[MRK_COLORS["navy"]]
                        )
                        
                        # Add reference line for annual average
                        fig.add_hline(
                            y=annual_avg,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Annual Average",
                            annotation_position="bottom right"
                        )
                        
                        fig.update_layout(
                            plot_bgcolor=MRK_COLORS["white"],
                            paper_bgcolor=MRK_COLORS["white"],
                            font=dict(color=MRK_COLORS["charcoal"]),
                            xaxis=dict(tickangle=45),
                            height=400
                        )
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Identify high and low seasons
                            high_months = seasonal_avg[seasonal_avg["pct_diff"] > 10]
                            low_months = seasonal_avg[seasonal_avg["pct_diff"] < -10]
                            
                            st.markdown(f"""
                                <div class="card" style="height: 350px; overflow-y: auto; padding: 15px;">
                                    <h4 style="color: {MRK_COLORS['navy']}; margin-top: 0;">Seasonal Insights</h4>
                                    
                                    <p><strong>High-Cost Months:</strong></p>
                                    <ul>
                                        {"".join([f'<li>{row["month_name"]}: {format_currency(row["totalCost"])} ({row["pct_diff"]:.1f}% above average)</li>' for _, row in high_months.iterrows()]) if not high_months.empty else "<li>No significant high-cost months identified</li>"}
                                    </ul>
                                    
                                    <p><strong>Low-Cost Months:</strong></p>
                                    <ul>
                                        {"".join([f'<li>{row["month_name"]}: {format_currency(row["totalCost"])} ({abs(row["pct_diff"]):.1f}% below average)</li>' for _, row in low_months.iterrows()]) if not low_months.empty else "<li>No significant low-cost months identified</li>"}
                                    </ul>
                                    
                                    <p><strong>Recommendation:</strong> Adjust allowances seasonally to account for these variations, especially during {', '.join(high_months["month_name"].tolist()) if not high_months.empty else "summer months"}.</p>
                                </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.info(f"Unable to analyze seasonal patterns: {str(e)}")
                
                # Add a calculator section
                st.subheader("Utility Allowance Calculator")
                
                with st.expander("Customize Allowance Calculation"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        custom_buffer = st.slider("Buffer Factor (σ)", 0.5, 2.0, allowance_buffer, 0.1, 
                                                help="Number of standard deviations to add to the average cost")
                    
                    with col2:
                        custom_base = st.radio("Base Calculation On", 
                                             ["All Available Data", "Last 12 Months", "Last 6 Months"], 
                                             index=1)
                    
                    if st.button("Calculate Custom Allowance"):
                        # Filter data based on selection
                        if custom_base == "Last 12 Months" and len(monthly_data) > 12:
                            calculation_data = monthly_data.sort_values(['year', 'month'], ascending=[False, False]).head(12)
                        elif custom_base == "Last 6 Months" and len(monthly_data) > 6:
                            calculation_data = monthly_data.sort_values(['year', 'month'], ascending=[False, False]).head(6)
                        else:
                            calculation_data = monthly_data
                        
                        # Calculate custom allowance
                        custom_avg = calculation_data['totalCost'].mean()
                        custom_std = calculation_data['totalCost'].std()
                        custom_min = calculation_data['totalCost'].min()
                        custom_max = calculation_data['totalCost'].max()
                        custom_recommended = custom_avg + (custom_buffer * custom_std)
                        
                        # Display results
                        st.markdown(f"""
                            <div style="background-color:{BLUE_PALETTE[3]}; padding:20px; border-radius:5px; margin-top:20px;">
                                <h4 style="color:white; margin-top:0;">Custom Allowance Calculation Results</h4>
                                <p style="color:white;">Based on {calculation_data.shape[0]} months of data with a buffer of {custom_buffer}σ:</p>
                                <div style="color:white; font-size:1.5rem; font-weight:bold; margin:10px 0;">
                                    Recommended Allowance: {format_currency(custom_recommended)}
                                </div>
                                <table style="width:100%; color:white;">
                                    <tr>
                                        <td style="padding:5px;">Average Cost:</td>
                                        <td style="padding:5px;">{format_currency(custom_avg)}</td>
                                        <td style="padding:5px;">Standard Deviation:</td>
                                        <td style="padding:5px;">{format_currency(custom_std)}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding:5px;">Minimum Cost:</td>
                                        <td style="padding:5px;">{format_currency(custom_min)}</td>
                                        <td style="padding:5px;">Maximum Cost:</td>
                                        <td style="padding:5px;">{format_currency(custom_max)}</td>
                                    </tr>
                                </table>
                            </div>
                        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""---""")
        st.markdown(
            f"""
            <div style='text-align:center; background-color:{MRK_COLORS["navy"]}; color:{MRK_COLORS["white"]}; padding:20px; border-radius:5px;'>
                <p style='margin-bottom:5px;'>MRK Portfolio Water Analytics Dashboard</p>
                <p style='margin-bottom:5px;'>Version {APP_VERSION} | Last updated: {BUILD_DATE}</p>
                <p style='margin:0; font-size:0.8rem;'>Please contact Matthew Dixon (mdixon@mrkpartners.com) or Josh Blankstein (jblankstein@mrkpartners.com) for troubleshooting or support</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error("Please check that your CSV file has the necessary columns and data format.")
        
        st.markdown(f"""
            <div style="background-color: {MRK_COLORS['orange']}; color: white; padding: 15px; border-radius: 5px; margin-top: 15px;">
                <h4 style="color: white; margin-top: 0;">Troubleshooting Tips</h4>
                <p>Your CSV should include, at minimum, columns for:</p>
                <ul style="margin-bottom: 0;">
                    <li>Month (or month name)</li>
                    <li>Year</li>
                    <li>Total usage</li>
                    <li>Total cost</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Contact information
        st.markdown(f"""
            <div class="contact-box">
                <h4>Need Help?</h4>
                <p>Please contact the Portfolio Analysts for troubleshooting or support:</p>
                <p><strong>Matthew Dixon:</strong> mdixon@mrkpartners.com</p>
                <p><strong>Josh Blankstein:</strong> jblankstein@mrkpartners.com</p>
            </div>
        """, unsafe_allow_html=True)
        
else:    
    # Contact information prominently displayed
    st.markdown(f"""
        <div class="contact-box" style="background-color: {MRK_COLORS['navy']}; margin-bottom: 30px;">
            <h4 style="color: white !important;">Support Contacts</h4>
            <p style="color: white;">Please reach out to the Portfolio Analysts for troubleshooting or assistance:</p>
            <p style="color: white;"><strong style="color: orange;">Matthew Dixon</strong> (mdixon@mrkpartners.com)</p>
            <p style="color: white;"><strong style="color: orange;">Josh Blankstein</strong> (jblankstein@mrkpartners.com)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show instructions for uploading data
    st.write("""
    To get started, please upload your CSV file containing water utility data using the file uploader in the sidebar.
    
    Your CSV should include the following columns:
    - Month (or month name)
    - Year
    - Total usage
    - Total cost
    
    Additional columns that enhance the analysis:
    - Days in period
    - Daily usage
    - Tier information
    - Non-usage expenses
    
    After uploading, you can select which columns represent each required data field.
    """)
    
    # Example CSV structure
    st.markdown(f"""
        <div style="margin-top:30px;">
            <h3 style="color:{MRK_COLORS['navy']};">Example CSV Format</h3>
        </div>
    """, unsafe_allow_html=True)
    
    example_data = pd.DataFrame([
        {"Month": "Jan", "Year": 2024, "Total_Usage": 120, "Total_Effective_Expense": 980, "Days_In_Period": 31, "Daily_Usage": 80, "Tier": 1},
        {"Month": "Feb", "Year": 2024, "Total_Usage": 115, "Total_Effective_Expense": 950, "Days_In_Period": 29, "Daily_Usage": 79, "Tier": 1},
        {"Month": "Mar", "Year": 2024, "Total_Usage": 125, "Total_Effective_Expense": 1020, "Days_In_Period": 31, "Daily_Usage": 85, "Tier": 2},
    ])
    
    st.dataframe(example_data, hide_index=True)
    
    # Sample CSV download with MRK styling
    sample_csv = example_data.to_csv(index=False)
    
    st.markdown(f"""
        <div style="margin-top:20px; text-align:center;">
            <p>Template for manual imports</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.download_button(
            label="Download Sample CSV Template",
            data=sample_csv,
            file_name="mrk_water_utility_template.csv",
            mime="text/csv"
        )
    
    # Feature highlights with MRK styling
    st.markdown(f"""
        <div style="margin-top:40px; margin-bottom:20px;">
            <h3 style="color:{MRK_COLORS['navy']}; text-align:center;">Key Dashboard Features</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="feature-box">
                <h4>Cost Driver Analysis</h4>
                <p>Understand whether changes in water costs are driven by <strong>usage</strong> (our control) or <strong>rate increases</strong> (city driven).</p>
                <p class="accent">Target actionable cost savings</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="feature-box">
                <h4>Utility Allowance Calculator</h4>
                <p>based on historical usage patterns and seasonal variations.</p>
                <p class="accent">Set more accurate resident allowances</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="feature-box">
                <h4>Tier Analysis</h4>
                <p>Visualize the tiered rate structures impact your water costs and identify opportunities to optimize usage to stay in lower-cost tiers.</p>
                <p class="accent">Optimize usage for lower rates</p>
            </div>
        """, unsafe_allow_html=True)
    
    # App version and footer
    st.markdown(f"""
        <div style='text-align:center; background-color:{MRK_COLORS["navy"]}; color:{MRK_COLORS["white"]}; padding:20px; border-radius:5px; margin-top:40px;'>
            <p style='margin-bottom:5px;'>MRK Portfolio Water Analytics Dashboard</p>
            <p style='margin-bottom:5px;'>Version {APP_VERSION} | Last updated: {BUILD_DATE}</p>
            <p style='margin:0; font-size:0.8rem;'>MRK Partners</p>
        </div>
    """, unsafe_allow_html=True)
