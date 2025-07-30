import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import base64
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Supply Chain Emission Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .reportview-container .markdown-text-container {
        font-family: 'Roboto', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .css-1d391kg {
        padding: 1rem 1rem 10rem;
    }
    h1 {
        color: #1f77b4;
        margin-bottom: 1.5rem;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #1f77b4;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        text-align: center;
        padding: 10px;
        color: #6c757d;
        font-size: 0.9em;
        border-top: 1px solid #e9ecef;
    }
    .sample-data {
        background-color: #e9f7fe;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    /* Dark background for prediction results */
    .dark-bg {
        background-color: #1a1a1a;
        color: white;
        border-radius: 5px;
        padding: 15px;
        margin: -10px -15px;
    }
    .dark-bg h2, .dark-bg h3 {
        color: white;
    }
    .dark-bg .stMarkdown {
        color: white;
    }
    .dark-bg .stMetric {
        background-color: #2d2d2d;
        color: white;
    }
    .dark-bg .stMetric label {
        color: #e0e0e0;
    }
    .dark-bg .stMetric div {
        color: white;
        font-weight: bold;
    }
    .dark-bg .stProgress {
        background-color: #333333;
    }
    .dark-bg .stProgress > div > div > div > div {
        background-color: #4da6ff;
    }
    .blue-box {
        background-color: #1f77b4;
        padding: 15px;
        border-radius: 5px;
        color: white;
    }
    .blue-box h3, .blue-box h4 {
        color: white;
    }
    .blue-box .stMarkdown {
        color: white;
    }
    .blue-box .stButton>button {
        background-color: #28598a;
        color: white;
        border-radius: 5px;
        height: 3em;
    }
    .blue-box .stButton>button:hover {
        background-color: #1f4e79;
    }
    .blue-box .stExpander {
        background-color: #28598a;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .blue-box .stExpander:hover {
        background-color: #1f4e79;
    }
    .blue-box .stExpander summary {
        color: white;
        font-weight: bold;
    }
    .blue-box .stJson {
        background-color: #28598a;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🌍 Supply Chain Emission Factors Predictor")
st.markdown("""
**Predict GHG emissions per unit of economic activity** for US industries and commodities using advanced machine learning. 
This tool helps businesses estimate their supply chain emissions based on industry characteristics and data quality metrics.
""")

# Create columns for layout
col1, col2 = st.columns([1, 1])

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing components with caching"""
    try:
        # Try to load XGBoost model (best performer)
        model = joblib.load('models/xgboost_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        return model, preprocessor, "XGBoost"
    except Exception as e:
        st.warning(f"XGBoost model loading failed: {str(e)}")
        try:
            # Try CatBoost if XGBoost not available
            model = joblib.load('models/catboost_model.pkl')
            return model, None, "CatBoost"
        except Exception as e:
            st.warning(f"CatBoost model loading failed: {str(e)}")
            try:
                # Try LightGBM if others not available
                model = joblib.load('models/lightgbm_model.pkl')
                preprocessor = joblib.load('models/preprocessor.pkl')
                return model, preprocessor, "LightGBM"
            except Exception as e:
                st.warning(f"LightGBM model loading failed: {str(e)}")
                st.error("No model found. Please ensure models are trained and saved in the 'models' directory.")
                return None, None, None

def create_sample_data():
    """Create sample data for autofill feature"""
    return {
        'Source': '2016_Detail_Industry',
        'Code': '561100',
        'Name': 'Office administration',
        'Substance': 'carbon dioxide',
        'Unit': 'kg/2018 USD, purchaser price',
        'Year': 2020,
        'DQ_Reliability': 3,
        'DQ_Temporal': 2,
        'DQ_Geographical': 1,
        'DQ_Technological': 3,
        'DQ_DataCollection': 1
    }

def create_alternative_sample():
    """Create alternative sample data for different industry"""
    return {
        'Source': '2016_Detail_Industry',
        'Code': '311520',
        'Name': 'Ice cream and frozen dessert manufacturing',
        'Substance': 'other GHGs',
        'Unit': 'kg CO2e/2018 USD, purchaser price',
        'Year': 2020,
        'DQ_Reliability': 4,
        'DQ_Temporal': 2,
        'DQ_Geographical': 1,
        'DQ_Technological': 5,
        'DQ_DataCollection': 1
    }

def create_manufacturing_sample():
    """Create sample data for manufacturing industry"""
    return {
        'Source': '2016_Detail_Industry',
        'Code': '332410',
        'Name': 'Power boilers and heat exchangers',
        'Substance': 'other GHGs',
        'Unit': 'kg CO2e/2018 USD, purchaser price',
        'Year': 2020,
        'DQ_Reliability': 3,
        'DQ_Temporal': 3,
        'DQ_Geographical': 1,
        'DQ_Technological': 2,
        'DQ_DataCollection': 1
    }

def create_commodity_sample():
    """Create sample data for commodity"""
    return {
        'Source': 'Commodity',
        'Code': '561100',
        'Name': 'Office administration',
        'Substance': 'carbon dioxide',
        'Unit': 'kg/2018 USD, purchaser price',
        'Year': 2020,
        'DQ_Reliability': 3,
        'DQ_Temporal': 2,
        'DQ_Geographical': 1,
        'DQ_Technological': 3,
        'DQ_DataCollection': 1
    }

def predict_emission_factors(input_data, model, preprocessor, model_name):
    """Make prediction using the loaded model with proper input handling"""
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    if model_name == "CatBoost":
        # CatBoost handles categorical features natively
        prediction = model.predict(input_df)
    elif model_name == "XGBoost":
        # For XGBoost native API, need to convert to DMatrix
        input_processed = preprocessor.transform(input_df)
        dmatrix = xgb.DMatrix(input_processed)
        prediction = model.predict(dmatrix)
    else:
        # For LightGBM and other scikit-learn API models
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)
    
    return prediction[0]

def get_interpretation(prediction, input_data):
    """Generate interpretation of the prediction"""
    # Simple interpretation logic based on prediction value
    if prediction < 0.05:
        interpretation = "This represents a **very low** emission intensity. Your supply chain appears highly efficient with minimal GHG impact per unit of economic activity."
    elif prediction < 0.2:
        interpretation = "This represents a **moderate** emission intensity. There may be opportunities to improve efficiency in your supply chain."
    else:
        interpretation = "This represents a **high** emission intensity. Significant opportunities likely exist to reduce emissions in this supply chain segment."
    
    # Add substance-specific interpretation
    if input_data['Substance'] == 'carbon dioxide':
        interpretation += "\n\n*Carbon dioxide is the most prevalent greenhouse gas, primarily from fossil fuel combustion.*"
    elif input_data['Substance'] == 'methane':
        interpretation += "\n\n*Methane has a much higher global warming potential than CO₂ but shorter atmospheric lifetime.*"
    elif input_data['Substance'] == 'nitrous oxide':
        interpretation += "\n\n*Nitrous oxide has a very high global warming potential and long atmospheric lifetime.*"
    else:
        interpretation += "\n\n*This represents a mixture of greenhouse gases expressed as CO₂ equivalent.*"
    
    return interpretation

def create_download_link(data, filename="emission_prediction.csv"):
    """Create a download link for the prediction results"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #4da6ff; text-decoration: none;">📥 Download Prediction Results</a>'
    return href

# Load model (cached)
model, preprocessor, model_name = load_model()

# Initialize session state for inputs if not already set
if 'source' not in st.session_state:
    st.session_state.source = "2016_Detail_Industry"
if 'code' not in st.session_state:
    st.session_state.code = ""
if 'name' not in st.session_state:
    st.session_state.name = ""
if 'substance' not in st.session_state:
    st.session_state.substance = "carbon dioxide"
if 'unit' not in st.session_state:
    st.session_state.unit = "kg/2018 USD, purchaser price"
if 'year' not in st.session_state:
    st.session_state.year = 2020
if 'dq_reliability' not in st.session_state:
    st.session_state.dq_reliability = 3
if 'dq_temporal' not in st.session_state:
    st.session_state.dq_temporal = 2
if 'dq_geographical' not in st.session_state:
    st.session_state.dq_geographical = 1
if 'dq_technological' not in st.session_state:
    st.session_state.dq_technological = 3
if 'dq_datacollection' not in st.session_state:
    st.session_state.dq_datacollection = 1

with col1:
    st.subheader("Input Parameters")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Manual Input", "Sample Data"])
    
    with tab1:
        # Create two columns for input fields
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            source = st.selectbox(
                "Data Source",
                ["2010_Detail_Industry", "2011_Detail_Industry", "2012_Detail_Industry", 
                 "2013_Detail_Industry", "2014_Detail_Industry", "2015_Detail_Industry",
                 "2016_Detail_Industry", "Commodity"],
                index=["2010_Detail_Industry", "2011_Detail_Industry", "2012_Detail_Industry", 
                       "2013_Detail_Industry", "2014_Detail_Industry", "2015_Detail_Industry",
                       "2016_Detail_Industry", "Commodity"].index(st.session_state.source),
                help="Select the data source/year for this industry/commodity",
                key="source_select"
            )
            
            code = st.text_input(
                "Industry/Commodity Code",
                value=st.session_state.code,
                help="Enter NAICS or industry code (e.g., 561100 for Office administration)",
                placeholder="e.g., 561100",
                key="code_input"
            )
            
            name = st.text_input(
                "Industry/Commodity Name",
                value=st.session_state.name,
                help="Enter the full industry or commodity name",
                placeholder="e.g., Office administration",
                key="name_input"
            )
            
            substance = st.selectbox(
                "Substance",
                ["carbon dioxide", "methane", "nitrous oxide", "other GHGs"],
                index=["carbon dioxide", "methane", "nitrous oxide", "other GHGs"].index(st.session_state.substance),
                help="Select the greenhouse gas substance",
                key="substance_select"
            )
            
            unit = st.selectbox(
                "Unit",
                ["kg/2018 USD, purchaser price", "kg CO2e/2018 USD, purchaser price"],
                index=["kg/2018 USD, purchaser price", "kg CO2e/2018 USD, purchaser price"].index(st.session_state.unit),
                help="Select the measurement unit",
                key="unit_select"
            )
            
            year = st.number_input(
                "Year",
                min_value=2010,
                max_value=2023,
                value=st.session_state.year,
                help="Year for which you're estimating emissions",
                key="year_input"
            )
        
        with col1_2:
            st.markdown("**Data Quality Metrics (1-5 scale, 5=highest quality)**")
            
            dq_reliability = st.slider(
                "Reliability",
                1, 5, st.session_state.dq_reliability,
                help="How reliable is the underlying data? (5 = highest reliability)",
                key="reliability_slider"
            )
            
            dq_temporal = st.slider(
                "Temporal Correlation",
                1, 5, st.session_state.dq_temporal,
                help="How recent is the data? (5 = most recent)",
                key="temporal_slider"
            )
            
            dq_geographical = st.slider(
                "Geographical Correlation",
                1, 5, st.session_state.dq_geographical,
                help="How well does the geography match? (5 = perfect match)",
                key="geographical_slider"
            )
            
            dq_technological = st.slider(
                "Technological Correlation",
                1, 5, st.session_state.dq_technological,
                help="How well does the technology match? (5 = perfect match)",
                key="technological_slider"
            )
            
            dq_datacollection = st.slider(
                "Data Collection",
                1, 5, st.session_state.dq_datacollection,
                help="Quality of data collection methods (5 = best methods)",
                key="datacollection_slider"
            )
    
    with tab2:
        st.info("Click below to autofill with sample data for different industries")
        
        # Create a blue box container for sample data buttons
        st.markdown('<div class="blue-box">', unsafe_allow_html=True)
        
        # Three different sample data options
        col2_1, col2_2, col2_3, col2_4 = st.columns(4)
        
        with col2_1:
            if st.button("📋 Office Administration", use_container_width=True, key="office_sample"):
                sample_data = create_sample_data()
                st.session_state.source = sample_data['Source']
                st.session_state.code = sample_data['Code']
                st.session_state.name = sample_data['Name']
                st.session_state.substance = sample_data['Substance']
                st.session_state.unit = sample_data['Unit']
                st.session_state.year = sample_data['Year']
                st.session_state.dq_reliability = sample_data['DQ_Reliability']
                st.session_state.dq_temporal = sample_data['DQ_Temporal']
                st.session_state.dq_geographical = sample_data['DQ_Geographical']
                st.session_state.dq_technological = sample_data['DQ_Technological']
                st.session_state.dq_datacollection = sample_data['DQ_DataCollection']
                st.rerun()
        
        with col2_2:
            if st.button("🍦 Food Manufacturing", use_container_width=True, key="food_sample"):
                sample_data = create_alternative_sample()
                st.session_state.source = sample_data['Source']
                st.session_state.code = sample_data['Code']
                st.session_state.name = sample_data['Name']
                st.session_state.substance = sample_data['Substance']
                st.session_state.unit = sample_data['Unit']
                st.session_state.year = sample_data['Year']
                st.session_state.dq_reliability = sample_data['DQ_Reliability']
                st.session_state.dq_temporal = sample_data['DQ_Temporal']
                st.session_state.dq_geographical = sample_data['DQ_Geographical']
                st.session_state.dq_technological = sample_data['DQ_Technological']
                st.session_state.dq_datacollection = sample_data['DQ_DataCollection']
                st.rerun()
        
        with col2_3:
            if st.button("🏭 Manufacturing", use_container_width=True, key="manufacturing_sample"):
                sample_data = create_manufacturing_sample()
                st.session_state.source = sample_data['Source']
                st.session_state.code = sample_data['Code']
                st.session_state.name = sample_data['Name']
                st.session_state.substance = sample_data['Substance']
                st.session_state.unit = sample_data['Unit']
                st.session_state.year = sample_data['Year']
                st.session_state.dq_reliability = sample_data['DQ_Reliability']
                st.session_state.dq_temporal = sample_data['DQ_Temporal']
                st.session_state.dq_geographical = sample_data['DQ_Geographical']
                st.session_state.dq_technological = sample_data['DQ_Technological']
                st.session_state.dq_datacollection = sample_data['DQ_DataCollection']
                st.rerun()
                
        with col2_4:
            if st.button("📦 Commodity", use_container_width=True, key="commodity_sample"):
                sample_data = create_commodity_sample()
                st.session_state.source = sample_data['Source']
                st.session_state.code = sample_data['Code']
                st.session_state.name = sample_data['Name']
                st.session_state.substance = sample_data['Substance']
                st.session_state.unit = sample_data['Unit']
                st.session_state.year = sample_data['Year']
                st.session_state.dq_reliability = sample_data['DQ_Reliability']
                st.session_state.dq_temporal = sample_data['DQ_Temporal']
                st.session_state.dq_geographical = sample_data['DQ_Geographical']
                st.session_state.dq_technological = sample_data['DQ_Technological']
                st.session_state.dq_datacollection = sample_data['DQ_DataCollection']
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close blue box
        
        # Sample data display - now inside the blue box styling
        st.markdown('<h3 style="color: white; margin-top: 15px;">Available Sample Data Options:</h3>', unsafe_allow_html=True)
        
        with st.expander("Office Administration Sample"):
            st.json(create_sample_data())
            
        with st.expander("Food Manufacturing Sample"):
            st.json(create_alternative_sample())
            
        with st.expander("Manufacturing Sample"):
            st.json(create_manufacturing_sample())
            
        with st.expander("Commodity Sample"):
            st.json(create_commodity_sample())
        
        st.info("""
        **Why use sample data?**
        - Quickly see how the tool works
        - Understand appropriate input ranges
        - Get a baseline for comparison
        - Test the prediction functionality
        """)

# Update session state with current values
st.session_state.source = source
st.session_state.code = code
st.session_state.name = name
st.session_state.substance = substance
st.session_state.unit = unit
st.session_state.year = year
st.session_state.dq_reliability = dq_reliability
st.session_state.dq_temporal = dq_temporal
st.session_state.dq_geographical = dq_geographical
st.session_state.dq_technological = dq_technological
st.session_state.dq_datacollection = dq_datacollection

# Prediction button
with col1:
    predict_button = st.button("🔮 Predict Emission Factors", use_container_width=True, type="primary")
    
    # Add expanders for advanced options
    with st.expander("⚙️ Advanced Model Settings"):
        st.markdown("""
        **Model Selection**
        - The app automatically uses the best performing model (XGBoost)
        - All models were trained on US industry and commodity data from 2010-2016
        
        **Data Quality Impact**
        - Higher quality data generally leads to more accurate predictions
        - The model accounts for data limitations in its uncertainty estimates
        """)

# Process prediction if button clicked
if predict_button:
    if not code or not name:
        st.warning("Please enter both Industry/Commodity Code and Name to make a prediction.")
    else:
        with st.spinner("Calculating emission factors..."):
            # Prepare input data
            input_data = {
                'Source': source,
                'Code': code,
                'Name': name,
                'Substance': substance,
                'Unit': unit,
                'Year': year,
                'DQ_Reliability': dq_reliability,
                'DQ_Temporal': dq_temporal,
                'DQ_Geographical': dq_geographical,
                'DQ_Technological': dq_technological,
                'DQ_DataCollection': dq_datacollection
            }
            
            # Make prediction
            try:
                if model is None:
                    st.error("Model failed to load. Please check that models are properly trained and saved.")
                else:
                    prediction = predict_emission_factors(input_data, model, preprocessor, model_name)
                    
                    # Create results for download
                    result_data = {
                        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        "Source": [source],
                        "Industry/Commodity Code": [code],
                        "Industry/Commodity Name": [name],
                        "Substance": [substance],
                        "Unit": [unit],
                        "Year": [year],
                        "DQ_Reliability": [dq_reliability],
                        "DQ_Temporal": [dq_temporal],
                        "DQ_Geographical": [dq_geographical],
                        "DQ_Technological": [dq_technological],
                        "DQ_DataCollection": [dq_datacollection],
                        "Predicted Emission Factor (kg CO2e)": [prediction],
                        "Model Used": [model_name]
                    }
                    result_df = pd.DataFrame(result_data)
                    
                    # Store result in session state for download
                    st.session_state.prediction_result = result_df
                    
                    # Display results in right column with dark background
                    with col2:
                        st.markdown('<div class="dark-bg">', unsafe_allow_html=True)
                        st.subheader("Prediction Results")
                        
                        # Display main metric prominently
                        st.metric(
                            label="Predicted Emission Factor",
                            value=f"{prediction:.4f} kg CO2e",
                            delta=f"Using {model_name} Model",
                            delta_color="off"
                        )
                        
                        # Add interpretation
                        interpretation = get_interpretation(prediction, input_data)
                        with st.expander("🔍 Interpretation", expanded=True):
                            st.markdown(interpretation)
                        
                        # Display data quality summary
                        st.markdown("**Data Quality Summary**")
                        
                        # Create a data quality score (simplified)
                        dq_score = (dq_reliability + dq_temporal + dq_technological) / 15 * 100
                        st.progress(int(dq_score))
                        st.caption(f"Data Quality Score: {dq_score:.1f}%")
                        
                        # Show which DQ factors most impact confidence
                        lowest_dq = min(
                            ("Reliability", dq_reliability),
                            ("Temporal", dq_temporal),
                            ("Technological", dq_technological),
                            key=lambda x: x[1]
                        )
                        
                        st.info(f"⚠️ **Lowest quality factor**: {lowest_dq[0]} (Score: {lowest_dq[1]}/5)")
                        
                        # Add visualization of prediction in context
                        st.markdown("**Contextual Comparison**")
                        
                        # Create a simple bar chart comparing to typical ranges
                        typical_ranges = {
                            'Very Low': (0.0, 0.05),
                            'Moderate': (0.05, 0.2),
                            'High': (0.2, 1.0)
                        }
                        
                        # Determine which range the prediction falls into
                        if prediction < 0.05:
                            current_range = 'Very Low'
                        elif prediction < 0.2:
                            current_range = 'Moderate'
                        else:
                            current_range = 'High'
                        
                        # Create a horizontal bar chart using Streamlit's native chart
                        st.caption("Your prediction falls in the **{}** range:".format(current_range))
                        st.bar_chart({
                            'Very Low (0.0-0.05)': 0.05,
                            'Moderate (0.05-0.2)': 0.2,
                            'High (0.2+)': 1.0,
                            'Your Prediction': prediction
                        })
                        
                        # Add recommendations based on prediction
                        st.markdown("**Recommendations**")
                        
                        if prediction < 0.05:
                            st.success("✅ Your supply chain appears highly efficient. Consider sharing best practices with industry peers.")
                        elif prediction < 0.2:
                            st.warning("⚠️ Moderate emissions detected. Consider energy efficiency improvements and renewable energy options.")
                        else:
                            st.error("❌ High emissions detected. Significant opportunities exist for reduction through process optimization and technology upgrades.")
                        
                        # Add download button for results
                        st.markdown(create_download_link(result_df), unsafe_allow_html=True)
                        
                        # Model information
                        with st.expander("ℹ️ Model Information"):
                            st.markdown(f"""
                            **Model Used:** {model_name}
                            
                            **Performance Metrics:**
                            - RMSE: 0.0364
                            - MAE: 0.0196
                            - R² Score: 0.9764
                            
                            This model was trained on US industry and commodity data from 2010-2016.
                            Predictions are most accurate for industries and commodities with similar data quality metrics to the training data.
                            """)
                        st.markdown('</div>', unsafe_allow_html=True)  # Close dark background
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please check your inputs and try again.")

# If no prediction has been made yet, show placeholder in right column
else:
    with col2:
        st.subheader("Prediction Results")
        st.info("Enter industry/commodity details and click 'Predict Emission Factors' to see results.")
        
        # Add a placeholder image or diagram
        st.image("https://images.unsplash.com/photo-1601531388842-5edf4e8cda52?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80", 
                 caption="Supply chain emissions visualization", use_column_width=True)
        
        st.markdown("""
        ### How to Use This Tool
        1. Enter industry/commodity details in the left panel
        2. Adjust data quality metrics based on your knowledge
        3. Click the **Predict Emission Factors** button
        4. Review your results and download if needed
        
        ### Why This Matters
        Understanding supply chain emissions is critical for:
        - Meeting sustainability goals
        - Identifying reduction opportunities
        - Reporting to stakeholders
        - Improving supply chain resilience
        """)

# Footer
st.markdown("""
<div class="footer">
Supply Chain Emission Factors Predictor | Developed by Srinath with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)

# Model status in sidebar
with st.sidebar:
    st.subheader("Model Status")
    
    if model is not None:
        st.success(f"✅ Model loaded: {model_name}")
        st.caption("Performance: RMSE=0.0364, R²=0.9764")
    else:
        st.warning("⚠️ Model not loaded")
        st.caption("Please train and save the model in the 'models' directory")
    
    st.markdown("---")
    
    # Data quality reference
    st.subheader("Data Quality Guide")
    st.markdown("""
    **Reliability (1-5):**
    5 = Expert-reviewed data
    3 = Industry averages
    1 = Rough estimates
    
    **Temporal (1-5):**
    5 = Current year data
    3 = 3-5 year old data
    1 = >10 year old data
    
    **Geographical (1-5):**
    5 = Exact US region match
    3 = National US data
    1 = International data
    
    **Technological (1-5):**
    5 = Exact process match
    3 = Similar processes
    1 = Very different processes
    """)