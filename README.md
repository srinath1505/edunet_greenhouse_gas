# Supply Chain Emission Factors Prediction System

![Project Banner](https://home.nps.gov/goga/learn/nature/images/combined_good.jpg?maxwidth=1300&maxheight=1300&autorotate=false)

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27.0%2B-orange?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🌍 Project Description

This project provides a machine learning solution for predicting greenhouse gas (GHG) emission factors across US industries and commodities. The system estimates supply chain emissions per unit of economic activity (kg CO2e/2018 USD) using advanced regression models trained on historical data from 2010-2016.

The application helps businesses, researchers, and policymakers estimate their supply chain emissions based on industry characteristics and data quality metrics, enabling better carbon accounting and sustainability planning.

## ✨ Key Features

- **Advanced Prediction Models**: Uses XGBoost, LightGBM, and CatBoost for accurate emission factor estimation
- **Data Quality Integration**: Incorporates DQ metrics (Reliability, Temporal, Geographical, Technological, Data Collection) to improve prediction reliability
- **Interactive Web Interface**: Streamlit-based UI with intuitive input controls and visualizations
- **Multiple Data Sources**: Handles both industry and commodity data from 2010-2016
- **Comprehensive Interpretation**: Provides context and recommendations based on prediction results
- **Downloadable Results**: Export predictions for further analysis or reporting

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/supply-chain-emission-predictor.git
   cd supply-chain-emission-predictor
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download and place the dataset:
   - Obtain the `SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx` dataset
   - Place it in the project root directory

5. Train the models:
   ```bash
   python train_models.py
   ```

## 🖥️ Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The app will automatically open in your default web browser at `http://localhost:8501`

## 📚 Project Structure

```
edunet_greenhouse_gas/
├── data/                       # Dataset files
│   └── SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx
├── models/                     # Trained model files
│   ├── xgboost_model.pkl
│   ├── preprocessor.pkl
│   └── ...
├── app.py                      # Main Streamlit app
├── Supply_Chain_Emission_Analysis.ipynb                # basic analysis of dataset
├── updated_greenhouse_gas_emission_week2.ipynb                # week 2 submission
├── week1_submissinn.md         # week 1 markdown
├── final_GHG.py                # Final Model training script
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🧪 Model Performance

| Model      | RMSE    | MAE     | R² Score |
|------------|---------|---------|----------|
| XGBoost    | 0.0364  | 0.0196  | 0.9764   |
| LightGBM   | 0.0794  | 0.0418  | 0.8877   |
| CatBoost   | 0.0724  | 0.0226  | 0.9066   |

*Note: XGBoost was selected as the primary model due to its superior accuracy and handling of the dataset structure.*

## 🎨 Application Interface

### Input Parameters
- **Data Source**: Select from industry data (2010-2016) or commodity data
- **Industry/Commodity Details**: Code, name, substance, unit, and year
- **Data Quality Metrics**: Five-point scale for reliability, temporal correlation, etc.

### Prediction Results

- **Emission Factor**: Main prediction with model information
- **Interpretation**: Contextual analysis of the results
- **Data Quality Summary**: Visual representation of input quality
- **Contextual Comparison**: Placement within typical emission ranges
- **Recommendations**: Actionable insights based on prediction

## 📦 Dependencies

The project requires the following Python packages:

```
pandas==2.0.3
numpy==1.24.4
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==3.3.5
catboost==1.2.1
streamlit==1.27.0
joblib==1.3.2
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📬 Contact

For questions or collaboration opportunities, please contact:

Project Maintainer - Srinath - Srinathselvakumar1505@gmail.com

Project Link: [https://github.com/srinath1505/edunet_greenhouse_gas](https://github.com/srinath1505/edunet_greenhouse_gas)

---

*This project was developed to support accurate carbon accounting and supply chain sustainability analysis using US industry and commodity data from 2010-2016.*
