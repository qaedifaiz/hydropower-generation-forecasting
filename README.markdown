# Hydropower Generation Forecasting with Deep Learning Models

## Overview
This repository contains the implementation of a research project focused on forecasting hydropower generation at Temenggor Power Station in Perak, Malaysia, using deep learning models. The study investigates the relationship between weather conditions and hydropower output, leveraging localized data to improve forecasting accuracy in Malaysia’s tropical climate. The project employs feature selection techniques and compares the performance of four deep learning models: Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), CNN-LSTM, and LSTM-CNN.

## Research Objectives
1. Conduct Exploratory Data Analysis (EDA) to understand the dataset and identify patterns in hydropower generation.
2. Apply feature selection using Mutual Information (MI) to identify key weather variables impacting hydropower generation.
3. Develop and compare deep learning models (CNN, LSTM, CNN-LSTM, LSTM-CNN) for forecasting accuracy across different time steps (7, 30, and 90 days).

## Dataset
The dataset comprises:
- **ERA5 Climate Data**: Sourced from ClimateEngine.org, including variables like precipitation, temperature, wind speed, and pressure, collected from 2019 to 2023.
- **Temenggor Power Station Data**: Daily hydropower generation data (in MW) from 2019 to 2023, measured every 30 minutes and aggregated daily.

The dataset is split into 70% training and 30% testing samples for model development and evaluation.

## Methodology
The research follows the CRISP-DM framework:
1. **Data Understanding**: EDA to analyze trends and variability in hydropower generation.
2. **Feature Selection**: Mutual Information (MI) test to identify seven key weather variables (e.g., Mean Dew Point Temperature, Eastward Wind Component).
3. **Modeling**: Four deep learning models (CNN, LSTM, CNN-LSTM, LSTM-CNN) implemented using Python’s Keras library with ReLU activation and He weight initialization.
4. **Evaluation**: Models evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) across 7, 30, and 90-day time steps.
5. **Deployment**: Models trained on 2019–2023 data and used to forecast 90 days into 2024.

## Key Findings
- **Feature Selection**: Improved model performance, particularly for CNN and hybrid models, with LSTM showing the best results using filtered datasets.
- **Model Performance**: LSTM outperformed other models across all time steps, achieving the lowest error metrics (MAE: 0.580, MSE: 0.531, RMSE: 0.729 for 7-day forecasts with filtered data).
- **Challenges**: CNN struggled with longer time steps due to overfitting, while hybrid models showed moderate improvements but could not surpass LSTM.
- **Forecasting**: LSTM accurately captured general trends but struggled with abrupt spikes in power generation.

## Repository Structure
```
├── data/
│   ├── era5_climate_data.csv     # ERA5 climate dataset
│   ├── temenggor_power_data.csv  # Temenggor hydropower generation data
├── scripts/
│   ├── eda.py                    # Exploratory Data Analysis
│   ├── feature_selection.py      # Mutual Information feature selection
│   ├── models/
│   │   ├── cnn.py                # CNN model implementation
│   │   ├── lstm.py               # LSTM model implementation
│   │   ├── cnn_lstm.py           # CNN-LSTM hybrid model
│   │   ├── lstm_cnn.py           # LSTM-CNN hybrid model
│   ├── train.py                  # Model training and evaluation
│   ├── forecast.py               # Future forecasting script
├── figures/
│   ├── eda_plots/                # EDA visualizations
│   ├── model_performance/        # Model performance graphs
│   ├── forecast_plots/           # Forecast vs actual plots
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hydropower-forecasting.git
   cd hydropower-forecasting
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Python 3.8+ and the following libraries are installed:
   - TensorFlow/Keras
   - NumPy
   - Pandas
   - Matplotlib
   - Scikit-learn

## Usage
1. **Exploratory Data Analysis**:
   ```bash
   python scripts/eda.py
   ```
   Outputs visualizations in `figures/eda_plots/`.

2. **Feature Selection**:
   ```bash
   python scripts/feature_selection.py
   ```
   Generates filtered dataset based on MI scores.

3. **Train Models**:
   ```bash
   python scripts/train.py
   ```
   Trains all models and saves performance metrics.

4. **Forecasting**:
   ```bash
   python scripts/forecast.py
   ```
   Generates 90-day forecasts using the LSTM model.

## Results
- **Best Model**: LSTM achieved the lowest error metrics (MAE: 0.580, MSE: 0.531, RMSE: 0.729) for 7-day forecasts using filtered data.
- **Feature Impact**: Mean Dew Point Temperature, Eastward Wind Component, and Mean Temperature were the most influential variables.
- **Future Forecast**: The LSTM model provides reliable 90-day forecasts, though it struggles with abrupt changes in power generation.

## Future Work
- Investigate advanced preprocessing techniques to handle noisy data and abrupt changes.
- Explore additional hybrid models or ensemble methods to improve long-term forecasting.
- Implement real-time forecasting at Temenggor Power Station for operational use.
- Incorporate more climate scenarios to enhance model resilience.

## Citation
If you use this code or data, please cite:
> Mohd Qaedi Faiz Abdul Aziz and Sofianita Mutalib. "Application of Deep Learning Models for Forecasting Hydropower Plant Power Generation." Universiti Teknologi MARA, 2025.

## Contact
For inquiries, contact:
- Mohd Qaedi Faiz Abdul Aziz (mohdqaedifaiz@example.com)
- Sofianita Mutalib (sofianita@example.com)