# Hydropower Generation Forecasting with Deep Learning Models

## Overview
This repository contains the implementation of a research project focused on forecasting hydropower generation at Temenggor Power Station in Perak, Malaysia, using deep learning models. The study investigates the relationship between weather conditions and hydropower output, leveraging localized data to improve forecasting accuracy in Malaysia’s tropical climate. The project employs feature selection techniques and compares the performance of four deep learning models: Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), CNN-LSTM, and LSTM-CNN.

## Repository Structure
```
├── data/
│   ├── era5_climate_data.csv     # ERA5 climate dataset
│   ├── temenggor_power_data.csv  # Temenggor hydropower generation data
├── paper/
│   ├── Mohd_Qaedi_Faiz_Application_of_Deep_Learning_Models_For_Forecasting_Hydropower_Plant_Power_Generation.pdf  # Research paper
├── scripts/
│   ├── eda_feature_selection.ipynb  # Exploratory Data Analysis and Mutual Information feature selection
│   ├── models/
│   │   ├── cnn.ipynb             # CNN model implementation and training
│   │   ├── lstm.ipynb            # LSTM model implementation, training, and 90-day forecasting
│   │   ├── cnn_lstm.ipynb        # CNN-LSTM hybrid model implementation and training
│   │   ├── lstm_cnn.ipynb        # LSTM-CNN hybrid model implementation and training
├── figures/
│   ├── eda_plots/                # EDA visualizations
│   ├── model_performance/        # Model performance graphs
│   ├── forecast_plots/           # Forecast vs actual plots
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

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
5. **Deployment**: Models trained on 2019–2023 data and used to forecast 90 days into 2024, with the LSTM model selected as the best performer.

## Results
- **Best Model**: LSTM achieved the lowest error metrics (MAE: 0.580, MSE: 0.531, RMSE: 0.729) for 7-day forecasts with filtered data.
- **Feature Impact**: Mean Dew Point Temperature, Eastward Wind Component, and Mean Temperature were the most influential variables.
- **Future Forecast**: The LSTM model provides reliable 90-day forecasts, implemented in `lstm.ipynb`, though it struggles with abrupt changes in power generation.
- **Performance Metrics**: The performance of the four deep learning models (CNN, LSTM, CNN-LSTM, LSTM-CNN) was evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) across 7, 30, and 90-day time steps. The results are presented for both unfiltered and filtered datasets, with the lowest error for each metric and time step highlighted in **bold**.

### Unfiltered Dataset
The following tables show the performance metrics for models trained on the unfiltered dataset, grouped by MAE, MSE, and RMSE:

#### MAE (Mean Absolute Error)
| Model     | 7 Days    | 30 Days   | 90 Days   |
|-----------|-----------|-----------|-----------|
| CNN       | 0.610     | 0.694     | 0.817     |
| LSTM      | **0.587** | **0.620** | **0.611** |
| CNN-LSTM  | 0.681     | 0.691     | 0.674     |
| LSTM-CNN  | 0.663     | 0.726     | 0.946     |

#### MSE (Mean Squared Error)
| Model     | 7 Days    | 30 Days   | 90 Days   |
|-----------|-----------|-----------|-----------|
| CNN       | 0.578     | 0.730     | 1.056     |
| LSTM      | **0.522** | **0.605** | **0.570** |
| CNN-LSTM  | 0.737     | 0.729     | 0.718     |
| LSTM-CNN  | 0.698     | 0.828     | 1.450     |

#### RMSE (Root Mean Squared Error)
| Model     | 7 Days    | 30 Days   | 90 Days   |
|-----------|-----------|-----------|-----------|
| CNN       | 0.760     | 0.854     | 1.028     |
| LSTM      | **0.723** | **0.778** | **0.755** |
| CNN-LSTM  | 0.858     | 0.854     | 0.847     |
| LSTM-CNN  | 0.836     | 0.910     | 1.204     |

### Filtered Dataset
The following tables show the performance metrics for models trained on the filtered dataset, which includes only the seven key variables identified via Mutual Information (MI), grouped by MAE, MSE, and RMSE:

#### MAE (Mean Absolute Error)
| Model     | 7 Days    | 30 Days   | 90 Days   |
|-----------|-----------|-----------|-----------|
| CNN       | 0.612     | **0.653** | 0.748     |
| LSTM      | **0.580** | 0.626     | **0.624** |
| CNN-LSTM  | 0.611     | 0.640     | 0.629     |
| LSTM-CNN  | 0.629     | 0.745     | 0.893     |

#### MSE (Mean Squared Error)
| Model     | 7 Days    | 30 Days   | 90 Days   |
|-----------|-----------|-----------|-----------|
| CNN       | 0.571     | 0.676     | 0.865     |
| LSTM      | **0.531** | **0.598** | **0.589** |
| CNN-LSTM  | 0.586     | 0.642     | 0.611     |
| LSTM-CNN  | 0.626     | 0.871     | 1.246     |

#### RMSE (Root Mean Squared Error)
| Model     | 7 Days    | 30 Days   | 90 Days   |
|-----------|-----------|-----------|-----------|
| CNN       | 0.756     | **0.822** | 0.930     |
| LSTM      | **0.729** | 0.773     | **0.767** |
| CNN-LSTM  | 0.766     | 0.801     | 0.782     |
| LSTM-CNN  | 0.791     | 0.933     | 1.116     |

**Performance Observations**:
- LSTM consistently outperformed other models across all time steps, with the best performance on the filtered dataset for 7-day forecasts (MAE: **0.580**, MSE: **0.531**, RMSE: **0.729**).
- Feature selection improved performance for most models, particularly for CNN and hybrid models at longer time steps (90 days).
- CNN struggled with longer time steps due to overfitting, while LSTM-CNN showed the highest errors, especially for 90-day forecasts.

## Key Findings
- **Feature Selection**: Improved model performance, particularly for CNN and hybrid models, with LSTM showing the best results using filtered datasets.
- **Model Performance**: LSTM outperformed other models across all time steps, achieving the lowest error metrics.
- **Challenges**: CNN struggled with longer time steps due to overfitting, while hybrid models showed moderate improvements but could not surpass LSTM.
- **Forecasting**: LSTM accurately captured general trends but struggled with abrupt changes in power generation. The 90-day future forecast is implemented in the LSTM model.

## Future Work
- Investigate advanced preprocessing techniques to handle noisy data and abrupt changes.
- Explore additional hybrid models or ensemble methods to improve long-term forecasting.
- Implement real-time forecasting at Temenggor Power Station for operational use.
- Incorporate more climate scenarios to enhance model resilience.

## Research Paper
The full research paper, detailing the methodology, results, and findings, is available in the `paper/` folder as a PDF:
- [Application of Deep Learning Models for Forecasting Hydropower Plant Power Generation](./paper/Mohd_Qaedi_Faiz_Application_of_Deep_Learning_Models_For_Forecasting_Hydropower_Plant_Power_Generation.pdf)

## Citation
If you use this code or data, please cite:
> Mohd Qaedi Faiz Abdul Aziz and Sofianita Mutalib. "Application of Deep Learning Models for Forecasting Hydropower Plant Power Generation." Universiti Teknologi MARA, 2025.

## Contact
For inquiries, contact:
- Mohd Qaedi Faiz Abdul Aziz (mohdqaedifaiz@example.com)
- Sofianita Mutalib (sofianita@example.com)