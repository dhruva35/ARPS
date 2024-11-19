# App Rating Prediction System (ARPS)

A comprehensive machine learning system for predicting mobile application ratings across multiple platforms (Google Play Store, Apple App Store, and Amazon App Store).

## Project Overview

The App Rating Prediction System (ARPS) is designed to analyze and predict mobile application ratings using various features such as app size, price, category, and platform. The system processes data from multiple app stores and uses machine learning models to provide accurate rating predictions.

## Features

- Multi-platform support (Google Play Store, Apple App Store, Amazon App Store)
- Robust data preprocessing pipeline
- Multiple machine learning models (Random Forest and Linear Regression)
- Feature importance analysis
- Cross-validation for model evaluation
- Model persistence for later use

## Project Structure

```
ARPS/
├── src/
│   ├── data/
│   │   ├── raw/                    # Raw data from different app stores
│   │   ├── preprocessor.py         # Data preprocessing module
│   │   └── process_data.py         # Data processing script
│   ├── models/
│   │   ├── saved/                  # Saved model files
│   │   ├── model_trainer.py        # Model training module
│   │   └── train_models.py         # Model training script
│   └── web/                        # Web interface components
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ARPS
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python src/data/process_data.py
```

2. Train Models:
```bash
python src/models/train_models.py
```

## Model Performance

### Random Forest
- RMSE: 1.1972
- MAE: 1.0386
- Cross-validation RMSE: 1.1857

### Linear Regression
- RMSE: 1.1770
- MAE: 1.0312
- Cross-validation RMSE: 1.1704

## Feature Importance

1. App Size: 46.91%
2. Downloads: 15.01%
3. Price: 11.91%
4. Store Platform: ~3% each
5. App Categories: 1.81-2.61%

## Dependencies

- numpy
- pandas
- scikit-learn
- joblib
- pathlib
- logging

## Future Enhancements

1. Additional Features:
   - App description sentiment analysis
   - Developer reputation
   - Update frequency

2. Model Improvements:
   - Gradient Boosting models
   - Neural Networks
   - Ensemble methods

3. System Extensions:
   - Real-time prediction API
   - Batch prediction support
   - Web interface for predictions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
