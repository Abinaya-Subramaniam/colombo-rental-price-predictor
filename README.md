# Colombo Rental Price Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A Machine Learning system for predicting residential rental prices in Colombo, Sri Lanka using robust regression techniques and ensemble methods. Designed to handle real estate market outliers and provide actionable business insights.

---

## Overview

This project addresses the critical need for accurate rental price estimation in urban real estate markets by developing unconventional models that capture complex relationships between property features and rental values. Unlike traditional approaches, we focus on robust statistical methods that resist outliers and provide reliable predictions across diverse property types.

### Why This Project?

- **Market Need**: Colombo's rental market lacks transparent pricing mechanisms
- **Data Challenges**: Real estate data contains outliers, missing values, and non-linear relationships
- **Business Impact**: Enables data-driven decisions for landlords, tenants, and agents
- **Technical Innovation**: Combines robust regression with modern explainability techniques

---

## Key Features

- **Robust Algorithms**: Theil-Sen, Passive Aggressive, and ensemble models
- **Outlier Resistance**: Handles premium properties and data anomalies effectively
- **SHAP Explainability**: Transparent feature importance and business insights
- **Production Pipeline**: Modular, reproducible, and deployment-ready architecture
- **Market Specific**: Tailored for Colombo real estate market characteristics
- **Business Insights**: Actionable recommendations for stakeholders

---

## Project Structure

```
colombo-rental-predictor/
├── data/                   
│   ├── raw/               
│   ├── processed/     
│ 
│── api/
├   ├── __init__.py
├── ├── main.py      
│
├── rental_price_prediction/  
│   ├── __init__.py       
│   ├── config.py          
│   ├── dataset.py         
│   ├── features.py        
│   ├── modeling/          
│   │   ├── train.py      
│   │   └── predict.py    
│   └── plots.py          
│
├── notebooks/              
│
├── docs/                  
│   ├── data_dictionary.md
│   └── experiments/
│       └── feature_engineering.md
│       └── model_comparision.md
│
├── models/                
├── reports/               
│   └── figures/          
├── references/            
├── tests/                 
│
├── requirements.txt       
├── setup.cfg             
├── pyproject.toml        
├── Makefile              
├── LICENSE               
└── README.md            
```

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/Abinaya-Subramaniam/colombo-rental-predictor.git
cd colombo-rental-predictor

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

```

### Basic Usage

```python
from rental_price_prediction.modeling.predict import predict_rental_price
from rental_price_prediction.dataset import load_data

model = load_model('models/best_model.pkl')

property_features = {
    'Size_in_Sqft': 1200,
    'Bedrooms': 3,
    'Bathrooms': 2,
    'Neighborhood': 'Downtown',
    'Furnished': 'Furnished',
    'Building_Type': 'Apartment',
    'Property_Age': 5,
    'Distance_to_City_Center': 2.5
}

predicted_price = predict_rental_price(model, property_features)
print(f"Predicted Rental Price: LKR {predicted_price:,.2f}")
```
## API & Docker Deployment

This project provides a FastAPI-based REST API to serve rental price predictions. You can run it locally or using Docker.

### Using Docker


```bash
docker build -t colombo-rental-api .
docker run -d -p 8000:8000 colombo-rental-api
http://localhost:8000/docs  


## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Data Guide](docs/data_dictionary.md)**: Detailed Data Descriptions
- **[Model Documentation](docs/model_documentation.md)**: Model Details
- **[Feature Engineering](docs/experiments/feature_engineering.md)**: Feature Transformation details
- **[Model Experiments](docs/experiments/)**: Experiment Tracking and results

---

## Methodology

### Data Processing Pipeline

1. **Data Cleaning**: Missing value imputation, outlier removal, text standardization
2. **Feature Engineering**: Scaling numerical features, one-hot encoding categoricals
3. **Feature Selection**: SHAP-based importance ranking
4. **Model Training**: Cross-validated ensemble with robust regressors
5. **Evaluation**: Multi-metric assessment with business context

### Models Implemented

- **Huber Regressor**: Robust linear regression resistant to outliers with epsilon-insensitive loss
- **RANSAC Regressor**: Random Sample Consensus for robust outlier rejection
- **Voting Ensemble**: Combines Huber and RANSAC with optimized weights [3, 1]
- **Tree-Based Models**: Random Forest, Gradient Boosting, XGBoost for comparison
- **Baseline Models**: Decision Tree, Bagging for performance benchmarking

### Key Technologies

- **Python 3.8+**: Core language
- **scikit-learn**: Machine learning framework
- **pandas**: Data manipulation
- **SHAP**: Model explainability
- **matplotlib/seaborn**: Visualization

---

## Results & Insights

### Business Recommendations

- Properties over 1,500 sqft show exponential price growth
- Downtown properties justify 10-15% premium pricing
- Furnished properties attract 5-8% higher rents
- Properties within 3km of city center have strongest demand

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---


## Acknowledgments

- Cookiecutter Data Science project template
- Colombo real estate data providers
- scikit-learn community for robust regression implementations
- SHAP library for explainability tools

---

## Future Roadmap

- [ ] Add time-series analysis for rental trends
- [ ] Integrate geospatial features (GIS data)
- [ ] Expand to other Sri Lankan cities
- [ ] Build interactive web dashboard
- [ ] Add rental yield predictions for investors

---

**Note**: Model performance metrics are based on historical data and should be validated on current market conditions.