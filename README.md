# NBA MVP Prediction

A machine learning project that predicts NBA Most Valuable Player (MVP) award winners using a two-stage modeling approach with 93% accuracy.

## Project Overview

This project aims to predict NBA MVP award winners using player statistics and advanced metrics. By analyzing historical data from 1983 onwards, the model achieves **93% accuracy** in predicting the exact MVP winner and **95% accuracy** in placing the actual MVP within the top 3 candidates.

## Data Sources

Original Source: [Kaggle - Sumitro Datta](https://kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats)

- Per 100 Possession Statistics
- Advanced Player Statistics
- Player Award Shares (for MVP voting records)

All data is from the 1982-83 season onward, as the NBA's statistical tracking methods were standardized around this time.

## Methodology

### Data Preprocessing
- Combined statistics for players who played for multiple teams within a season
- Handled missing values using domain-appropriate methods
- Removed highly correlated features to avoid multicollinearity
- Created target variables for both MVP candidates and winners

### Feature Engineering & Selection
- Used feature importance analysis to identify key predictors
- Removed redundant statistics based on correlation analysis
- Kept the most meaningful basketball metrics while reducing dimensionality

### Two-Stage Modeling Approach

#### Stage 1: MVP Candidate Identification
- Predicts which players receive any MVP votes
- Uses XGBoost classifier optimized for high recall
- Creates a filtered pool of MVP candidates

#### Stage 2: MVP Winner Selection
- Takes only MVP candidates from Stage 1
- Uses a separate XGBoost classifier optimized for precision
- Ranks candidates by probability of winning

This approach mimics the actual MVP selection process (first identifying worthy players, then selecting a winner) and significantly improves accuracy compared to single-model approaches.

## Performance

- **Top-1 Accuracy**: 93% (correctly identified 40 of 43 MVPs in test data)
- **Top-3 Accuracy**: 95% (placed actual MVP in top 3 for 41 of 43 seasons)

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- missingno
- joblib

## Installation and Usage

1. Clone this repository
   ```bash
   git clone https://github.com/mertso13/nba-mvp-prediction.git
   cd nba-mvp-prediction
   ```

2. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to see the full analysis
   ```bash
   jupyter notebook mvp_prediction.ipynb
   ```

## To use the trained model on new data

```python
import pandas as pd
import joblib

# Load the models
stage1_model = joblib.load('models/mvp_stage1_model.pkl')
stage2_model = joblib.load('models/mvp_stage2_model.pkl')

# Load your data
new_season_data = pd.read_csv('your_data.csv')

# Predict MVP (using predict_mvp function from notebook)
predictions = predict_mvp(new_season_data, stage1_model, stage2_model)

# Display top 5 predicted MVP candidates
print(predictions[['player', 'candidate_prob', 'mvp_prob']].head(5))
```

## Key Findings

The most important features for MVP prediction include:
- Win Shares per 48 minutes (ws_48)
- Box Plus/Minus (bpm)
- True-shot percentage (ts_percent)
- Usage percentage (usg_percent)
- Team success metrics

The two-stage approach significantly outperforms single-model approaches:
- Stage 1 properly identifies MVP candidates with high recall
- Stage 2 excels at selecting the winner among legitimate candidates