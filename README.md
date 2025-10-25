# NFL Win Probability Prediction System

A real-time machine learning system that predicts NFL game outcomes with high accuracy using ensemble methods and temporal validation strategies.

## 🏈 Overview

This project develops a win probability model for NFL games that processes play-by-play data to predict game outcomes in real-time. The system achieves a Brier score of 0.0881, surpassing ESPN's benchmark of 0.0905, while maintaining low prediction latency suitable for live broadcasts.

## 📊 Key Features

- **Real-time Predictions**: Low latency for live game predictions
- **Dual Model System**: 
  - Win probability model for in-game predictions (Brier score: 0.0881)
  - Game outcome classifier for pre-game predictions (Accuracy: 66.7%)
- **Temporal Validation**: Chronological train/test splits to ensure real-world performance
- **Advanced Feature Engineering**: EPA metrics, momentum indicators, and Vegas line integration

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
lightgbm >= 3.3.0
nflreadpy >= 0.3.0
```

### Installation
```bash
# Clone repository
git clone https://github.com/SunnyYadav16/nfl-prediction-using-ml.git
cd nfl-prediction-using-ml
```

## 🔧 Pipeline Overview

### 1. Data Collection & Cleaning
- 370,000+ plays from 2018-2025 NFL seasons
- Handles missing values, team relocations, data type conversions
- Filters to meaningful competitive plays only

### 2. Feature Engineering
- **Game State**: Score differential, time remaining, down/distance
- **Team Performance**: Win percentage, 3-game momentum EPA
- **Vegas Lines**: Spread, over/under, market expectations
- **Situational**: Division games, primetime, playoffs

### 3. Model Architecture
- **XGBoost** (40%): Primary gradient boosting model
- **Random Forest** (30%): Probability classification
- **LightGBM** (30%): Fast gradient boosting
- **Isotonic Calibration**: Probability adjustment

### 4. Temporal Validation
```
Train: 2018-2022 seasons (210,847 plays)
Valid: 2023 season (42,156 plays)  
Test: 2024 season (38,472 plays)
```

## 📈 Performance Metrics

| Metric | Our Model | ESPN Benchmark | Improvement |
|--------|-----------|----------------|-------------|
| Brier Score | 0.0881 | 0.0905 | +2.7% |
| Accuracy | 73.2% | ~70% | +4.6% |
| Latency | 89ms | N/A | Production Ready |
| Calibration Error | 0.012 | N/A | Excellent |

## 🎯 Key Insights

1. **Vegas Lines Dominate**: Spread line accounts for 12.1% of feature importance
2. **Time Matters**: Score differential impact varies dramatically with time remaining
3. **Momentum is Real**: 3-game EPA rolling average adds 8.7% predictive power
4. **Home Advantage**: Home teams win 52.7% of games

### Weekly Predictions
```python
# Predict entire week
predictions = predict_week_games(season=2024, week=10)
```

## 📚 References

- [NFL Data Python Tutorials](https://github.com/tejseth/nfl-tutorials-2022)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [EPA Methodology](https://www.nflfastr.com/articles/beginners_guide.html)

## 📧 Contact
Sunny Yadav - yadav.sunny@northeastern.edu
