# Football-pass-prediction
Machine Learning Project 3 - Competition Football pass prediction


## 👥 Authors

Samira Ben Ahmed (s2503328)

Mohamed-Khalil Ankri (s2502523)

Ishahk Hamad (s2402246)


# Football Pass Prediction (Project 3)

## 📌 Project Overview
This project lies within the field of sports analytics. [cite_start]The objective is to predict the recipient of a football pass based on a snapshot of player positions during a match[cite: 10]. The solution involves training a machine learning model to estimate both the **receiver ID** (classification) and the **probability** of each player receiving the ball, optimizing for two key metrics:
* [cite_start]**Classification Accuracy ($Acc_{TS}$)**: The percentage of correct receiver predictions[cite: 32].
* **Brier Score ($BS_{TS}$)**: A measure of the calibration quality of the predicted probabilities.

**Course:** ELEN0062 - Introduction to Machine Learning  
**Date:** December 2025

---

## 📂 Repository Structure

```text
football-pass-prediction/
├── data/
│   ├── raw/                  # Place the 3 original CSV files here
│   │   ├── input_train_set.csv
│   │   ├── output_train_set.csv
│   │   └── input_test_set.csv
│   └── processed/            # Generated feature files (created automatically)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
    ├── 03_model_training.ipynb
│   └── 04_make_submission.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py           # Feature engineering logic (Distance, Angle, Pressure, Ranks)
│   ├── models.py             # Model definitions (HistGradientBoosting, Calibration)
│   ├── utils.py              # Data loading helpers
│   └── make_submission.py    # Main script to generate the final submission
├── submissions/              # Output folder for Gradescope CSVs
├── requirements.txt          # Python dependencies
└── README.md