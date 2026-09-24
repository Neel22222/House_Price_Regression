# 🏠 House Price Prediction

A full-stack ML web application that predicts residential house prices using a trained scikit-learn model, served via a **FastAPI** backend and a **Streamlit** frontend.

---

## Project Structure

```
House_Price_Regression/
├── house_price_regression_dataset.csv   # Training dataset (1,000 rows)
├── train_model.py                       # Model training script
├── app.py                               # FastAPI backend
├── frontend.py                          # Streamlit frontend
├── requirements.txt                     # Python dependencies
├── model/
│   ├── model.pkl                        # Saved trained model
│   └── meta.json                        # Feature stats & metrics
└── plots/
    └── feature_importance.png           # Feature importance chart
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This compares LinearRegression, RandomForest, and GradientBoosting — picks the best by R², and saves it to `model/`.

---

## Running the App

You need **two terminals**:

### Terminal 1 — Start the FastAPI backend
```bash
uvicorn app:app --reload
```
API docs available at: http://localhost:8000/docs

### Terminal 2 — Start the Streamlit frontend
```bash
streamlit run frontend.py
```
UI available at: http://localhost:8501

---

## API Endpoints

| Method | Endpoint      | Description                        |
|--------|---------------|------------------------------------|
| GET    | `/`           | Health check                       |
| GET    | `/model-info` | Model name, metrics, feature stats |
| POST   | `/predict`    | Predict house price                |

### POST /predict — Request Body
```json
{
  "Square_Footage": 2000,
  "Num_Bedrooms": 3,
  "Num_Bathrooms": 2,
  "Year_Built": 2005,
  "Lot_Size": 2.5,
  "Garage_Size": 1,
  "Neighborhood_Quality": 7
}
```

---

## Dataset Features

| Feature               | Description                        |
|-----------------------|------------------------------------|
| Square_Footage        | Total interior living area (sq ft) |
| Num_Bedrooms          | Number of bedrooms (1–5)           |
| Num_Bathrooms         | Number of bathrooms (1–5)          |
| Year_Built            | Year the house was constructed     |
| Lot_Size              | Land area in acres                 |
| Garage_Size           | Garage capacity (cars, 0–5)        |
| Neighborhood_Quality  | Quality score 1–10                 |
| House_Price           | Target — sale price in USD         |

---

## Model Results

| Model             | MAE      | RMSE     | R²     |
|-------------------|----------|----------|--------|
| LinearRegression  | $8,175   | $10,071  | 0.9984 |
| GradientBoosting  | $12,196  | $14,881  | 0.9966 |
| RandomForest      | $16,035  | $19,678  | 0.9940 |
