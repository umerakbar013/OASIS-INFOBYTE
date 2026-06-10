# 🚗 Car Price Prediction with Machine Learning

> **OASIS INFOBYTE — Data Science Internship | Task 3**

A complete machine learning pipeline to predict the **selling price of used cars** based on features such as fuel type, transmission, kilometers driven, present market price, and car age.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models & Results](#-models--results)
- [Visualizations](#-visualizations)
- [Tech Stack](#-tech-stack)

---

## 📌 Project Overview

The price of a car depends on many factors — brand goodwill, horsepower, mileage, and more. This project trains and compares three regression models to accurately predict used car selling prices:

- **Linear Regression** — baseline model
- **Random Forest Regressor** — ensemble tree model
- **Gradient Boosting Regressor** — best performing model (**R² = 0.9661**)

---

## 📊 Dataset

**File:** `data/car_data.csv`  
**Rows:** 301 | **Columns:** 9

| Column | Description |
|---|---|
| `Car_Name` | Name of the car |
| `Year` | Year of manufacture |
| `Selling_Price` | Target variable — price in Lakhs (₹) |
| `Present_Price` | Current showroom price in Lakhs (₹) |
| `Driven_kms` | Kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Selling_type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |

**Engineered Feature:** `Car_Age = 2024 - Year`

---

## 📁 Project Structure

```
car_price_prediction/
│
├── data/
│   └── car_data.csv              # Dataset
│
├── plots/
│   ├── 01_selling_price_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_scatter.png
│   ├── 04_fuel_type_boxplot.png
│   ├── 05_age_vs_price.png
│   ├── 06_model_comparison.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_feature_importance.png
│   └── 09_residuals.png
│
├── car_price_prediction.py       # Main script
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repo (if not already)
git clone https://github.com/umerakbar013/OASIS-INFOBYTE.git
cd OASIS-INFOBYTE

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
# Navigate to the task folder
cd car_price_prediction

# Run the script
python car_price_prediction.py
```

All 9 plots will be saved automatically to the `plots/` folder.

---

## 📈 Models & Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 1.2219 | 1.8792 | 0.8467 |
| Random Forest | 0.6441 | 0.9824 | 0.9581 |
| **Gradient Boosting** | **0.5476** | **0.8838** | **0.9661** |

🏆 **Gradient Boosting** achieved the best performance with **96.61% variance explained**.

---

## 📊 Visualizations

| Plot | Description |
|---|---|
| Selling Price Distribution | Histogram of the target variable |
| Correlation Heatmap | Feature correlation matrix |
| Present vs Selling Price | Scatter plot coloured by car age |
| Fuel Type Boxplot | Price distribution per fuel type |
| Car Age vs Price | Depreciation trend |
| Model Comparison | MAE / RMSE / R² bar charts |
| Actual vs Predicted | Best model fit plot |
| Feature Importance | Top features from Random Forest |
| Residuals Analysis | Residuals vs fitted + distribution |

---

## 🛠 Tech Stack

- **Python 3.x**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualizations
- **Scikit-learn** — machine learning models & evaluation

---

## 👨‍💻 Author

**Umer Akbar** — BS Artificial Intelligence, SZABIST Islamabad  
OASIS INFOBYTE Data Science Internship
