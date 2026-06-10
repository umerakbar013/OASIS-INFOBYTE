# 📈 Sales Prediction Using Python

> **OASIS INFOBYTE — Data Science Internship | Task 5**

A complete machine learning pipeline to predict **product sales** based on advertising spend across TV, Radio, and Newspaper channels.

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

Sales prediction means forecasting how much of a product people will buy based on advertising budgets. This project trains and compares five regression models on advertising spend data:

- **Linear Regression** — baseline
- **Ridge Regression** — L2 regularization
- **Lasso Regression** — L1 regularization
- **Random Forest** — ensemble model
- **Gradient Boosting** — best model (**R² = 0.9832**)

---

## 📊 Dataset

**File:** `data/Advertising.csv`
**Rows:** 200 | **Columns:** 4

| Column | Description |
|---|---|
| `TV` | TV advertising budget ($000) |
| `Radio` | Radio advertising budget ($000) |
| `Newspaper` | Newspaper advertising budget ($000) |
| `Sales` | Product sales ($000) — Target variable |

---

## 📁 Project Structure

```
sales_prediction/
│
├── data/
│   └── Advertising.csv
│
├── plots/
│   ├── 01_feature_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_channel_vs_sales.png
│   ├── 04_avg_budget_by_channel.png
│   ├── 05_model_comparison.png
│   ├── 06_actual_vs_predicted.png
│   ├── 07_feature_importance.png
│   └── 08_residuals.png
│
├── sales_prediction.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python sales_prediction.py
```

---

## 📈 Models & Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 1.4608 | 1.7816 | 0.8994 |
| Ridge Regression | 1.4643 | 1.7872 | 0.8988 |
| Lasso Regression | 1.4613 | 1.7913 | 0.8983 |
| Random Forest | 0.6287 | 0.7572 | 0.9818 |
| **Gradient Boosting** | **0.6161** | **0.7286** | **0.9832** |

🏆 **Gradient Boosting** achieved the best performance with **R² = 0.9832** — explaining 98.32% of variance in sales.

**Key Insight:** TV advertising has the strongest impact on sales, followed by Radio. Newspaper has minimal effect.

---

## 📊 Visualizations

| Plot | Description |
|---|---|
| Feature Distributions | Histogram for all 4 columns |
| Correlation Heatmap | Feature correlation matrix |
| Channel vs Sales | Scatter + trendline per channel |
| Avg Budget by Channel | Bar chart of mean budgets |
| Model Comparison | MAE / RMSE / R² bar charts |
| Actual vs Predicted | Best model fit scatter |
| Feature Importance | TV vs Radio vs Newspaper impact |
| Residuals | Residuals vs fitted + distribution |

---

## 🛠 Tech Stack

- **Python 3.x**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn** — Linear/Ridge/Lasso/RF/GB regression

---

## 👨‍💻 Author

**Umer Akbar** — BS Artificial Intelligence, SZABIST Islamabad
OASIS INFOBYTE Data Science Internship
