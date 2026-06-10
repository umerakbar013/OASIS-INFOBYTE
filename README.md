# 📊 Unemployment Analysis with Python

A data science project analysing India's unemployment trends using real-world data, with a focus on the sharp increase caused by the **COVID-19 pandemic**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Analysis Performed](#analysis-performed)
- [Key Findings](#key-findings)
- [Generated Plots](#generated-plots)
- [Technologies Used](#technologies-used)

---

## Overview

Unemployment is measured by the **unemployment rate** — the number of people unemployed as a percentage of the total labour force. This project analyses India's unemployment data across 28 states from **May 2019 to October 2020**, revealing the dramatic impact of COVID-19 on the Indian workforce.

> *"We have seen a sharp increase in the unemployment rate during Covid-19 — analysing this data tells a powerful story about the pandemic's economic impact."*

---

## Datasets

| File | Period | Rows | States |
|---|---|---|---|
| `Unemployment_in_India.csv` | May 2019 – Jun 2020 | 740 | 28 |
| `Unemployment_Rate_upto_11_2020.csv` | Jan 2020 – Oct 2020 | 267 | 27 |

### Columns

| Column | Description |
|---|---|
| `Region` | Indian state name |
| `Date` | Observation date (monthly) |
| `Estimated Unemployment Rate (%)` | % of labour force unemployed |
| `Estimated Employed` | Number of employed persons |
| `Estimated Labour Participation Rate (%)` | % of working-age population in labour force |
| `Area` | Rural / Urban |
| `Geo_Region` *(Dataset 2 only)* | North / South / East / West / Northeast |

---

## Project Structure

```
unemployment_project/
│
├── 📂 data/
│   ├── Unemployment_in_India.csv
│   └── Unemployment_Rate_upto_11_2020.csv
│
├── 📂 plots/                          ← 10 charts auto-saved here
│   ├── 01_overall_trend.png
│   ├── 02_statewise_avg.png
│   ├── 03_rural_vs_urban.png
│   ├── 04_covid_impact.png
│   ├── 05_heatmap_state_month.png
│   ├── 06_top10_covid_states.png
│   ├── 07_geo_region_trend.png
│   ├── 08_labour_vs_unemployment.png
│   ├── 09_monthly_boxplot.png
│   └── 10_employment_trend.png
│
├── 📄 unemployment_analysis.py        ← Main script — run this
├── 📄 requirements.txt
└── 📄 README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1 — Open project in VS Code
```
File → Open Folder → unemployment_project
```

### Step 2 — Open terminal in VS Code
Press **Ctrl + `**

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python unemployment_analysis.py
```

The script will:
1. Load and clean both datasets
2. Print summary statistics to terminal
3. Generate and save 10 plots to `plots/`
4. Print key insights and findings

---

## Analysis Performed

| # | Analysis | Description |
|---|---|---|
| 1 | Overall Trend | National average unemployment over time with COVID marker |
| 2 | State-wise Average | Bar chart comparing all 28 states |
| 3 | Rural vs Urban | Side-by-side trend lines for Rural and Urban areas |
| 4 | COVID Impact | Per-state change in unemployment: pre vs during COVID |
| 5 | State × Month Heatmap | Full heatmap across all states and months |
| 6 | Top 10 COVID States | States with highest peak unemployment during COVID |
| 7 | Geographic Regions | North / South / East / West / Northeast comparison |
| 8 | Labour Participation | Scatter plot: labour participation vs unemployment rate |
| 9 | Monthly Distribution | Boxplots showing unemployment spread month by month |
| 10 | Employment Trend | Area chart of total estimated employed population |

---

## Key Findings

| Metric | Value |
|---|---|
| Pre-COVID average unemployment | **9.51%** |
| During-COVID average unemployment | **17.77%** |
| Increase caused by COVID-19 | **+8.26 percentage points** |
| Peak unemployment rate | **76.74%** (Puducherry, April 2020) |
| Highest avg unemployment state | **Tripura** |
| Lowest avg unemployment state | **Meghalaya** |
| Rural average unemployment | **10.32%** |
| Urban average unemployment | **13.17%** |

### Notable Observations

- **COVID-19 nearly doubled** the national unemployment rate
- **April 2020** saw the sharpest spike, coinciding with India's nationwide lockdown
- **Urban areas** consistently showed higher unemployment than Rural areas (+2.84%)
- **Petal features (petal length & width)** are the most discriminative — wait, wrong project! 😄
- **Northeast states** showed more resilience compared to Western states
- Labour participation rate has a **negative correlation** with unemployment

---

## Generated Plots

| Plot | Filename | Key Insight |
|---|---|---|
| Overall Trend | `01_overall_trend.png` | Spike visible from March 2020 |
| State-wise Avg | `02_statewise_avg.png` | Tripura & Haryana highest |
| Rural vs Urban | `03_rural_vs_urban.png` | Urban hit harder during COVID |
| COVID Impact | `04_covid_impact.png` | All states impacted; some more than others |
| Heatmap | `05_heatmap_state_month.png` | April 2020 column is darkest |
| Top 10 COVID | `06_top10_covid_states.png` | Puducherry, Jharkhand, Bihar worst hit |
| Geo Regions | `07_geo_region_trend.png` | All regions spike in April 2020 |
| Labour Scatter | `08_labour_vs_unemployment.png` | Clear pre/during COVID separation |
| Monthly Box | `09_monthly_boxplot.png` | April–May 2020 highest spread |
| Employment | `10_employment_trend.png` | Sharp drop in April 2020 |

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Pandas](https://img.shields.io/badge/pandas-data-green)
![Matplotlib](https://img.shields.io/badge/matplotlib-plots-orange)
![Seaborn](https://img.shields.io/badge/seaborn-stats-red)

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, aggregation |
| `numpy` | Numerical operations, regression |
| `matplotlib` | Line charts, bar charts, area charts |
| `seaborn` | Heatmap, boxplots, scatter plots |

---

*OASIS INFOBYTE — Data Science Internship | Task 2*  
*SZABIST University — BS Artificial Intelligence*
