# 📧 Email Spam Detection with Machine Learning

> **OASIS INFOBYTE — Data Science Internship | Task 4**

A complete NLP + machine learning pipeline to classify emails as **Spam** or **Ham (Not Spam)** using TF-IDF vectorization and multiple classification models.

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

Spam mail is a type of junk message sent to massive numbers of users, frequently containing scams or phishing content. This project builds a spam detector using NLP techniques and compares three classification models:

- **Naive Bayes** — classic NLP baseline
- **Logistic Regression** — linear classifier
- **Random Forest** — best performing model (**F1 = 0.9084**)

---

## 📊 Dataset

**File:** `data/spam.csv`
**Rows:** 5,572 messages

| Column | Description |
|---|---|
| `Label` | `spam` or `ham` |
| `Message` | Raw email/SMS text |

**Class Distribution:** 4,825 Ham · 747 Spam

---

## 📁 Project Structure

```
email_spam_detection/
│
├── data/
│   └── spam.csv
│
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_message_length.png
│   ├── 03_avg_length_by_class.png
│   ├── 04_model_comparison.png
│   ├── 05_confusion_matrices.png
│   ├── 06_roc_curves.png
│   └── 07_top_spam_words.png
│
├── email_spam_detection.py
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
python email_spam_detection.py
```

---

## 📈 Models & Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|
| Naive Bayes | 0.9731 | 0.9524 | 0.7919 | 0.8649 | 0.9891 |
| Logistic Regression | 0.9713 | 0.9916 | 0.7919 | 0.8806 | 0.9856 |
| **Random Forest** | **0.9776** | **1.0000** | **0.8322** | **0.9084** | **0.9807** |

🏆 **Random Forest** achieved the best F1 Score of **0.9084** with **100% Precision** — zero false positives.

---

## 📊 Visualizations

| Plot | Description |
|---|---|
| Class Distribution | Spam vs Ham bar chart |
| Message Length | Length distribution per class |
| Avg Length by Class | Average characters in spam vs ham |
| Model Comparison | Accuracy / Precision / Recall / F1 / AUC |
| Confusion Matrices | Per-model confusion matrices |
| ROC Curves | All three models on one chart |
| Top Spam Words | Top 20 spam indicator words (Naive Bayes) |

---

## 🛠 Tech Stack

- **Python 3.x**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn** — TF-IDF, Naive Bayes, Logistic Regression, Random Forest

---

## 👨‍💻 Author

**Umer Akbar** — BS Artificial Intelligence, SZABIST Islamabad
OASIS INFOBYTE Data Science Internship
