# PROJECT_CONTEXT.md
## RetainIQ — Customer Intelligence Dashboard

### What We're Building
A research-grade, industry-ready customer intelligence system on Telco churn data.
Not just a churn predictor — a full pipeline from raw data to a business-facing dashboard
that a data scientist trusts and a manager can use.

---

### Why This Project
- Demonstrates end-to-end ML: cleaning → EDA → segmentation → prediction → deployment
- Differentiates from existing portfolio (CyberShield, QualiEval) which are both security-focused
- Targets business audience — shows ability to translate model output into business decisions
- Research angle: imbalance handling comparison, threshold analysis, SHAP explainability

---

### Dataset
- Source: Telco Customer Churn (IBM Sample Dataset)
- File: `Telco_customer_churn.xlsx`
- Size: 7,043 customers, 33 columns → 29 after cleaning
- Target: `Churn Value` (1 = churned, 0 = stayed)
- Churn rate: 26.5% (mild imbalance)

---

### What's In Scope
- Customer Segmentation (KMeans clustering + PCA visualization)
- Churn Prediction (3-model comparison: LR vs XGBoost vs Random Forest)
- Imbalance handling comparison (class_weight vs SMOTE)
- Threshold tuning analysis
- SHAP explainability (global + per-customer)
- Business impact quantification (revenue at risk, tiered risk)
- Streamlit dashboard (5 tabs)

### What's Out of Scope
- Real-time data ingestion
- Live API deployment
- Prophet forecasting (moving average used instead — dataset has no real dates)
- Deep learning models

---

### Key Decisions Made

| Decision | Reason |
|---|---|
| Dropped `Churn Score` from modeling | Data leakage — computed from churn itself, wouldn't exist for new customers |
| Dropped `Churn Reason` | Only exists after customer has already left — future information |
| Dropped `Churn Label` | Redundant — same information as `Churn Value` in text format |
| Dropped `CustomerID`, `Count`, `Country`, `State`, `City`, `Lat Long` | ID columns, single-value columns, geographic noise |
| Dropped `Gender`, `Phone Service`, `Multiple Lines_No phone service` | SelectKBest score near 0 — no predictive signal |
| Kept `df` for EDA, created `df_ml` for modeling | EDA should see full data; model needs clean features only |
| Used `class_weight='balanced'` over SMOTE | Tested both — class_weight AUC 0.8466 vs SMOTE 0.8258 at 26.5% imbalance. Mild imbalance doesn't benefit from synthetic oversampling |
| Final model: Logistic Regression | Highest AUC (0.8466) and recall (0.78) among 3 models. Simple linear relationships in this dataset don't require ensemble complexity |
| Optimal threshold: 0.4 | Recall 0.87, F1 0.61 — best balance for churn use case where missing a churner costs more than a false alarm |
| Tiered risk approach | High Risk (>0.6), Medium Risk (0.4-0.6), Safe (<0.4) — more actionable than binary flag for retention team |
| K=4 for clustering | Elbow bend at k=4, silhouette score 0.34 — sufficient business granularity without overfitting segments |

---

### Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, imbalanced-learn
- SHAP
- Matplotlib, Seaborn
- Streamlit (dashboard)
- Pathlib (professional path handling)
- Joblib (model persistence)

---

### Project Structure
```
RetainAI/
├── PROJECT_CONTEXT.md
├── README.md
├── requirements.txt
├── notebooks/
│   └── customer_intelligence.ipynb
├── dashboard/
│   └── app.py
├── models/
│   ├── lr_churn_model.pkl
│   ├── kmeans_model.pkl
│   ├── kmeans_scaler.pkl
│   ├── feature_cols.pkl
│   └── df_ml_final.csv
└── plots/
    ├── churn_distribution.png
    ├── eda_plots.png
    ├── clustering_evaluation.png
    ├── pca_segments.png
    ├── segments.png
    ├── confusion_matrices.png
    ├── shap_importance.png
    ├── shap_impact.png
    ├── business_impact.png
    └── churn_forecast.png
```

---

### Key Results

**Clustering:**
| Segment | Customers | Avg Tenure | Avg Monthly Charges | Churn Rate |
|---|---|---|---|---|
| High Value Loyal | 1916 | 59 months | ₹93 | 16% |
| New At-Risk | 1950 | 11 months | ₹58 | 39% |
| Mid Value At-Risk | 2168 | 18 months | ₹61 | 35% |
| Budget Loyal | 1009 | 55 months | ₹32 | 4% |

**Model Comparison (class_weight, threshold=0.5):**
| Model | AUC | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.8466 | 0.78 | 0.62 |
| XGBoost | 0.8310 | 0.69 | 0.61 |
| Random Forest | 0.8364 | 0.49 | 0.55 |

**Research Finding — class_weight vs SMOTE:**
| Method | AUC | Recall |
|---|---|---|
| class_weight | 0.8466 | 0.78 |
| SMOTE | 0.8258 | 0.69 |

**Business Impact:**
- High Risk customers: 2,373 (33.7%)
- Monthly revenue at risk: ₹1,83,817
- Total revenue at risk (High + Medium): ₹2,47,212/month

---

### EDA Findings
- Short tenure = high churn risk (median 10 months for churned vs 40 for retained)
- Higher monthly charges = more churn (~₹80 vs ~₹60)
- Month-to-month contracts churn the most
- Fiber optic internet users churn heavily despite paying more
- Electronic check payment method = highest churn
- Churn is highest in months 1-6 (47-62%) and stabilizes at ~7% after month 60

---

