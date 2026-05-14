#  RetainIQ — Customer Intelligence Dashboard

Research-grade churn prediction with business-ready output. Built to answer one question: which customers will leave, why, and what does that cost?


**What This Project Does**
Most churn projects stop at model accuracy. This one goes further:

-Segments 7,043 customers into 4 behavioral groups using KMeans clustering
-Predicts individual churn probability with a validated Logistic Regression model
-Explains each prediction using SHAP — not just which features matter globally, but why a specific customer is at risk
-Quantifies business impact — ₹2,47,212/month in revenue at risk across risk tiers
-Delivers a Streamlit dashboard where a retention team can look up any customer, score new customers in bulk, and see the full research methodology

**Key Results**
Model Performance
Three models were compared with two imbalance handling techniques. Logistic Regression with class_weight='balanced' won on both AUC (0.8466) and recall (0.78). XGBoost achieved AUC 0.8310 with recall 0.69. Random Forest achieved AUC 0.8364 but recall dropped to 0.49 — missing too many actual churners to be useful.
Final model: Logistic Regression · class_weight='balanced' · threshold 0.4 · AUC 0.8466 · Recall 0.87
Why LR over ensemble models: This dataset has largely linear relationships. Adding ensemble complexity didn't improve results — it hurt recall on the minority class.
Research Finding — class_weight vs SMOTE
At 26.5% imbalance (mild), SMOTE hurt performance across all 3 models. class_weight adjusts the loss function penalty without generating synthetic samples — cleaner signal, better results. LR with class_weight: AUC 0.8466, Recall 0.78. LR with SMOTE: AUC 0.8258, Recall 0.69. Tested across all 3 models. Same pattern held.
Customer Segments
KMeans with k=4 (chosen via elbow method + silhouette score 0.34) identified 4 distinct segments:
Mid Value At-Risk — 2,168 customers · 35% churn rate · ₹61 avg monthly charges · largest group, highest retention priority
New At-Risk — 1,950 customers · 39% churn rate · ₹58 avg monthly charges · new customers leaving early, onboarding problem
High Value Loyal — 1,916 customers · 16% churn rate · ₹93 avg monthly charges · fewer customers but highest individual revenue, personal outreach justified
Budget Loyal — 1,009 customers · 4% churn rate · ₹32 avg monthly charges · very stable, low priority
Business Impact

2,373 High Risk customers — churn probability above 0.6
1,027 Medium Risk customers — churn probability 0.4–0.6
₹1,83,817/month at immediate risk (High Risk only)
₹2,47,212/month total revenue at risk (High + Medium combined)


**Threshold Analysis**
Default threshold of 0.5 is rarely optimal on imbalanced data. Lowering it catches more churners at the cost of more false alarms — the right tradeoff depends on retention budget.
At threshold 0.3: Recall 0.90, Precision 0.43, F1 0.59 — catches almost everyone but too many false alarms.
At threshold 0.4: Recall 0.87, Precision 0.48, F1 0.61 — best balance for this use case.
At threshold 0.5: Recall 0.78, Precision 0.52, F1 0.62 — higher precision but misses more real churners.
Chosen: 0.4 — For churn, missing a churner costs more than a wasted retention call. Recall of 0.87 catches most at-risk customers while keeping precision acceptable.

**Project Structure**
RetainAI/
├── notebooks/
│   └── customer_intelligence.ipynb   # Full analysis pipeline
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── models/
│   ├── lr_churn_model.pkl
│   ├── kmeans_model.pkl
│   ├── kmeans_scaler.pkl
│   ├── feature_cols.pkl
│   └── df_ml_final.csv
├── plots/                            # All generated visualizations
├── PROJECT_CONTEXT.md                # Decision log
├── requirements.txt
└── README.md

**Dashboard*8
Overview — KPI cards: total customers, churn rate, high risk count, revenue at risk
Customer Lookup — Search any Customer ID → see risk tier, churn probability, segment, recommended action
Segments — Segment profiles, revenue at risk by segment, churn rate comparison
Batch Scoring — Upload a CSV of new customers → model scores all of them → download results with risk tiers
Model Research — Full Round 1-4 comparison: class_weight vs SMOTE, all 3 models, threshold tuning results

**Analysis Pipeline**
1. Load & Inspect       → shape, nulls, duplicates, unique values
2. EDA                  → distributions, churn by segment, feature correlations
3. Feature Engineering  → encode categoricals, fix dtypes, SelectKBest selection
4. Clustering           → KMeans + elbow + silhouette → 4 segments + PCA visualization
5. Churn Prediction     → 3 models × 2 imbalance methods × 5 thresholds
6. SHAP Explainability  → global feature importance + per-customer waterfall
7. Business Impact      → tiered risk, revenue at risk, retention priority
8. Forecasting          → churn rate over tenure, 6-month moving average forecast
9. Save Models          → pkl + processed data for dashboard

**Tech Stack**
Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn · SHAP · Matplotlib · Seaborn · Streamlit 

**What I Learned**
Data quality over model complexity. LR outperformed XGBoost here because the data's signal is largely linear. Model choice matters less than understanding your data.
SMOTE is not a default. At mild imbalance, synthetic oversampling adds noise. class_weight handles it more cleanly. Always test both.
Threshold is a business decision. The optimal threshold depends on whether missing a churner or wasting a retention call is more costly. It's not a technical choice — it's a business one.
SHAP bridges the gap. Feature importance tells you what matters globally. SHAP tells a retention manager why their specific customer is at risk. That's the difference between a data science output and a business tool.
