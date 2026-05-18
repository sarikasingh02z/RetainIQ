import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="RetainIQ — Customer Intelligence",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv('models/df_ml_final.csv')

@st.cache_resource
def load_models():
    model = joblib.load('models/lr_churn_model.pkl')
    scaler = joblib.load('models/kmeans_scaler.pkl')
    kmeans = joblib.load('models/kmeans_model.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, scaler, kmeans, feature_cols

df = load_data()
model, scaler, kmeans, feature_cols = load_models()

st.sidebar.title("RetainIQ")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Customer Lookup",
    "Segments",
    "Batch Scoring",
    "Model Research"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Telco Churn · 7,043 customers")
st.sidebar.markdown("**Model:** Logistic Regression · AUC 0.8466")
st.sidebar.markdown("**Threshold:** 0.4 · Recall 0.87")

if page == "Overview":
    st.title("Customer Intelligence Dashboard")
    st.markdown("Churn prediction · Customer segmentation · Revenue impact")
    st.markdown("---")
    st.markdown("""
**RetainIQ** identifies which customers are likely to leave, explains why using SHAP, 
and quantifies the revenue at risk — so retention teams know exactly who to call and when.
**Pipeline:** Data cleaning → EDA → KMeans segmentation → 3-model churn prediction → 
SMOTE vs class_weight comparison → threshold tuning → SHAP explainability → business impact
**Final model:** Logistic Regression · AUC 0.8466 · Threshold 0.4 · Recall 0.87
    """)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    high_risk = len(df[df['Risk_Tier'] == 'High Risk'])
    revenue_at_risk = df[df['Risk_Tier'] == 'High Risk']['Monthly_Revenue'].sum()
    churn_rate = df['Churn Value'].mean() * 100

    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("High Risk Customers", f"{high_risk:,}")
    col4.metric("Monthly Revenue at Risk", f"₹{revenue_at_risk:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segments")
        seg_counts = df['Segment_Name'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        seg_counts.plot(kind='bar', ax=ax, color=['steelblue','tomato','orange','green'], edgecolor='white')
        ax.set_title('Segment Distribution', fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Risk Tier Distribution")
        risk_counts = df['Risk_Tier'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        risk_counts.plot(kind='bar', ax=ax, color=['tomato','orange','steelblue'], edgecolor='white')
        ax.set_title('Risk Tiers', fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        st.pyplot(fig)
        plt.close()

elif page == "Customer Lookup":
    st.title("Customer Lookup")
    st.markdown("Search any customer to see their churn risk profile.")
    st.markdown("---")

    customer_ids = df['CustomerID'].tolist()
    search = st.selectbox("Search Customer ID", options=customer_ids)

    if search:
        customer = df[df['CustomerID'] == search].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Tier", customer['Risk_Tier'])
        col2.metric("Churn Probability", f"{customer['Churn_Probability']:.1%}")
        col3.metric("Segment", customer['Segment_Name'])

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Profile")
            st.write(f"**Tenure Months:** {customer['Tenure Months']}")
            st.write(f"**Monthly Revenue:** ₹{customer['Monthly_Revenue']:.2f}")
            st.write(f"**Churn Probability:** {customer['Churn_Probability']:.1%}")
            st.write(f"**Risk Tier:** {customer['Risk_Tier']}")
            st.write(f"**Segment:** {customer['Segment_Name']}")

        with col2:
            st.subheader("Recommended Action")
            actions = {
                'High Risk': 'Immediate outreach — offer retention deal',
                'Medium Risk': 'Monitor — schedule follow up next month',
                'Safe': 'No action needed'
            }
            st.info(actions.get(customer['Risk_Tier'], 'N/A'))
            if customer['Risk_Tier'] == 'High Risk':
                st.warning(f"This customer contributes ₹{customer['Monthly_Revenue']:.0f}/month. Losing them costs ₹{customer['Monthly_Revenue']*12:.0f}/year.")

elif page == "Segments":
    st.title("Customer Segments")
    st.markdown("KMeans clustering on Tenure, Monthly Charges, Total Charges, CLTV")
    st.markdown("---")

    st.subheader("Segment Profiles")
    seg_profile = df.groupby('Segment_Name').agg(
        Customers=('Churn Value', 'count'),
        Avg_Tenure=('Tenure Months', 'mean'),
        Avg_Monthly_Charges=('Monthly Charges', 'mean'),
        Churn_Rate=('Churn Value', 'mean'),
        High_Risk_Count=('Risk_Tier', lambda x: (x == 'High Risk').sum())
    ).round(2)
    seg_profile['Churn_Rate'] = (seg_profile['Churn_Rate'] * 100).round(1).astype(str) + '%'
    st.dataframe(seg_profile, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue at Risk by Segment")
        seg_revenue = df[df['Risk_Tier'] == 'High Risk'].groupby('Segment_Name')['Monthly_Revenue'].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        seg_revenue.plot(kind='bar', ax=ax, color=['tomato','orange','steelblue','green'], edgecolor='white')
        ax.set_title('High Risk Revenue by Segment', fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        ax.set_ylabel('Monthly Revenue (₹)')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn Rate by Segment")
        seg_churn = df.groupby('Segment_Name')['Churn Value'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        seg_churn.plot(kind='bar', ax=ax, color=['steelblue','tomato','orange','green'], edgecolor='white')
        ax.set_title('Churn Rate by Segment', fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        ax.set_ylabel('Churn Rate (%)')
        st.pyplot(fig)
        plt.close()

    selected_seg = st.selectbox("Filter by segment", options=['All'] + df['Segment_Name'].unique().tolist())
    if selected_seg != 'All':
        st.markdown("---")
        st.subheader(f"Customers in {selected_seg}")
        filtered = df[df['Segment_Name'] == selected_seg][['CustomerID', 'Tenure Months', 'Monthly_Revenue', 'Churn_Probability', 'Risk_Tier']].round(2)
        st.dataframe(filtered, use_container_width=True)

elif page == "Batch Scoring":
    st.title("Batch Customer Scoring")
    st.markdown("Upload a CSV of new customers to get churn risk predictions.")
    st.markdown("---")
    st.info("CSV must have the same columns as the training data — no CustomerID, no Churn columns needed.")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(new_df)} customers")
        st.dataframe(new_df.head(), use_container_width=True)

        try:
            new_df_aligned = new_df.reindex(columns=feature_cols, fill_value=0)
            probs = model.predict_proba(new_df_aligned)[:, 1]

            def risk_tier(p):
                if p >= 0.6: return 'High Risk'
                elif p >= 0.4: return 'Medium Risk'
                else: return 'Safe'

            results = new_df.copy()
            results['Churn_Probability'] = probs.round(3)
            results['Risk_Tier'] = [risk_tier(p) for p in probs]
            results = results.sort_values('Churn_Probability', ascending=False)

            st.markdown("---")
            st.subheader("Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("High Risk", len(results[results['Risk_Tier'] == 'High Risk']))
            col2.metric("Medium Risk", len(results[results['Risk_Tier'] == 'Medium Risk']))
            col3.metric("Safe", len(results[results['Risk_Tier'] == 'Safe']))

            st.dataframe(results, use_container_width=True)
            st.download_button("Download Results CSV", results.to_csv(index=False),
                               file_name="scored_customers.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Model Research":
    st.title("Model Research")
    st.markdown("Imbalance handling comparison + threshold analysis")
    st.markdown("---")

    st.subheader("Round 1 — Class Weight vs SMOTE (threshold=0.5)")
    st.markdown("""
**Logistic Regression** · class_weight: AUC 0.8466 · Recall 0.78 · F1 0.62  
**Logistic Regression** · SMOTE: AUC 0.8258 · Recall 0.69 · F1 0.61  
**XGBoost** · class_weight: AUC 0.8310 · Recall 0.69 · F1 0.61  
**XGBoost** · SMOTE: AUC 0.8277 · Recall 0.62 · F1 0.60  
**Random Forest** · class_weight: AUC 0.8364 · Recall 0.49 · F1 0.55  
**Random Forest** · SMOTE: AUC 0.8334 · Recall 0.64 · F1 0.60  
    """)

    st.markdown("---")
    st.subheader("Round 2 — LR + class_weight + Threshold Tuning")
    st.markdown("""
Threshold 0.30 · Recall 0.8957 · Precision 0.4345 · F1 0.5852  
Threshold 0.35 · Recall 0.8824 · Precision 0.4539 · F1 0.5995  
**Threshold 0.40 · Recall 0.8663 · Precision 0.4758 · F1 0.6142 — chosen**  
Threshold 0.45 · Recall 0.8316 · Precision 0.4984 · F1 0.6232  
Threshold 0.50 · Recall 0.7807 · Precision 0.5150 · F1 0.6206  
    """)

    st.markdown("---")
    st.subheader("Key Findings")
    st.success("Class weight outperforms SMOTE at 26.5% imbalance — AUC 0.8466 vs 0.8258")
    st.success("Optimal threshold: 0.4 — Recall 0.87, balances catching churners vs false alarms")
    st.info("Tested across 3 models. Same pattern held — class_weight consistently better at mild imbalance.")
    st.info("At mild imbalance, SMOTE adds noise. class_weight adjusts penalty without synthetic data.")
