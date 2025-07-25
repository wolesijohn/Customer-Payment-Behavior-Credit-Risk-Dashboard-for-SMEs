# credit_risk_dashboard.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("sme_engineered_customer_data.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("credit_risk_model.pkl")
    le_industry = joblib.load("le_industry.pkl")
    le_region = joblib.load("le_region.pkl")
    return model, le_industry, le_region

# Main app
def main():
    st.set_page_config(page_title="Customer Payment Risk Dashboard", layout="wide")
    st.title("📊 Customer Payment Behavior & Credit Risk Dashboard")

    df = load_data()
    model, le_industry, le_region = load_model()

    # Sidebar filters
    st.sidebar.header("Filter Customers")
    industry_filter = st.sidebar.multiselect("Industry", options=df["Industry"].unique(), default=df["Industry"].unique())
    region_filter = st.sidebar.multiselect("Region", options=df["Region"].unique(), default=df["Region"].unique())

    filtered_df = df[(df["Industry"].isin(industry_filter)) & (df["Region"].isin(region_filter))]

    st.subheader("📌 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{filtered_df['Customer_ID'].nunique()}")
    col2.metric("Avg Late Payment Rate", f"{filtered_df['Late_Payment_Rate'].mean():.2%}")
    col3.metric("Avg Delay (Days)", f"{filtered_df['Avg_Delay_Days'].mean():.1f}")

    st.markdown("### 🧭 Payment Behavior by Industry")
    fig1 = px.box(filtered_df, x="Industry", y="Avg_Delay_Days", color="Industry")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🧮 Predict Credit Risk for Selected Customers")
    if st.checkbox("Run Prediction on Displayed Customers"):
        # Encode features
        filtered_df["Industry_encoded"] = le_industry.transform(filtered_df["Industry"])
        filtered_df["Region_encoded"] = le_region.transform(filtered_df["Region"])

        features = [
            "Late_Payment_Rate", "Default_Rate", "Avg_Delay_Days",
            "Total_Amount_Invoiced", "Credit_Term_Days",
            "Industry_encoded", "Region_encoded"
        ]
        preds = model.predict(filtered_df[features])
        probas = model.predict_proba(filtered_df[features])[:, 1]

        filtered_df["Predicted_Risk"] = preds
        filtered_df["Risk_Probability"] = probas
        filtered_df["Risk_Label"] = filtered_df["Predicted_Risk"].map({0: "Low/Medium", 1: "High"})

        st.markdown("### 🔍 Risk Predictions")
        st.dataframe(filtered_df[[
            "Customer_ID", "Industry", "Region", "Late_Payment_Rate",
            "Avg_Delay_Days", "Default_Rate", "Total_Amount_Invoiced",
            "Risk_Probability", "Risk_Label"
        ]].sort_values("Risk_Probability", ascending=False).reset_index(drop=True))

        fig2 = px.histogram(filtered_df, x="Risk_Probability", nbins=20, title="Distribution of Predicted Risk Scores")
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
