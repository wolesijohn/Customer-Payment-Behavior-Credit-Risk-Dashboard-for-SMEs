# credit_risk_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("sme_engineered_customer_data.csv")

# Create binary target: 'High' risk = 1, else 0
df["Risk_Target"] = df["Risk_Category"].apply(lambda x: 1 if x == "High" else 0)

# Encode categorical features
le_industry = LabelEncoder()
le_region = LabelEncoder()
df["Industry_encoded"] = le_industry.fit_transform(df["Industry"])
df["Region_encoded"] = le_region.fit_transform(df["Region"])

# Feature columns
features = [
    "Late_Payment_Rate", "Default_Rate", "Avg_Delay_Days",
    "Total_Amount_Invoiced", "Credit_Term_Days",
    "Industry_encoded", "Region_encoded"
]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df["Risk_Target"], test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "credit_risk_model.pkl")
joblib.dump(le_industry, "le_industry.pkl")
joblib.dump(le_region, "le_region.pkl")
