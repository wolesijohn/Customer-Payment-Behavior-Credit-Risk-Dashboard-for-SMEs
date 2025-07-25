import pandas as pd

# Load CSV files
customers_df = pd.read_csv("sme_customers.csv", parse_dates=["Customer_Since"])
invoices_df = pd.read_csv("sme_invoices.csv", parse_dates=["Invoice_Date", "Due_Date", "Paid_Date"])

# Merge customer and invoice data
df = invoices_df.merge(customers_df, on="Customer_ID", how="left")

# Feature Engineering

# 1. Extract year-month for trend analysis
df["Invoice_YearMonth"] = df["Invoice_Date"].dt.to_period("M").astype(str)

# 2. Flag late payments (paid after due date)
df["Is_Late"] = df["Delay_Days"].apply(lambda x: x > 0 if pd.notnull(x) else False)

# 3. Flag default (unpaid invoice)
df["Is_Default"] = df["Is_Paid"] == False

# 4. Group features per customer
agg_features = df.groupby("Customer_ID").agg(
    Total_Invoices=('Amount', 'count'),
    Avg_Invoice_Amount=('Amount', 'mean'),
    Late_Payment_Rate=('Is_Late', 'mean'),
    Default_Rate=('Is_Default', 'mean'),
    Avg_Delay_Days=('Delay_Days', 'mean'),
    Total_Amount_Invoiced=('Amount', 'sum'),
    Total_Amount_Paid=('Amount', lambda x: x[df.loc[x.index, "Is_Paid"]].sum())
).reset_index()

# 5. Merge aggregated features with customer profile
engineered_df = customers_df.merge(agg_features, on="Customer_ID", how="left")

# 6. Fill missing delay days for customers without invoices
engineered_df["Avg_Delay_Days"] = engineered_df["Avg_Delay_Days"].fillna(0)

# 7. Export the engineered dataset
engineered_df.to_csv("sme_engineered_customer_data.csv", index=False)
