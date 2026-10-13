import os
import joblib
import pandas as pd

ARTIFACT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deployment_artifacts.joblib")

# Friendly names shown in the apps
LABELS = {
    "creator_followers_k": "Creator followers (thousands)",
    "engagement_rate_pct": "Engagement rate (%)",
    "paid_boost_spend_usd": "Paid boost spend (USD)",
    "avg_watch_time_sec": "Average watch time (seconds)",
    "promo_discount_pct": "Promo discount offered (%)",
    "content_format": "Content format",
    "audience_intent_tier": "Audience intent tier",
    "cart_value_usd": "Cart value (USD)",
    "past_return_rate": "Customer's past return rate (0 to 1)",
    "size_bracketing_flag": "Size bracketing (1 = several sizes of same item in cart)",
    "delivery_days_taken": "Delivery time (days)",
    "discount_pct_applied": "Discount applied at checkout (%)",
    "product_category": "Product category",
    "fit_review_sentiment": "Fit review sentiment",
}


def label_for(name):
    return LABELS.get(name, name)


def load_artifacts(path=ARTIFACT_FILE):
    return joblib.load(path)


def build_row(raw_inputs, all_columns, cap_limits, selected_features):
    # Start with a row of zeros that has every training column
    row = pd.DataFrame(0.0, index=[0], columns=all_columns)
    for name, value in raw_inputs.items():
        if name in row.columns:
            row[name] = float(value)              # numeric column
        else:
            dummy_name = name + "_" + str(value)  # category -> its one-hot column
            if dummy_name in row.columns:
                row[dummy_name] = 1.0             # (the reference category stays all zeros)
    # Cap extreme values with the limits learned from the training data
    for col, (low, high) in cap_limits.items():
        row[col] = row[col].clip(low, high)
    return row[selected_features]


def predict_revenue(art, inputs):
    row = build_row(inputs, art["reg_all_columns"], art["reg_cap_limits"], art["reg_features"])
    return max(0.0, float(art["reg_model"].predict(row)[0]))


def predict_return_risk(art, inputs):
    row = build_row(inputs, art["clf_all_columns"], art["clf_cap_limits"], art["clf_features"])
    return float(art["clf_model"].predict_proba(row)[0, 1])


def risk_level(probability):
    # Returns a risk label and a suggested business action
    if probability < 0.30:
        return "Low risk", "No action needed. Process the order as normal."
    if probability < 0.70:
        return "Medium risk", "Show size-guide / fit tips at checkout and send a sizing reminder after purchase."
    return "High risk", "Offer a size consultation or exchange-first option, and consider limiting free returns on this order."
