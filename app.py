
import streamlit as st
from model_utils import (load_artifacts, predict_revenue, predict_return_risk,
                         risk_level, label_for)

st.set_page_config(page_title="ML Portfolio Project", page_icon="📈")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


art = get_artifacts()


def input_widget(prefix, name, spec):
    # One input per feature: dropdown for categories, slider for numbers
    key = prefix + "_" + name
    if spec["type"] == "category":
        options = spec["options"]
        return st.selectbox(label_for(name), options, index=options.index(spec["default"]), key=key)
    return st.slider(label_for(name), min_value=spec["min"], max_value=spec["max"],
                     value=spec["default"], step=spec["step"], key=key)


st.title("ML Portfolio Project: Prediction App")
tab_reg, tab_clf = st.tabs(["Campaign revenue (Regression)", "Order return risk (Classification)"])

# ---------------- Regression tab ----------------
with tab_reg:
    st.subheader("Forecast the 14-day revenue of an influencer campaign")
    reg_values = {name: input_widget("reg", name, spec) for name, spec in art["reg_inputs"].items()}

    if st.button("Predict revenue", key="reg_button"):
        revenue = predict_revenue(art, reg_values)
        st.metric("Predicted revenue (14 days)", "$" + format(revenue, ",.0f"))
        st.caption("Typical error of this model on unseen campaigns: about +/- $"
                   + format(art["reg_mae"], ",.0f"))

# ---------------- Classification tab ----------------
with tab_clf:
    st.subheader("Estimate the return risk of an order")
    clf_values = {name: input_widget("clf", name, spec) for name, spec in art["clf_inputs"].items()}

    if st.button("Predict return risk", key="clf_button"):
        probability = predict_return_risk(art, clf_values)
        level, action = risk_level(probability)
        st.metric("Return probability", format(probability * 100, ".1f") + "%")
        if level == "Low risk":
            st.success(level + ": " + action)
        elif level == "Medium risk":
            st.warning(level + ": " + action)
        else:
            st.error(level + ": " + action)
