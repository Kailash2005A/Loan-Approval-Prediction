# =========================
# Professional Loan Approval Predictor
# =========================

import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1E3A8A;
}

.stButton>button {
    width: 100%;
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1D4ED8;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------

model = joblib.load("loan_model.pkl")

label_encoders = joblib.load("label_encoders.pkl")

# -------------------------
# Header
# -------------------------

st.title("🏦 Loan Approval Predictor")

st.divider()

# -------------------------
# Layout
# -------------------------

left_col, right_col = st.columns([1, 1])

# =========================================================
# LEFT SIDE → INPUTS
# =========================================================

with left_col:

    st.subheader("📋 Applicant Details")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    income_annum = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        value=700000,
        step=50000
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        value=200000,
        step=50000
    )

    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        value=5
    )

    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=700
    )

    total_assets = st.number_input(
        "Total Assets Value (₹)",
        min_value=0,
        value=800000,
        step=50000
    )

    predict_button = st.button("Predict Loan Status")

# =========================================================
# RIGHT SIDE → RESULTS
# =========================================================

with right_col:

    st.subheader("📊 Loan Analysis Report")

    if predict_button:

        # =====================================================
        # Feature Engineering
        # =====================================================

        loan_to_income_ratio = loan_amount / (income_annum + 1)

        asset_to_loan_ratio = total_assets / (loan_amount + 1)

        repayment_capacity = income_annum

        # =====================================================
        # EMI Calculation
        # =====================================================

        monthly_interest_rate = 0.08 / 12

        months = loan_term * 12

        if months > 0:

            estimated_emi = (
                loan_amount *
                monthly_interest_rate *
                (1 + monthly_interest_rate) ** months
            ) / (
                ((1 + monthly_interest_rate) ** months) - 1
            )

        else:

            estimated_emi = 0

        monthly_income = income_annum / 12

        emi_ratio = estimated_emi / (monthly_income + 1)

        # =====================================================
        # Encode Categories
        # =====================================================

        gender_encoded = label_encoders["gender"].transform([gender])[0]

        education_encoded = label_encoders["education"].transform([education])[0]

        self_employed_encoded = label_encoders["self_employed"].transform([self_employed])[0]

        # =====================================================
        # Create Input Data
        # =====================================================

        input_data = pd.DataFrame([[
            gender_encoded,
            education_encoded,
            self_employed_encoded,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            total_assets,
            loan_to_income_ratio,
            asset_to_loan_ratio
        ]], columns=[
            "gender",
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "total_assets",
            "loan_to_income_ratio",
            "asset_to_loan_ratio"
        ])

        # =====================================================
        # Banking Rules
        # =====================================================

        rejection_reason = ""

        risk_score = 85

        # Low CIBIL
        if cibil_score < 550:

            rejection_reason = "Low CIBIL Score"

            risk_score = 20

        # Loan too high
        elif loan_amount > income_annum * 3:

            rejection_reason = "Loan amount too high compared to annual income"

            risk_score = 30

        # Assets too low
        elif total_assets < loan_amount * 0.5:

            rejection_reason = "Insufficient assets"

            risk_score = 35

        # Low repayment capacity
        elif repayment_capacity < 100000:

            rejection_reason = "Low repayment capacity"

            risk_score = 45

        # EMI burden too high
        elif emi_ratio > 0.5:

            rejection_reason = "Monthly EMI burden too high"

            risk_score = 25

        # =====================================================
        # Applicant Summary
        # =====================================================

        st.markdown("### 🧾 Applicant Summary")

        st.write(f"**Annual Income:** ₹{income_annum:,}")

        st.write(f"**Loan Amount:** ₹{loan_amount:,}")

        st.write(f"**Loan Term:** {loan_term} Years")

        st.write(f"**CIBIL Score:** {cibil_score}")

        st.write(f"**Total Assets:** ₹{total_assets:,}")

        st.divider()

        # =====================================================
        # Rule Based Rejection
        # =====================================================

        if rejection_reason != "":

            st.error("❌ Loan Rejected")

            st.warning(rejection_reason)

            st.markdown("### ⚠️ Risk Analysis")

            st.progress(risk_score)

            st.write("Risk Level: HIGH")

        else:

            # =================================================
            # ML Prediction
            # =================================================

            prediction = model.predict(input_data)[0]

            result = label_encoders["loan_status"].inverse_transform([prediction])[0]

            if result == "Approved":

                st.success("✅ Loan Approved")

                st.markdown("### 📈 Risk Analysis")

                st.progress(risk_score)

                st.write("Risk Level: LOW")

            else:

                st.error("❌ Loan Rejected")

                st.markdown("### 📈 Risk Analysis")

                st.progress(50)

                st.write("Risk Level: MEDIUM")

        st.divider()

        # =====================================================
        # Financial Metrics
        # =====================================================

        st.markdown("### 📊 Financial Metrics")

        st.write(f"**Loan to Income Ratio:** {loan_to_income_ratio:.2f}")

        st.write(f"**Asset to Loan Ratio:** {asset_to_loan_ratio:.2f}")

        st.write(f"**Repayment Capacity:** ₹{repayment_capacity:,.0f}")

        st.write(f"**Estimated Monthly EMI:** ₹{estimated_emi:,.0f}")

        st.write(f"**EMI to Income Ratio:** {emi_ratio:.2f}")