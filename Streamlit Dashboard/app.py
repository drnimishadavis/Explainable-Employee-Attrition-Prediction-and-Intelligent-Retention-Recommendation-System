import streamlit as st
import pandas as pd
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Employee Attrition Intelligence AI",
    page_icon="🏢",
    layout="wide"
)

# =====================================================
# LOAD MODEL FILES
# =====================================================
@st.cache_resource
def load_resources():
    model = pickle.load(open("xgb_attrition_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    explainer = pickle.load(open("shap_explainer.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
    recommendation_mapping = pickle.load(open("recommendation_mapping.pkl", "rb"))
    return model, scaler, explainer, feature_names, recommendation_mapping

model, scaler, explainer, feature_names, recommendation_mapping = load_resources()

# =====================================================
# INITIALIZE SESSION STATE DEFAULTS
# =====================================================
if "active_form" not in st.session_state:
    st.session_state.active_form = "Profile"

# Set up global default values to prevent pipeline crashes on unvisited tabs
defaults = {
    "Age": 30, "Gender": "Male", "MaritalStatus": "Single", "DistanceFromHome": 5,
    "Education": 3, "EducationField": "Life Sciences", "TotalWorkingYears": 8, "NumCompaniesWorked": 2,
    "Department": "Research & Development", "JobRole": "Research Scientist", "JobLevel": 1,
    "BusinessTravel": "Travel_Rarely", "OverTime": "No", "YearsAtCompany": 5, "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3, "TrainingTimesLastYear": 3, "PerformanceRating": 3,
    "MonthlyIncome": 5000, "DailyRate": 800, "HourlyRate": 60, "MonthlyRate": 15000,
    "PercentSalaryHike": 15, "StockOptionLevel": 0, "EnvironmentSatisfaction": 3,
    "JobSatisfaction": 3, "RelationshipSatisfaction": 3, "WorkLifeBalance": 3, "JobInvolvement": 3
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =====================================================
# MODERN CSS WITH WHITE TEXT & METRICS FIXED
# =====================================================
st.markdown(
"""
<style>
/* Main background */
.stApp{
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #7c3aed);
}

/* Remove top padding */
.block-container{
    padding-top:2rem;
    max-width:1500px;
}

/* Website Navbar */
.navbar{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(20px);
    border-radius:25px;
    padding:30px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0px 10px 40px rgba(0,0,0,0.25);
    text-align:center;
    color:white;
}

.navbar h1{
    font-size:42px;
    font-weight:900;
    margin-bottom:10px;
}

.navbar p{
    font-size:18px;
}

/* Glass Cards */
.glass-card{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(20px);
    border-radius:25px;
    padding:25px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0px 15px 40px rgba(0,0,0,0.25);
    color:white;
    margin-bottom: 20px;
}

/* Result Card */
.result-card{
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(25px);
    border-radius:30px;
    padding:35px;
    text-align:center;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0px 15px 50px rgba(0,0,0,0.3);
    color:white;
}

/* Buttons */
.stButton button{
    height:55px;
    width:100%;
    border-radius:20px;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6);
    color:white;
    font-size:18px;
    font-weight:800;
    border:none;
}

/* Global Text Formatting */
h1, h2, h3, h4, p, label {
    color: white !important;
}

/* Targets Streamlit Metric Component Labels and Value Blocks */
[data-testid="stMetricLabel"] p {
    color: white !important;
}

[data-testid="stMetricValue"] div {
    color: white !important;
}
</style>
""",
    unsafe_allow_html=True
)

# =====================================================
# HEADER NAVBAR
# =====================================================
st.markdown(
"""
<div class="navbar">
<h1>🏢 Employee Attrition Intelligence AI</h1>
<p>XGBoost Prediction • SHAP Explainability • HR Retention Intelligence</p>
</div>
""",
    unsafe_allow_html=True
)
st.write("")

# =====================================================
# CUSTOM TABS NAVIGATION (3 TABS STYLE)
# =====================================================
nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("👤 Employee Profile", use_container_width=True):
        st.session_state.active_form = "Profile"
with nav2:
    if st.button("💼 Career Growth & Compensation", use_container_width=True):
        st.session_state.active_form = "Career"
with nav3:
    if st.button("😊 Satisfaction & Engagement", use_container_width=True):
        st.session_state.active_form = "Satisfaction"

# =====================================================
# MAIN TWO-COLUMN SPLIT
# =====================================================
left_panel, right_panel = st.columns([1, 1.25])

# =====================================================
# LEFT PANEL: DYNAMIC INTERACTIVE TAB FORMS
# =====================================================
with left_panel:
    
    # --- TAB 1: EMPLOYEE PROFILE ---
    if st.session_state.active_form == "Profile":
        st.markdown('<div class="glass-card"><h2>👤 Employee Profile</h2></div>', unsafe_allow_html=True)
        st.slider("Age", 18, 60, key="Age")
        st.selectbox("Gender", ["Male", "Female"], key="Gender")
        st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="MaritalStatus")
        st.slider("Distance From Home", 0, 50, key="DistanceFromHome")
        st.selectbox("Education Level", [1, 2, 3, 4, 5], key="Education")
        st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"], key="EducationField")
        st.number_input("Total Working Years", 0, 40, key="TotalWorkingYears")
        st.number_input("Number of Companies Worked", 0, 10, key="NumCompaniesWorked")

    # --- TAB 2: CAREER GROWTH & COMPENSATION (COMBINED) ---
    elif st.session_state.active_form == "Career":
        st.markdown('<div class="glass-card"><h2>💼 Career Growth & Compensation</h2></div>', unsafe_allow_html=True)
        
        # Sub-columns inside the tab to maintain structural hierarchy
        col_job, col_comp = st.columns(2)
        
        with col_job:
            st.subheader("Operational Profile")
            st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"], key="Department")
            st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Sales Representative", "Human Resources"], key="JobRole")
            st.selectbox("Job Level", [1, 2, 3, 4, 5], key="JobLevel")
            st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], key="BusinessTravel")
            st.selectbox("OverTime", ["Yes", "No"], key="OverTime")
            st.number_input("Years At Company", 0, 40, key="YearsAtCompany")
            st.number_input("Years In Current Role", 0, 20, key="YearsInCurrentRole")
            st.number_input("Years Since Last Promotion", 0, 20, key="YearsSinceLastPromotion")
            st.number_input("Years With Current Manager", 0, 20, key="YearsWithCurrManager")
            st.number_input("Training Times Last Year", 0, 10, key="TrainingTimesLastYear")
            st.selectbox("Performance Rating", [1, 2, 3, 4], key="PerformanceRating")

        with col_comp:
            st.subheader("Financial Package")
            st.number_input("Monthly Income", 1000, 100000, key="MonthlyIncome")
            st.number_input("Daily Rate", 0, 2000, key="DailyRate")
            st.number_input("Hourly Rate", 0, 100, key="HourlyRate")
            st.number_input("Monthly Rate", 0, 30000, key="MonthlyRate")
            st.slider("Percent Salary Hike", 0, 50, key="PercentSalaryHike")
            st.selectbox("Stock Option Level", [0, 1, 2, 3], key="StockOptionLevel")

    # --- TAB 3: SATISFACTION & ENGAGEMENT ---
    elif st.session_state.active_form == "Satisfaction":
        st.markdown('<div class="glass-card"><h2>😊 Satisfaction & Engagement</h2></div>', unsafe_allow_html=True)
        st.selectbox("Environment Satisfaction", [1, 2, 3, 4], key="EnvironmentSatisfaction")
        st.selectbox("Job Satisfaction", [1, 2, 3, 4], key="JobSatisfaction")
        st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], key="RelationshipSatisfaction")
        st.selectbox("Work Life Balance", [1, 2, 3, 4], key="WorkLifeBalance")
        st.selectbox("Job Involvement", [1, 2, 3, 4], key="JobInvolvement")

    st.write("")
    predict_button = st.button("🚀 Predict Attrition Risk", use_container_width=True)

# =====================================================
# RIGHT PANEL: AI ANALYTICS & VISUALIZATION DASHBOARD
# =====================================================
with right_panel:
    st.markdown('<div class="glass-card"><h2>🤖 AI Prediction Dashboard</h2></div>', unsafe_allow_html=True)

    if predict_button:
        # Pull latest modified properties straight from global session state map
        employee_data = {k: st.session_state[k] for k in defaults.keys()}
        input_df = pd.DataFrame([employee_data])

        # Feature Engineering Pipeline
        input_df["IncomePerJobLevel"] = input_df["MonthlyIncome"] / input_df["JobLevel"]
        input_df["ExperienceRatio"] = input_df["YearsAtCompany"] / (input_df["TotalWorkingYears"] + 1)
        input_df["PromotionGap"] = input_df["YearsAtCompany"] - input_df["YearsSinceLastPromotion"]
        input_df["ManagerTenureRatio"] = input_df["YearsWithCurrManager"] / (input_df["YearsAtCompany"] + 1)

        # Vector Transform Pipeline
        encoded = pd.get_dummies(input_df)
        encoded = encoded.reindex(columns=feature_names, fill_value=0)
        scaled_input = scaler.transform(encoded)

        # Model Inference Pipeline
        probability = model.predict_proba(scaled_input)[0][1]
        threshold = 0.30
        prediction = 1 if probability >= threshold else 0

        # Display Result Metric Block
        status_text = "⚠️ HIGH ATTRITION RISK" if prediction == 1 else "✅ LOW ATTRITION RISK"
        st.markdown(
            f"""
            <div class="result-card">
            <h1>{status_text}</h1>
            <h2>Probability</h2>
            <h1>{probability*100:.2f} %</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Plotly Risk Score Gauging
        st.subheader("📊 Attrition Risk Score")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={"text": "Risk %"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # SHAP Valuation Extraction
        st.divider()
        st.subheader("🔍 Explainable AI - Top Attrition Factors")

        try:
            shap_result = explainer(scaled_input)
            shap_values = shap_result.values[0]
        except:
            shap_values = explainer.shap_values(scaled_input)[0]

        shap_df = pd.DataFrame({"Feature": feature_names, "Impact": abs(shap_values)})
        shap_df = shap_df.sort_values("Impact", ascending=False).head(10)

        fig = px.bar(shap_df, x="Impact", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

        # Action Recommendations
        st.divider()
        st.subheader("🎯 HR Action Recommendation")

        actions = []
        for feature in shap_df["Feature"]:
            clean = feature.split("_")[0]
            if clean in recommendation_mapping:
                actions.append({
                    "Risk Factor": clean,
                    "HR Action": recommendation_mapping[clean]
                })

        if actions:
            st.dataframe(pd.DataFrame(actions), use_container_width=True, hide_index=True)
        else:
            st.success("No major intervention required.")

        # Financial Business Valuation Elements
        st.divider()
        st.subheader("💰 Business Impact Estimation")

        annual_salary = employee_data["MonthlyIncome"] * 12
        replacement_cost = annual_salary * 0.75
        expected_loss = probability * replacement_cost

        col1, col2, col3 = st.columns(3)
        col1.metric("Annual Salary", f"₹{annual_salary:,.0f}")
        col2.metric("Replacement Cost", f"₹{replacement_cost:,.0f}")
        col3.metric("Expected Loss", f"₹{expected_loss:,.0f}")
    else:
        st.info("Adjust the structural parameters on the left and trigger prediction to compute engine analytics.")