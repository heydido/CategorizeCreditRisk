import streamlit as st
import numpy as np
from src.CategorizeCreditRisk.components.prediction import CustomData
from src.CategorizeCreditRisk.pipeline.predict import PredictionPipeline


# Page layout
st.set_page_config(page_title="Categorize Credit Risk", page_icon="✅", layout="wide")
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# User interface
st.header("Categorize Credit Risk!")
st.write("This is a simple tool to categorize credit risk based on the input data.")
# st.image("", use_column_width=True)


# User inputs
st.subheader('User Inputs:')

maritalstatus = st.radio("maritalstatus:", ("Married", "Single"))
gender = st.radio("gender:", ("M", "F"))

education = st.radio(
    "education:", ("SSC", "OTHERS", "12TH", "UNDER GRADUATE", "GRADUATE", "PROFESSIONAL", "POST-GRADUATE")
)
encode_education = {
    "SSC": 1,
    "OTHERS": 1,
    "12TH": 2,
    "UNDER GRADUATE": 3,
    "GRADUATE": 3,
    "PROFESSIONAL": 3,
    "POST-GRADUATE": 4
}
education = encode_education[education]

last_prod_enq2 = st.radio("last_prod_enq2:", ("PL", "ConsumerLoan", "AL", "CC", "others", "HL"))
first_prod_enq2 = st.radio("first_prod_enq2:", ("PL", "ConsumerLoan", "AL", "CC", "others", "HL"))
pct_tl_open_l6m = st.select_slider("pct_tl_open_l6m:", options=np.arange(0.0, 1.01, 0.01))
pct_tl_closed_l6m = st.select_slider("pct_tl_closed_l6m:", options=np.arange(0.0, 1.01, 0.01))
tot_tl_closed_l12m = st.select_slider("tot_tl_closed_l12m:", options=np.arange(0, 51, 1))
pct_tl_closed_l12m = st.select_slider("pct_tl_closed_l12m:", options=np.arange(0.0, 1.01, 0.01))
tot_missed_pmnt = st.select_slider("tot_missed_pmnt:", options=np.arange(0, 101, 1))
cc_tl = st.select_slider("cc_tl:", options=np.arange(0, 21, 1))
home_tl = st.select_slider("home_tl:", options=np.arange(0, 21, 1))
pl_tl = st.select_slider("pl_tl:", options=np.arange(0, 21, 1))
secured_tl = st.select_slider("secured_tl:", options=np.arange(0, 21, 1))
unsecured_tl = st.select_slider("unsecured_tl:", options=np.arange(0, 21, 1))
other_tl = st.select_slider("other_tl:", options=np.arange(0, 21, 1))
age_oldest_tl = st.number_input("age_oldest_tl:", 0, 600)
age_newest_tl = st.number_input("age_newest_tl:", 0, 600)
time_since_recent_payment = st.number_input("time_since_recent_payment:", 0, 6000)
max_recent_level_of_deliq = st.number_input("max_recent_level_of_deliq:", 0, 1000)
num_deliq_6_12mts = st.select_slider("num_deliq_6_12mts:", options=np.arange(0, 145, 1))
num_times_60p_dpd = st.select_slider("num_times_60p_dpd:", options=np.arange(0, 51, 1))
num_std_12mts = st.select_slider("num_std_12mts:", options=np.arange(0, 151, 1))
num_sub = st.select_slider("num_sub:", options=np.arange(0, 51, 1))
num_sub_6mts = st.select_slider("num_sub_6mts:", options=np.arange(0, 11, 1))
num_sub_12mts = st.select_slider("num_sub_12mts:", options=np.arange(0, 21, 1))
num_dbt = st.select_slider("num_dbt:", options=np.arange(0, 51, 1))
num_dbt_12mts = st.select_slider("num_dbt_12mts:", options=np.arange(0, 31, 1))
num_lss = st.select_slider("num_lss:", options=np.arange(0, 101, 1))
recent_level_of_deliq = st.number_input("recent_level_of_deliq:", 0, 1000)
cc_enq_l12m = st.select_slider("cc_enq_l12m:", options=np.arange(0, 31, 1))
pl_enq_l12m = st.select_slider("pl_enq_l12m:", options=np.arange(0, 31, 1))
time_since_recent_enq = st.number_input("time_since_recent_enq:", 0, 5000)
enq_l3m = st.select_slider("enq_l3m:", options=np.arange(0, 31, 1))
netmonthlyincome = st.number_input("netmonthlyincome:", 0, 1000000, 0)
time_with_curr_empr = st.number_input("time_with_curr_empr:", 0, 1000)
pct_pl_enq_l6m_of_ever = st.select_slider("pct_pl_enq_l6m_of_ever:", options=np.arange(0.0, 1.01, 0.01))
pct_cc_enq_l6m_of_ever = st.select_slider("pct_cc_enq_l6m_of_ever:", options=np.arange(0.0, 1.01, 0.01))
cc_flag = st.radio("cc_flag:", (0, 1))
pl_flag = st.radio("pl_flag:", (0, 1))
hl_flag = st.radio("hl_flag:", (0, 1))
gl_flag = st.radio("gl_flag:", (0, 1))


custom_data = CustomData(
    maritalstatus=maritalstatus,
    gender=gender,
    last_prod_enq2=last_prod_enq2,
    first_prod_enq2=first_prod_enq2,
    pct_tl_open_l6m=pct_tl_open_l6m,
    pct_tl_closed_l6m=pct_tl_closed_l6m,
    tot_tl_closed_l12m=tot_tl_closed_l12m,
    pct_tl_closed_l12m=pct_tl_closed_l12m,
    tot_missed_pmnt=tot_missed_pmnt,
    cc_tl=cc_tl,
    home_tl=home_tl,
    pl_tl=pl_tl,
    secured_tl=secured_tl,
    unsecured_tl=unsecured_tl,
    other_tl=other_tl,
    age_oldest_tl=age_oldest_tl,
    age_newest_tl=age_newest_tl,
    time_since_recent_payment=time_since_recent_payment,
    max_recent_level_of_deliq=max_recent_level_of_deliq,
    num_deliq_6_12mts=num_deliq_6_12mts,
    num_times_60p_dpd=num_times_60p_dpd,
    num_std_12mts=num_std_12mts,
    num_sub=num_sub,
    num_sub_6mts=num_sub_6mts,
    num_sub_12mts=num_sub_12mts,
    num_dbt=num_dbt,
    num_dbt_12mts=num_dbt_12mts,
    num_lss=num_lss,
    recent_level_of_deliq=recent_level_of_deliq,
    cc_enq_l12m=cc_enq_l12m,
    pl_enq_l12m=pl_enq_l12m,
    time_since_recent_enq=time_since_recent_enq,
    enq_l3m=enq_l3m,
    netmonthlyincome=netmonthlyincome,
    time_with_curr_empr=time_with_curr_empr,
    cc_flag=cc_flag,
    pl_flag=pl_flag,
    pct_pl_enq_l6m_of_ever=pct_pl_enq_l6m_of_ever,
    pct_cc_enq_l6m_of_ever=pct_cc_enq_l6m_of_ever,
    hl_flag=hl_flag,
    gl_flag=gl_flag,
    education=education
)

# Prediction
if st.button('Predict'):
    # TODO: Call the prediction pipeline separately and use it here for each prediction
    predict_pipeline = PredictionPipeline()
    risk_category = predict_pipeline.predict(custom_data=custom_data)

    get_risk_category = {
        1: "P1 - Least Risk",
        2: "P2 - Some Risk",
        3: "P3 - More Risk",
        4: "P4 - Highest Risk"
    }
    risk_category = get_risk_category[risk_category]

    st.write('🚀The predicted risk category is: ', risk_category)
