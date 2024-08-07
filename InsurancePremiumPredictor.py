import pickle

import pandas
import sklearn
import xgboost
import streamlit as st


st.header('Insurance-Premium-Price-Prediction', divider='rainbow')
# df=pd.read_csv("/Users/LENOVO/PycharmProjects/StreamLitTutorial/cars24-car-price-cleaned.csv")
# st.dataframe(df)
col3, col4= st.columns(2)

with col3:
    Diabetes = st.selectbox(
        "Does the customer have Diabetes ?",
        ("Yes", "No"))


with col4:
    BloodPressureProblems = st.selectbox(
        "Does the customer have Blood Pressure Problems ?",
        ("Yes", "No"))

col5,col6 = st.columns(2)
with col5:
    AnyTransplants = st.selectbox(
        "Did the customer have Any Transplants ?",
        ("Yes", "No"))

with col6:
    AnyChronicDiseases = st.selectbox(
        "Does the customer have AnyChronic Diseases ?",
        ("Yes", "No"))

col7,col8 = st.columns(2)

with col7:
    KnownAllergies = st.selectbox(
        "Does the customer have Known Allergies ?",
        ("Yes", "No"))

with col8:
    HistoryOfCancerInFamily = st.selectbox(
        "Does the customer have History Of Cancer In Family ?",
        ("Yes", "No"))


NumberOfMajorSurgeries = st.selectbox(
    "How many Number Of Major Surgeries did the customer have ?",
    (0,1,2,3))


Age = st.slider("Select Age ", 18, 66, step=1)



col1, col2 = st.columns(2)

with col1:
    h_input = st.slider("Select Height(in cm) ", 145, 188, step=1)
    Height= h_input/100
with col2:
    Weight =  st.slider("Select Weight(in Kg) ", 51, 132, step=1)


BMI = Weight/(Height*Height)

encode_dict= {"Diabetes": {"Yes": 1, "No": 0} ,
              "BloodPressureProblems": {"Yes": 1, "No": 0},
              "AnyTransplants": {"Yes": 1, "No": 0},
              "AnyChronicDiseases": {"Yes": 1, "No": 0},
              "KnownAllergies": {"Yes": 1, "No": 0},
              "HistoryOfCancerInFamily": {"Yes": 1, "No": 0}

}


def model_pred(Age, Diabetes,BloodPressureProblems, AnyChronicDiseases,AnyTransplants,
                      KnownAllergies, HistoryOfCancerInFamily,BMI, NumberOfMajorSurgeries ):
    with open("regression.pkl","rb") as handle :
        model=pickle.load(handle)
        return model.predict([[Age, Diabetes,BloodPressureProblems, AnyChronicDiseases,AnyTransplants,
                      KnownAllergies, HistoryOfCancerInFamily,BMI, NumberOfMajorSurgeries]])


if st.button("Predict"):
    Diabetes = encode_dict["Diabetes"][Diabetes]
    BloodPressureProblems=encode_dict["BloodPressureProblems"][BloodPressureProblems]
    AnyTransplants = encode_dict["AnyTransplants"][AnyTransplants]
    AnyChronicDiseases = encode_dict["AnyChronicDiseases"][AnyChronicDiseases]
    KnownAllergies = encode_dict["KnownAllergies"][KnownAllergies]
    HistoryOfCancerInFamily = encode_dict["HistoryOfCancerInFamily"][HistoryOfCancerInFamily]
    price= model_pred(Age, Diabetes,BloodPressureProblems, AnyChronicDiseases,AnyTransplants,
                      KnownAllergies, HistoryOfCancerInFamily,BMI, NumberOfMajorSurgeries)
    st.write("Estimated Price(in Thousand)",price)