import streamlit as st
import pandas as pd
import numpy as np
from pickle import load, loads
from pathlib import Path
from assem.class_Preparing import Preparing
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
#st.title(r"$\Huge🇰🇿$")
st.title("Credit Scoring in Taiwan 2005")

my_data = st.sidebar.selectbox("Choose your dataset",
                               ["UCI_Credit_Card.csv"]
                               )
#st.sidebar.selectbox('Shold', min_value=0, max_value=1, step=0.01, format=None)
shold = st.sidebar.selectbox('shold',[0.6,0.65,0.70,0.75,0.8, 0.85])
df = pd.read_csv(BASE_DIR / my_data)

limit_ball = st.number_input('LIMIT_BALL', min_value=0, max_value=1000000, step=10000, format=None)        

with st.expander("Payment status"): # свертывающийся блок
        pd1 = st.selectbox("Month 1", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd2 = st.selectbox("Month 2", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd3 = st.selectbox("Month 3", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd4 = st.selectbox("Month 4", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd5 = st.selectbox("Month 5", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
        pd6 = st.selectbox("Month 6", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)

with st.expander("Bill amounts"):
        col1, col2 = st.columns(2)  # 2 колонки в блоке по 3 окошка
        with col1:
            b1 = st.number_input("Bill month 1", min_value=0, max_value=1000000, value=2000, step=1000)
            b2 = st.number_input("Bill month 2", min_value=0, max_value=1000000, value=2000, step=1000)
            b3 = st.number_input("Bill month 3", min_value=0, max_value=1000000, value=2000, step=1000)
        with col2:
            b4 = st.number_input("Bill month 4", min_value=0, max_value=1000000, value=2000, step=1000)
            b5 = st.number_input("Bill month 5", min_value=0, max_value=1000000, value=2000, step=1000)
            b6 = st.number_input("Bill month 6", min_value=0, max_value=1000000, value=2000, step=1000)

with st.expander("Payment amounts"):
        col3, col4 = st.columns(2)
        with col3:
            p1 = st.number_input("Payment month 1", min_value=0, max_value=1000000, value=2000, step=1000)
            p2 = st.number_input("Payment month 2", min_value=0, max_value=1000000, value=2000, step=1000)
            p3 = st.number_input("Payment month 3", min_value=0, max_value=1000000, value=2000, step=1000)
        with col4:
            p4 = st.number_input("Payment month 4", min_value=0, max_value=1000000, value=2000, step=1000)
            p5 = st.number_input("Payment month 5", min_value=0, max_value=1000000, value=2000, step=1000)
            p6 = st.number_input("Payment month 6", min_value=0, max_value=1000000, value=2000, step=1000)

st.success(f"credit = {limit_ball}") # вывод указанной суммы кредита
st.title("Predictions")

exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

if st.button("My predictions", disabled = False ): # кнопка предсказаний, кнопка активна

    with open(BASE_DIR / "dtree.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        dtree = load(f)                  # загруженная модель dtree
        pred = dtree.predict([exp])[0]
        pred_proba = dtree.predict_proba([exp])[0]
        #if pred == 0:
        if pred_proba[0]>shold:
            st.write("Decision Tree")
            st.success(f"{pred} -- no default")
            st.success(round(pred_proba[pred],4))
        #elif pred != 0:
        else:
            st.write("Decision Tree")
            st.error(f"{pred} -- default")
            st.error(round(pred_proba[pred],4)) 
    
    with open(BASE_DIR / "rfc.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        rfc = load(f)                  # загруженная модель rfc
        pred = rfc.predict([exp])[0]
        pred_proba = rfc.predict_proba([exp])[0]
        #if pred == 0:
        if pred_proba[0]>shold:
            st.write("Random Forest")
            st.success(f"{pred} -- no default")
            st.success(round(pred_proba[pred],4))
        #elif pred != 0:
        else:
            st.write("Random Forest")
            st.error(f"{pred} -- default")
            st.error(round(pred_proba[pred],4)) 

    with open(BASE_DIR / "boost_gr.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        boost_gr = load(f)                  # загруженная модель boost_gr
        pred = boost_gr.predict([exp])[0]
        pred_proba = boost_gr.predict_proba([exp])[0]
        #if pred == 0:
        if pred_proba[0]>shold:
            st.write("Gradient Boosting")
            st.success(f"{pred} -- no default")
            st.success(round(pred_proba[pred],4))
        #elif pred != 0:
        else:    
            st.write("Gradient Boosting")
            st.error(f"{pred} -- default")
            st.error(round(pred_proba[pred],4)) 

    model = keras.models.load_model(BASE_DIR / "my_model.h5", compile=False)
    columns_st = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                  'PAY_AMT1',  'PAY_AMT2',  'PAY_AMT3',  'PAY_AMT4',  'PAY_AMT5',  'PAY_AMT6']
    columns_old = ['LIMIT_BAL','PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','BILL_AMT1',
                   'BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                   'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    exp = pd.DataFrame( [np.array(exp)], columns = columns_old )
    with open(BASE_DIR / "scaler.pkl", "rb") as f:# rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        scaler = load(f)
    exp_st = scaler.transform( exp.loc[:, columns_st] )
    exp.loc[:, columns_st] = exp_st     
    exp = np.array(exp)
    pred_proba_nn = model.predict([exp])[0][0]
    
    if pred_proba_nn<1-shold:
        st.write("Neural Network")
        st.success(f"{0} -- no default")
        st.success(round(1-pred_proba_nn,4))
    #elif pred != 0:
    else:    
        st.write("Neural Netz")
        st.error(f"{1} -- default")
        st.error(round(1-pred_proba_nn,4))

    
    
