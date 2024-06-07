import streamlit as st
import predict_FP
import predict

navigation = st.sidebar.selectbox('Select Page:', ('Form Page','File Upload Page'))

if navigation == 'Form Page':
    predict_FP.run()
elif navigation == 'File Upload Page':
    predict.run()
