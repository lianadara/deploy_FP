import streamlit as st
import streamlit_fp.predict_FP as predict_FP
import streamlit_fp.predict as predict

navigation = st.sidebar.selectbox('Select Page:', ('Form Page','File Upload Page'))

if navigation == 'Form Page':
    predict_FP.run()
elif navigation == 'File Upload Page':
    predict.run()
