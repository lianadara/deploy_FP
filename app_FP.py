import streamlit as st
import form_page
import predict
import ml_FP

navigation = st.sidebar.selectbox('🔗Select Page:', ('📃Form Page','📂File Upload Page','💬Chatbot Page'))
st.sidebar.header("by Team 2")

if navigation == '📃Form Page':
    form_page.run()
elif navigation == '📂File Upload Page':
    predict.run()
elif navigation == '💬Chatbot Page':
    ml_FP.run()
