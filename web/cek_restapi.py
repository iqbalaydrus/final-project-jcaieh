import tomllib
import streamlit as st



REST_API_KEY = st.secrets["REST_API_KEY"]
st.write(f"rest api key : {REST_API_KEY}" )