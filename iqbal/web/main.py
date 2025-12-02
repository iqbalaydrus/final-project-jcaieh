import streamlit as st

DISCORD_CHANNEL_NAME = st.secrets["DISCORD_CHANNEL_NAME"]
REST_API_BASE_URL = st.secrets["REST_API_BASE_URL"]
REST_API_KEY = st.secrets["REST_API_KEY"]

st.title("Final Project")
st.write(
    """Upload an image with one or multiple car, bus, or van in it. And we'll output the count of each vechicle type."""
)

for k, v in st.session_state.items():
    st.session_state[k] = v

if not st.session_state.get("token") or st.session_state.token != DISCORD_CHANNEL_NAME:
    st.text_input("Enter our discord channel name - needed for auth:", key="token")
    st.write(":red[Incorrect channel name]")
