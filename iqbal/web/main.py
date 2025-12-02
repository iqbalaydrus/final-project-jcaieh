from typing import Any

import streamlit as st
import requests

DISCORD_CHANNEL_NAME = st.secrets["DISCORD_CHANNEL_NAME"]
REST_API_BASE_URL = st.secrets["REST_API_BASE_URL"]
REST_API_KEY = st.secrets["REST_API_KEY"]


def api(path: str, data: Any) -> requests.Response:
    return requests.post(
        REST_API_BASE_URL + path,
        headers={
            "Authorization": "Bearer " + REST_API_KEY,
            "Content-Type": "application/json",
        },
        json=data,
    )


st.title("Final Project")
st.write(
    """Upload an image with one or multiple car, bus, or van in it. And we'll output the count of each vehicle type."""
)

for k, v in st.session_state.items():
    st.session_state[k] = v

if not st.session_state.get("token") or st.session_state.token != DISCORD_CHANNEL_NAME:
    st.text_input("Enter our discord channel name - needed for auth:", key="token")
    st.write(":red[Incorrect channel name]")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
