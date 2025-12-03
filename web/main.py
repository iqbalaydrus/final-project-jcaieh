from typing import Any
from uuid import uuid4

import streamlit as st
import requests
import magic
from pydantic import ValidationError
from streamlit.runtime.uploaded_file_manager import UploadedFile

import schema

DISCORD_CHANNEL_NAME = st.secrets["DISCORD_CHANNEL_NAME"]
REST_API_BASE_URL = st.secrets["REST_API_BASE_URL"]
REST_API_KEY = st.secrets["REST_API_KEY"]


def chat_api(data: Any) -> schema.ChatResponse:
    resp = requests.post(
        REST_API_BASE_URL + "/chat",
        headers={
            "Authorization": "Bearer " + REST_API_KEY,
            "Content-Type": "application/json",
        },
        json=data,
    )
    resp.raise_for_status()
    return schema.ChatResponse.model_validate(resp.json())

def upload_api(file: UploadedFile):
    resp = requests.post(
        REST_API_BASE_URL + "/upload",
        files={"file": (file.name, file, "application/pdf")},
        data={"session_id": st.session_state.session_id},
    )
    resp.raise_for_status()

def main_program():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    uploaded_file = st.file_uploader(
        "Upload your CV",
        type="pdf",
        accept_multiple_files=False,
    )
    prompt = st.chat_input("Ask your question here")
    if uploaded_file and ("cv_uploaded" not in st.session_state or st.session_state.cv_uploaded != uploaded_file.name):
        mime_type = magic.from_buffer(uploaded_file.read(1024), mime=True)
        if mime_type != "application/pdf":
            st.write(":red[Please upload a PDF file]")
            return
        uploaded_file.seek(0)
        try:
            upload_api(uploaded_file)
        except requests.HTTPError as e:
            st.write(f"Upload Error: :red[{str(e)}]")
            return
        st.session_state.cv_uploaded = uploaded_file.name
    if prompt:
        history = st.session_state.messages.copy()
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            resp = chat_api(
                {
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                    "history": history,
                },
            )
        except (ValidationError, requests.HTTPError) as e:
            st.write(f"API Error: :red[{str(e)}]")
            return
        with st.chat_message("ai"):
            st.markdown(resp.message.content)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(resp.model_dump(mode="json"))



st.title("Indonesian Job Agent")
st.write(
    """An agent to help you find your dream job in Indonesia. Built with [Streamlit](https://streamlit.io) and [OpenAI](https://openai.com)."""
)

for k, v in st.session_state.items():
    st.session_state[k] = v
if not st.session_state.get("session_id"):
    st.session_state.session_id = str(uuid4())

if not st.session_state.get("token") or st.session_state.token != DISCORD_CHANNEL_NAME:
    st.text_input("Enter our discord channel name - needed for auth:", key="token")
    st.write(":red[Incorrect channel name]")
else:
    main_program()
