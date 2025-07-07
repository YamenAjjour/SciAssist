import os
import streamlit as st
import requests
import json



def answer_query(query: str) -> str:
    url = "http://127.0.0.1:8585/chat" # No need for '?' at the end, requests handles it
    response = requests.get(url, params={"q": query})
    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    data = response.json()       # Parses JSON directly
    return data.get("answer")    # Use .get() for safer dictionary access

st.title("SciAssist")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content" : prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = answer_query(prompt)
        response = st.write(answer)
    st.session_state.messages.append({"role" : "assistant", "content": response})