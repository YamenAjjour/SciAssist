import os
import streamlit as st
import requests
import json



def answer_question():
    url = "http://127.0.0.1:8585/chat" # No need for '?' at the end, requests handles it
    response = requests.get(url, params={"q": st.session_state.question})
    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    data = response.json()       # Parses JSON directly
    response = data.get("result")
    image_path = data.get("image_path")
    st.session_state.answer=response    # Use .get() for safer dictionary access
    if image_path:
        st.session_state.image_path= image_path

st.title("SciAssist")

if "image_path" not in st.session_state:
    st.session_state.image_path= None


with st.form('question_form'):
    st.text_input(label='Question:', value='Enter your Question',key="question")
    st.text_area(label="Answer", height=400, key="answer")
    image_placeholder = st.empty()
    if st.session_state.image_path:
        image_placeholder.image(st.session_state.image_path)
    st.form_submit_button('Ask', on_click=answer_question)