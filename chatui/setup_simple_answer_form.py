import os
import streamlit as st
import requests
import json



def answer_question():
    url = "http://127.0.0.1:8585/chat" # No need for '?' at the end, requests handles it
    response = requests.get(url, params={"q": st.session_state.question})
    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    data = response.json()       # Parses JSON directly
    st.session_state.answer=data.get("answer")    # Use .get() for safer dictionary access

st.title("SciAssist")





with st.form('question_form'):
    st.text_input(label='Question:', value='Enter your Question',key="question")
    st.text_area(label="answer", height=400, key="answer")
    st.form_submit_button('Ask', on_click=answer_question)