import os
import streamlit as st
from setup_rag import *
import requests
import json










def init_ragchain():
    global chain
    path = os.path.dirname(os.path.realpath(__file__))

    path_index = f"{path}/../data/index"
    path_dataset = f"{path}/../data/acl-publication-info.74k.parquet"
    if os.path.exists("/bigwork/nhwpajjy/pre-trained-models"):
        path_model = "/bigwork/nhwpajjy/pre-trained-models/TinyLlama-1.1B-Chat-v1.0"
    elif os.path.exists("/mnt/home/yajjour/pre-trained-models"):
        path_model ="/mnt/home/yajjour/pre-trained-models/TinyLlama-1.1B-Chat-v1.0"
    else:
        path_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    if not os.path.exists(path_index):
        create_index( path_dataset=path_dataset, path_index=path_index, debug=False)


def answer_query(query: str) -> str:
    url = "http://127.0.0.1:8585/chat" # No need for '?' at the end, requests handles it
    try:
        response = requests.get(url, params={"q": query})
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()       # Parses JSON directly
        return data.get("answer")    # Use .get() for safer dictionary access
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None # Or raise a custom exception, or handle as appropriate
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Response text: {response.text}") # Good for debugging
        return None
    except KeyError:
        print(f"Response did not contain 'answer' key: {data}")
        return None

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
        response = st.write_stream(answer)
    st.session_state.messages.append({"role" : "assistant", "content": response})