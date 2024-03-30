from utils import get_openai_key
from langchain.llms import OpenAI
import streamlit as st
import os


os.environ["OPENAI_API_KEY"] = get_openai_key(
    "/home/ravi0531rp/Desktop/CODES/n-nlp/llms-kn/secret.json"
)
st.title("Langchain Demo with OpenAI")

input_text = st.text_input("Search with Openai")


llm = OpenAI(temperature=0.8)

if input_text:
    pred = llm(input_text)
    st.write(pred)
