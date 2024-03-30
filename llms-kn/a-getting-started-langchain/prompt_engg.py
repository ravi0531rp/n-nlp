from utils import get_openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import os

# get OPENAI key
os.environ["OPENAI_API_KEY"] = get_openai_key(
    "/home/ravi0531rp/Desktop/CODES/n-nlp/llms-kn/secret.json"
)

# Streamlit engine
st.title("Celebrity Search Results")
input_text = st.text_input("Search with Openai")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=["name"], template="Tell me about the celebrity named {name}"
)

# OpenAI LLM
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True)

if input_text:
    pred = chain.run(input_text)
    st.write(pred)
