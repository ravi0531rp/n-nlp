from utils import get_openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import streamlit as st
import os

# get OPENAI key
os.environ["OPENAI_API_KEY"] = get_openai_key(
    "/home/ravi0531rp/Desktop/CODES/n-nlp/llms-kn/secret.json"
)

# Streamlit engine
st.title("Celebrity Search Results")
input_text = st.text_input("Search with Openai")


# OpenAI LLM
llm = OpenAI(temperature=0.8)

first_input_prompt = PromptTemplate(
    input_variables=["name"], template="Tell me about the celebrity named {name}"
)
chain_1 = LLMChain(
    llm=llm, prompt=first_input_prompt, verbose=True, output_key="person"
)

second_input_prompt = PromptTemplate(
    input_variables=["person"], template="When was {person} born"
)
chain_2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="dob")

parent_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)

if input_text:
    pred = parent_chain.run(input_text)
    st.write(pred)
