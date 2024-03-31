from utils import get_openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain
import streamlit as st
import os

# get OPENAI key
os.environ["OPENAI_API_KEY"] = get_openai_key(
    "/home/ravi0531rp/Desktop/CODES/n-nlp/llms-kn/secret.json"
)

# Streamlit engine
st.title("Celebrity Search Results")
input_text = st.text_input("Search with Openai")

# memory
person_memory = ConversationBufferMemory(input_key='name', memory_key= 'chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key= 'chat_history')

# OpenAI LLM
llm = OpenAI(temperature=0.8)

first_input_prompt = PromptTemplate(
    input_variables=["name"], template="Tell me about the celebrity named {name}"
)

chain_1 = LLMChain(
    llm=llm, prompt=first_input_prompt, verbose=True, output_key="person", memory = person_memory
)

second_input_prompt = PromptTemplate(
    input_variables=["person"], template="When was {person} born"
)
chain_2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="dob", memory = dob_memory)

parent_chain = SequentialChain(
    chains=[chain_1, chain_2],
    input_variables=["name"],
    output_variables=["person", "dob"],
    verbose=True,
)

if input_text:
    pred = parent_chain({"name": input_text})
    st.write(pred)

    with st.expander('Person name'):
        st.info(person_memory.buffer)

 