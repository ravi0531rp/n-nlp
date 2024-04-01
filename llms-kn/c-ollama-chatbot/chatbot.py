### ollama run phi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}"),
    ]
)
## streamlit framework

st.title("Phi based Chatbot...")
input_text = st.text_input("Search the topic u want")

# ollama LLAma2 LLm
llm = Ollama(model="phi")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
