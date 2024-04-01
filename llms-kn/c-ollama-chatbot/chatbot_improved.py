import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from dotenv import load_dotenv

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}"),
    ]
)
llm = Ollama(model="phi")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Initialize Streamlit app
st.title("Phi-2 based Chatbot")

# Initialize conversation history list
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Create input text field for user messages
user_input = st.text_input("Write something below:")

# If user submits a message
if st.button("Send"):
    if user_input:
        # Reverse the user input
        reversed_input = chain.invoke({"question": user_input})

        # Add user message and reversed message to conversation history
        st.session_state.conversation_history.append(("User", user_input))
        st.session_state.conversation_history.append(("Bot", reversed_input))

# Display conversation history
st.header("Conversation History")

# Display conversation history in structured format
for role, message in st.session_state.conversation_history:
    if role == "User":
        st.text(f"User: {message}")
    elif role == "Bot":
        st.text(f"Bot: {message}")
