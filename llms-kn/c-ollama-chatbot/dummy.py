import streamlit as st


def reverse_string(text):
    return text[::-1]


# Initialize Streamlit app
st.title("Simple Chatbot")

# Initialize conversation history list
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Create input text field for user messages
user_input = st.text_input("Write something below:")

# If user submits a message
if st.button("Send"):
    if user_input:
        # Reverse the user input
        reversed_input = reverse_string(user_input)

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
