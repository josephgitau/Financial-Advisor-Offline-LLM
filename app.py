import streamlit as st
from financial_advisor import query_ollama
import os

# Set the page configuration
st.set_page_config(
    page_title="Financial Advisor Chatbot", 
    page_icon="💰",
    layout="centered"
)

# Title and description
st.title("💰 Financial Advisor Chatbot")
st.markdown("""
Welcome to the Financial Advisor Chatbot! Ask any financial questions you have, and get practical advice.
""")

# sidebar
with st.sidebar:
    # ollama image and link
    st.image("https://ollama.com/public/ollama.png", width=50)
    st.subheader("Setup Check")
 
    # Quick self-check for Ollama API availability
    ollama_running = False
    try:
        import requests
        r = requests.get("http://localhost:11434")
        ollama_running = r.status_code == 200
    except:
        pass

    if ollama_running:
        st.success("Ollama API is running!")
    else:
        st.error("Ollama API is not reachable. Please ensure it is running locally.")

    st.markdown("""
    **About this App**
    1. Install and run the Ollama API locally.
    2. Ask financial questions and get advice.
    3. Always consult a licensed professional for personalized advice.
                
    [Ollama Website](https://ollama.com)
    """
    )

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state['chat_history'] = []

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm FinWise, your financial advisor bot. How can I assist you today?"},
    ]

# Display Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything about personal finance..."):
    # add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_ollama(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})