import streamlit as st
from financial_advisor import query_ollama
import os
import speech_recognition as sr
from gtts import gTTS
import pygame
import io
import tempfile

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

    # Speech settings
    st.subheader("Speech Settings")
    voice_output = st.checkbox("Enable Voice Output", value=True)
    voice_speed = st.slider("Voice Speed", 0.5, 2.0, 1.0, 0.1)
    
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

# Function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        audio = r.listen(source)
        st.info("Processing speech...")
    
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio.")
        return None
    except sr.RequestError:
        st.error("Sorry, there was an error with the speech recognition service.")
        return None

# Function to convert text to speech
def text_to_speech(text, speed=1.0):
    tts = gTTS(text=text, lang='en')
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_filename = fp.name
        
    tts.save(temp_filename)
    
    # Play the audio
    pygame.mixer.init()
    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()
    
    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Clean up
    pygame.mixer.quit()
    os.unlink(temp_filename)

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm FinWise, your financial advisor bot. How can I assist you today?"},
    ]

# Display Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Speech input button
if st.button("🎤 Speak"):
    user_input = speech_to_text()
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_ollama(user_input)
            st.markdown(response)
            
            # Voice output if enabled
            if voice_output:
                text_to_speech(response, voice_speed)

        st.session_state.messages.append({"role": "assistant", "content": response})

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
        
        # Voice output if enabled
        if voice_output:
            text_to_speech(response, voice_speed)

    st.session_state.messages.append({"role": "assistant", "content": response})