"""
Voice-Enabled Financial Advisor Chatbot
Extends the text-based chatbot with voice input/output capabilities
"""
import streamlit as st
from financial_advisor import query_ollama, query_ollama_streaming
from audio_processor import get_audio_processor
import time
import os
import tempfile
from pathlib import Path

# Set the page configuration
st.set_page_config(
    page_title="Voice Financial Advisor", 
    page_icon="🎤",
    layout="centered"
)

# Initialize audio processor (lazy-loaded)
audio_processor = get_audio_processor()

# Title and description
st.title("🎤 Voice Financial Advisor Chatbot")
st.markdown("""
Welcome to the Voice-Enabled Financial Advisor! Ask questions using your voice or text.
**Fully offline voice processing** - your audio never leaves your device!
""")

# Sidebar
with st.sidebar:
    # Ollama status
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
        st.success("✅ Ollama API is running!")
    else:
        st.error("❌ Ollama API is not reachable. Please ensure it is running locally.")

    st.divider()
    
    # Voice settings
    st.subheader("🎙️ Voice Settings")
    
    enable_voice = st.checkbox("Enable Voice Input", value=True)
    enable_tts = st.checkbox("Enable Voice Output", value=True)
    autoplay_audio = st.checkbox("Auto-play responses", value=True)
    
    st.divider()
    
    # Model settings
    st.subheader("⚙️ Model Settings")
    
    whisper_model = st.selectbox(
        "Whisper Model Size",
        options=["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower"
    )
    audio_processor.whisper_model_size = whisper_model
    
    tts_backend = st.selectbox(
        "TTS Backend",
        options=["piper", "gtts"],
        index=0,
        help="Piper is faster and offline, gTTS requires internet"
    )
    audio_processor.tts_backend = tts_backend
    
    if tts_backend == "piper":
        st.info("💡 Piper models are downloaded automatically on first use to ~/.local/share/piper-tts/voices/")
        
        # Allow custom model path
        custom_model = st.text_input(
            "Custom Piper Model Path (optional)",
            placeholder="/path/to/model.onnx",
            help="Leave empty to use default en_US-lessac-medium"
        )
        if custom_model and os.path.exists(custom_model):
            audio_processor.piper_model_path = custom_model
    
    st.divider()
    
    # Performance metrics
    st.subheader("📊 Performance")
    if 'metrics' in st.session_state and st.session_state.metrics:
        metrics = st.session_state.metrics
        st.metric("Last STT Time", f"{metrics.get('stt_time', 0):.2f}s")
        st.metric("Last LLM Time", f"{metrics.get('llm_time', 0):.2f}s")
        st.metric("Last TTS Time", f"{metrics.get('tts_time', 0):.2f}s")
        st.metric("Total E2E", f"{metrics.get('total_time', 0):.2f}s")

    st.divider()
    
    st.markdown("""
    **About this App**
    1. Install and run Ollama locally
    2. Click the microphone to record your question
    3. Get AI-powered financial advice
    4. Hear the response (optional)
    
    **Voice Pipeline:**
    - STT: faster-whisper (offline)
    - LLM: Ollama (streaming)
    - TTS: Piper-TTS (offline, preferred) or gTTS (online fallback)
    
    **Install Piper:**
    ```bash
    pip install piper-tts
    python -m piper.download_voices en_US-lessac-medium
    ```
                
    [Ollama Website](https://ollama.com) | [Piper TTS](https://github.com/rhasspy/piper)
    """
    )

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm FinWise, your voice-enabled financial advisor. How can I help you today?"}
        ]
        st.session_state.metrics = {}
        st.rerun()

# Initialize chat and metrics
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm FinWise, your voice-enabled financial advisor. How can I help you today?"},
    ]

if "metrics" not in st.session_state:
    st.session_state.metrics = {}

# Display Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display audio player if message has audio
        if "audio_path" in message and message["audio_path"] and os.path.exists(message["audio_path"]):
            st.audio(message["audio_path"], format="audio/wav")

# Voice input section
if enable_voice:
    st.divider()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🎤 Voice Input")
        audio_input = st.audio_input("Record your question")
    
    with col2:
        st.write("")
        st.write("")
        process_audio = st.button("🎯 Process Audio", disabled=audio_input is None)
    
    # Process voice input
    if process_audio and audio_input is not None:
        total_start = time.time()
        
        # Read audio bytes
        audio_bytes = audio_input.read()
        
        with st.spinner("🎧 Transcribing your question..."):
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            # Transcribe
            transcribed_text, stt_time = audio_processor.transcribe(tmp_path)
            os.unlink(tmp_path)
        
        if transcribed_text:
            st.success(f"✅ Heard: \"{transcribed_text}\" (took {stt_time:.2f}s)")
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            with st.chat_message("user"):
                st.markdown(transcribed_text)
            
            # Generate response with streaming
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    llm_start = time.time()
                    
                    # Collect streamed response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for token in query_ollama_streaming(transcribed_text):
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    llm_time = time.time() - llm_start
                
                # Generate audio response if enabled
                audio_path = None
                tts_time = 0
                
                if enable_tts and full_response:
                    with st.spinner("🔊 Generating audio response..."):
                        audio_path, tts_time = audio_processor.synthesize_speech(full_response)
                    
                    if audio_path and os.path.exists(audio_path):
                        st.audio(audio_path, format="audio/wav", autoplay=autoplay_audio)
            
            # Save response with audio path
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "audio_path": audio_path
            })
            
            # Update metrics
            total_time = time.time() - total_start
            st.session_state.metrics = {
                'stt_time': stt_time,
                'llm_time': llm_time,
                'tts_time': tts_time,
                'total_time': total_time
            }
            
            st.rerun()
        else:
            st.error("❌ Could not transcribe audio. Please try again.")

# Text input (always available)
st.divider()
st.subheader("⌨️ Text Input")

if prompt := st.chat_input("Or type your question here..."):
    total_start = time.time()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response with streaming
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            llm_start = time.time()
            
            # Collect streamed response
            response_placeholder = st.empty()
            full_response = ""
            
            for token in query_ollama_streaming(prompt):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            llm_time = time.time() - llm_start
        
        # Generate audio response if enabled
        audio_path = None
        tts_time = 0
        
        if enable_tts and full_response:
            with st.spinner("🔊 Generating audio response..."):
                audio_path, tts_time = audio_processor.synthesize_speech(full_response)
            
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path, format="audio/wav", autoplay=autoplay_audio)

    # Save response with audio path
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "audio_path": audio_path
    })
    
    # Update metrics
    total_time = time.time() - total_start
    st.session_state.metrics = {
        'stt_time': 0,  # No STT for text input
        'llm_time': llm_time,
        'tts_time': tts_time,
        'total_time': total_time
    }

# Footer
st.divider()
st.caption("⚠️ Disclaimer: This is an AI assistant. Always consult a licensed financial professional for personalized advice.")
