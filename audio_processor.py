"""
Audio Processing Module for Voice-Enabled Financial Advisor
Implements lazy-loaded models with thread safety for STT, TTS, and VAD
"""
import os
import time
import threading
import tempfile
import wave
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles speech-to-text, text-to-speech, and voice activity detection.
    Models are lazy-loaded and cached in memory after first use.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self):
        self._stt_model = None
        self._tts_model = None
        self._vad_model = None
        self._lock = threading.Lock()
        
        # Model configurations
        self.stt_backend = "faster-whisper"  # Options: "faster-whisper", "vosk"
        self.whisper_model_size = "base"  # Options: tiny, base, small, medium, large
        self.tts_backend = "piper"  # Options: "piper", "gtts"
        self.piper_model_path = None  # Will be set to downloaded model path
        
    def _load_stt_model(self):
        """Lazy load STT model (faster-whisper or vosk)"""
        if self._stt_model is None:
            with self._lock:
                if self._stt_model is None:  # Double-check locking
                    logger.info(f"Loading {self.stt_backend} STT model...")
                    start_time = time.time()
                    
                    if self.stt_backend == "faster-whisper":
                        from faster_whisper import WhisperModel
                        self._stt_model = WhisperModel(
                            self.whisper_model_size,
                            device="cpu",  # Use "cuda" if GPU available
                            compute_type="int8"  # Optimized for CPU
                        )
                    elif self.stt_backend == "vosk":
                        from vosk import Model as VoskModel
                        model_path = "model"  # Path to vosk model directory
                        if not os.path.exists(model_path):
                            raise FileNotFoundError(
                                f"Vosk model not found at {model_path}. "
                                "Download from https://alphacephei.com/vosk/models"
                            )
                        self._stt_model = VoskModel(model_path)
                    
                    load_time = time.time() - start_time
                    logger.info(f"STT model loaded in {load_time:.2f}s")
        
        return self._stt_model
    
    def _load_tts_model(self):
        """Lazy load TTS model (piper-tts)"""
        if self._tts_model is None:
            with self._lock:
                if self._tts_model is None:
                    logger.info("Loading Piper TTS model...")
                    start_time = time.time()
                    
                    try:
                        # Try to import piper
                        from piper import PiperVoice
                        
                        # Set up model directories
                        model_dir = Path.home() / ".local" / "share" / "piper-tts" / "voices"
                        model_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Current directory (where the script is running)
                        current_dir = Path(__file__).parent
                        
                        # Look for existing .onnx model
                        if self.piper_model_path and os.path.exists(self.piper_model_path):
                            model_path = self.piper_model_path
                        else:
                            # First, check current directory for model files
                            current_dir_models = list(current_dir.glob("*.onnx"))
                            # Filter out non-piper models
                            current_dir_models = [m for m in current_dir_models if 'lessac' in m.name or 'libritts' in m.name or 'amy' in m.name]
                            
                            if current_dir_models:
                                model_path = str(current_dir_models[0])
                                logger.info(f"Found Piper model in current directory: {model_path}")
                            else:
                                # Try to find any downloaded model in standard location
                                onnx_models = list(model_dir.glob("**/*.onnx"))
                                if onnx_models:
                                    model_path = str(onnx_models[0])
                                    logger.info(f"Found Piper model: {model_path}")
                                else:
                                    # Download default model using piper's download function
                                    logger.info("No Piper model found. Downloading en_US-lessac-medium...")
                                    import sys
                                    import subprocess
                                    
                                    # Use sys.executable to ensure we use the correct Python interpreter
                                    result = subprocess.run(
                                        [sys.executable, '-m', 'piper.download', '--model', 'en_US-lessac-medium', '--output-dir', str(current_dir)],
                                        capture_output=True,
                                        timeout=120
                                    )
                                    
                                    if result.returncode != 0:
                                        raise Exception(f"Failed to download model: {result.stderr.decode()}")
                                    
                                    # Find the downloaded model in current directory
                                    current_dir_models = list(current_dir.glob("*.onnx"))
                                    current_dir_models = [m for m in current_dir_models if 'lessac' in m.name]
                                    
                                    if not current_dir_models:
                                        raise FileNotFoundError("Model download succeeded but .onnx file not found")
                                    model_path = str(current_dir_models[0])
                        
                        # Load the voice model
                        voice = PiperVoice.load(model_path)
                        
                        self._tts_model = {
                            'engine': 'piper',
                            'voice': voice,
                            'model_path': model_path
                        }
                        
                    except Exception as e:
                        # Fallback to gTTS if piper not available
                        logger.warning(f"Piper-TTS not available: {e}, using gTTS as fallback")
                        try:
                            from gtts import gTTS
                            self._tts_model = {
                                'engine': 'gtts'
                            }
                        except ImportError:
                            logger.error("Neither Piper nor gTTS available!")
                            self._tts_model = {'engine': 'none'}
                    
                    load_time = time.time() - start_time
                    logger.info(f"TTS model loaded in {load_time:.2f}s")
        
        return self._tts_model
    
    def _load_vad_model(self):
        """Lazy load VAD model (silero-vad)"""
        if self._vad_model is None:
            with self._lock:
                if self._vad_model is None:
                    logger.info("Loading Silero VAD model...")
                    start_time = time.time()
                    
                    try:
                        import torch
                        model, utils = torch.hub.load(
                            repo_or_dir='snakers4/silero-vad',
                            model='silero_vad',
                            force_reload=False,
                            onnx=False
                        )
                        self._vad_model = {
                            'model': model,
                            'utils': utils
                        }
                    except Exception as e:
                        logger.warning(f"Could not load Silero VAD: {e}")
                        self._vad_model = None
                    
                    load_time = time.time() - start_time
                    logger.info(f"VAD model loaded in {load_time:.2f}s")
        
        return self._vad_model
    
    def transcribe(self, audio_data: Union[bytes, str], sample_rate: int = 16000) -> Tuple[str, float]:
        """
        Transcribe audio to text using faster-whisper or vosk.
        
        Args:
            audio_data: Audio bytes or path to audio file
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (transcribed_text, processing_time_seconds)
        """
        start_time = time.time()
        model = self._load_stt_model()
        
        try:
            if self.stt_backend == "faster-whisper":
                # Handle bytes or file path
                if isinstance(audio_data, bytes):
                    # Save bytes to temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        
                        # Convert bytes to WAV
                        with wave.open(tmp_path, 'wb') as wav_file:
                            wav_file.setnchannels(1)  # Mono
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes(audio_data)
                    
                    audio_path = tmp_path
                else:
                    audio_path = audio_data
                
                # Transcribe with faster-whisper
                segments, info = model.transcribe(
                    audio_path,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Combine segments
                text = " ".join([segment.text for segment in segments]).strip()
                
                # Clean up temp file
                if isinstance(audio_data, bytes):
                    os.unlink(tmp_path)
                    
            elif self.stt_backend == "vosk":
                from vosk import KaldiRecognizer
                import json
                
                rec = KaldiRecognizer(model, sample_rate)
                rec.SetWords(True)
                
                if isinstance(audio_data, bytes):
                    rec.AcceptWaveform(audio_data)
                else:
                    with open(audio_data, 'rb') as audio_file:
                        while True:
                            data = audio_file.read(4000)
                            if len(data) == 0:
                                break
                            rec.AcceptWaveform(data)
                
                result = json.loads(rec.FinalResult())
                text = result.get('text', '').strip()
            
            else:
                text = ""
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s: {text}")
            
            return text, processing_time
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", time.time() - start_time
    
    def synthesize_speech(self, text: str, output_path: Optional[str] = None) -> Tuple[str, float]:
        """
        Convert text to speech using piper-tts or gTTS fallback.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path. If None, creates temp file.
            
        Returns:
            Tuple of (audio_file_path, processing_time_seconds)
        """
        start_time = time.time()
        model = self._load_tts_model()
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        
        try:
            if model['engine'] == 'piper':
                # Use piper-tts Python API
                import wave
                from piper import SynthesisConfig
                
                voice = model['voice']
                
                # Optional: configure synthesis parameters
                syn_config = SynthesisConfig(
                    volume=1.0,
                    length_scale=1.0,  # Speaking speed (lower = faster)
                    noise_scale=0.667,  # Audio variation
                    noise_w_scale=0.8,  # Speaking variation
                    normalize_audio=True
                )
                
                # Synthesize to WAV file
                with wave.open(output_path, "wb") as wav_file:
                    voice.synthesize_wav(text, wav_file, syn_config=syn_config)
                    
            elif model['engine'] == 'gtts':
                # Fallback to gTTS (requires internet)
                from gtts import gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(output_path)
            
            else:
                logger.error("No TTS engine available")
                return "", time.time() - start_time
            
            processing_time = time.time() - start_time
            logger.info(f"Speech synthesis completed in {processing_time:.2f}s")
            
            return output_path, processing_time
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return "", time.time() - start_time
    
    def detect_voice_activity(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """
        Detect if audio contains speech using Silero VAD.
        
        Args:
            audio_data: Audio as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (has_speech, confidence_score)
        """
        model = self._load_vad_model()
        
        if model is None:
            # Fallback: simple energy-based detection
            energy = np.sqrt(np.mean(audio_data ** 2))
            threshold = 0.01
            return energy > threshold, energy
        
        try:
            import torch
            
            # Ensure audio is float32 tensor
            if not isinstance(audio_data, torch.Tensor):
                audio_tensor = torch.FloatTensor(audio_data)
            else:
                audio_tensor = audio_data
            
            # Resample if needed (VAD expects 16kHz)
            if sample_rate != 16000:
                # Simple resampling
                from scipy import signal
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                audio_tensor = torch.FloatTensor(audio_data)
            
            # Get VAD probability
            vad_model = model['model']
            (get_speech_timestamps, _, _, _, _) = model['utils']
            
            speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
            
            has_speech = len(speech_timestamps) > 0
            confidence = 1.0 if has_speech else 0.0
            
            return has_speech, confidence
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False, 0.0


# Global singleton instance
_audio_processor_instance = None
_instance_lock = threading.Lock()


def get_audio_processor() -> AudioProcessor:
    """Get or create the global AudioProcessor instance (thread-safe singleton)"""
    global _audio_processor_instance
    
    if _audio_processor_instance is None:
        with _instance_lock:
            if _audio_processor_instance is None:
                _audio_processor_instance = AudioProcessor()
    
    return _audio_processor_instance
