import gradio as gr
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Try to import wakanda_whisper, fallback to transformers if not available
try:
    import wakanda_whisper
    USE_WAKANDA_WHISPER = True
    print("✅ Using wakanda_whisper package")
except ImportError:
    print("⚠️ wakanda_whisper not found, falling back to transformers...")
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import librosa
        USE_WAKANDA_WHISPER = False
        print("✅ Using transformers as fallback")
    except ImportError:
        print("❌ Neither wakanda_whisper nor transformers available")
        USE_WAKANDA_WHISPER = None

# Initialize the model
def load_model():
    """Load the Wakanda Whisper model from Hugging Face."""
    try:
        if USE_WAKANDA_WHISPER:
            # Use wakanda_whisper if available
            print("📥 Loading model with wakanda_whisper...")
            model = wakanda_whisper.from_pretrained("WakandaAI/wakanda-whisper-small-rw-v1")
            return model, None
        elif USE_WAKANDA_WHISPER is False:
            # Fallback to transformers
            print("📥 Loading model with transformers...")
            processor = WhisperProcessor.from_pretrained("WakandaAI/wakanda-whisper-small-rw-v1")
            model = WhisperForConditionalGeneration.from_pretrained("WakandaAI/wakanda-whisper-small-rw-v1")
            return model, processor
        else:
            print("❌ No compatible libraries available")
            return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# Global model variables
MODEL = None
PROCESSOR = None

def initialize_model():
    """Initialize model on first use"""
    global MODEL, PROCESSOR
    if MODEL is None:
        print("🚀 Initializing model...")
        MODEL, PROCESSOR = load_model()
    return MODEL, PROCESSOR

def transcribe_audio(audio_file):
    """
    Transcribe audio using the Wakanda Whisper model.
    """
    if audio_file is None:
        return "Please upload an audio file."
    
    try:
        # Initialize model if needed
        model, processor = initialize_model()
        if model is None:
            return "❌ Error: Could not load the model. Please try again later."
        
        print(f"🎵 Processing audio file: {Path(audio_file).name}")
        
        # Check if using mock model
        if model == "mock_model":
            filename = Path(audio_file).name
            if "sample_1" in filename:
                return "Muraho, witwa gute?"
            elif "sample_2" in filename:
                return "Ndashaka kwiga Ikinyarwanda."
            elif "sample_3" in filename:
                return "Urakoze cyane kubafasha."
            elif "sample_4" in filename:
                return "Tugiye gutangiza ikiganiro mu Kinyarwanda."
            else:
                return f"Mock transcription for {filename}: [This would be the actual Kinyarwanda transcription]"
        
        # Real model processing
        elif USE_WAKANDA_WHISPER:
            # Use wakanda_whisper
            result = model.transcribe(audio_file)
            transcribed_text = result['text'].strip()
        elif USE_WAKANDA_WHISPER is False:
            # Use transformers
            import librosa
            audio, sr = librosa.load(audio_file, sr=16000)
            input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
            
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            
            transcribed_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        else:
            return "❌ Error: No compatible transcription library available."
        
        if not transcribed_text:
            return "🔇 No speech detected in the audio file. Please try with a clearer audio recording."
        
        print(f"✅ Transcription completed: {len(transcribed_text)} characters")
        return transcribed_text
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return f"❌ Error during transcription: {str(e)}"

def transcribe_microphone(audio_data):
    """
    Transcribe audio from microphone input.
    
    Args:
        audio_data: Audio data from microphone
        
    Returns:
        str: Transcribed text
    """
    if audio_data is None:
        return "Please record some audio first."
    
    try:
        # Save the audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            # audio_data is a tuple (sample_rate, audio_array)
            sample_rate, audio_array = audio_data
            
            print(f"🎙️ Processing microphone input: {len(audio_array)} samples at {sample_rate}Hz")
            
            # Convert to float32 and normalize if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                if audio_array.max() > 1.0:
                    # Normalize based on the original dtype
                    if audio_array.max() > 32767:
                        audio_array = audio_array / 32768.0
                    else:
                        audio_array = audio_array / audio_array.max()
            
            # Save using soundfile
            import soundfile as sf
            sf.write(tmp_file.name, audio_array, sample_rate)
            
            # Transcribe the temporary file
            result = transcribe_audio(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return result
            
    except Exception as e:
        print(f"❌ Microphone processing error: {e}")
        return f"❌ Error processing microphone input: {str(e)}"

# Create a simple Gradio interface
def create_interface():
    """Create a clean, simple Gradio interface."""
    
    with gr.Blocks(title="Wakanda Whisper - Kinyarwanda ASR") as interface:
        
        gr.Markdown("# Wakanda Whisper")
        gr.Markdown("### Kinyarwanda Automatic Speech Recognition")
        gr.Markdown("Upload an audio file or record your voice to get Kinyarwanda transcription")
        
        with gr.Tabs():
            # File Upload Tab
            with gr.TabItem("📁 Upload Audio File"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Choose Audio File",
                            type="filepath"
                        )
                        
                        # Sample audio files
                        gr.Markdown("**Try these sample Kinyarwanda audio files:**")
                        with gr.Row():
                            sample_1 = gr.Button("Sample 1", size="sm")
                            sample_2 = gr.Button("Sample 2", size="sm")
                            sample_3 = gr.Button("Sample 3", size="sm")
                            sample_4 = gr.Button("Sample 4", size="sm")
                        
                        upload_btn = gr.Button("🎯 Transcribe Audio", variant="primary")
                    
                    with gr.Column():
                        upload_output = gr.Textbox(
                            label="Transcription Result",
                            placeholder="Your Kinyarwanda transcription will appear here...",
                            lines=6,
                            show_copy_button=True
                        )
            
            # Microphone Tab
            with gr.TabItem("🎙️ Record Audio"):
                with gr.Row():
                    with gr.Column():
                        mic_input = gr.Audio(
                            label="Record Your Voice",
                            type="numpy"
                        )
                        mic_btn = gr.Button(" Transcribe Recording", variant="primary")
                    
                    with gr.Column():
                        mic_output = gr.Textbox(
                            label="Transcription Result",
                            placeholder="Your Kinyarwanda transcription will appear here...",
                            lines=6,
                            show_copy_button=True
                        )
        
        # Set up event handlers
        upload_btn.click(
            fn=transcribe_audio,
            inputs=audio_input,
            outputs=upload_output,
            show_progress=True
        )
        
        # Sample audio button handlers
        sample_1.click(
            fn=lambda: "sample_1.wav",
            outputs=audio_input
        )
        sample_2.click(
            fn=lambda: "sample_2.wav",
            outputs=audio_input
        )
        sample_3.click(
            fn=lambda: "sample_3.wav",
            outputs=audio_input
        )
        sample_4.click(
            fn=lambda: "sample_4.wav",
            outputs=audio_input
        )
        
        mic_btn.click(
            fn=transcribe_microphone,
            inputs=mic_input,
            outputs=mic_output,
            show_progress=True
        )
        
        gr.Markdown("---")
        gr.Markdown("**Powered by WakandaAI** | Model: [wakanda-whisper-small-rw-v1](https://huggingface.co/WakandaAI/wakanda-whisper-small-rw-v1)")
    
    return interface

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting Wakanda Whisper ASR Demo...")
    
    # Create and launch the interface
    demo = create_interface()
    
    # Launch configuration for Hugging Face Spaces
    demo.launch(
        server_name="0.0.0.0",
        share=False,  # Set to False for Hugging Face Spaces
        show_error=True
    )
