import gradio as gr
import numpy as np
import tempfile
import os
from pathlib import Path

# Mock model for testing when real model can't load
USE_MOCK_MODEL = True

def initialize_model():
    """Initialize model - using mock for testing"""
    global USE_MOCK_MODEL
    if USE_MOCK_MODEL:
        print("🧪 Using mock model for testing (real model has PyTorch compatibility issues)")
        return "mock_model", None
    return None, None

def transcribe_audio(audio_file):
    """
    Transcribe audio using mock model for testing.
    """
    if audio_file is None:
        return "Please upload an audio file."
    
    try:
        # Initialize model if needed
        model, processor = initialize_model()
        if model is None:
            return "❌ Error: Could not load the model. Please try again later."
        
        filename = Path(audio_file).name
        print(f"🎵 Processing audio file: {filename}")
        
        # Mock transcription based on sample files
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
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return f"❌ Error during transcription: {str(e)}"

def transcribe_microphone(audio_data):
    """
    Transcribe audio from microphone input.
    """
    if audio_data is None:
        return "Please record some audio first."
    
    try:
        sample_rate, audio_array = audio_data
        duration = len(audio_array) / sample_rate
        
        print(f"🎙️ Processing microphone input: {duration:.1f} seconds at {sample_rate}Hz")
        
        return f"Mock transcription for {duration:.1f}s audio: [This would be the actual Kinyarwanda transcription]"
            
    except Exception as e:
        print(f"❌ Microphone processing error: {e}")
        return f"❌ Error processing microphone input: {str(e)}"

# Create a simple Gradio interface
def create_interface():
    """Create a clean, simple Gradio interface."""
    
    with gr.Blocks(title="Wakanda Whisper - Kinyarwanda ASR") as interface:
        
        gr.Markdown("# 🎤 Wakanda Whisper")
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
                        mic_btn = gr.Button("🎯 Transcribe Recording", variant="primary")
                    
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
    print("🚀 Starting Wakanda Whisper ASR (Mock Mode for Testing)...")
    
    # Create and launch the interface
    demo = create_interface()
    
    # Launch configuration - let Gradio find an available port
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True
    )
