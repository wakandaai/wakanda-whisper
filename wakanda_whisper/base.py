from huggingface_hub import hf_hub_download
import whisper
from whisper import _ALIGNMENT_HEADS
from whisper.model import Whisper, ModelDimensions
from whisper.transcribe import transcribe as transcribe_function
import torch
import yaml
import functools


def from_pretrained(model_name: str, device: str = None):
    """
    Load a pretrained model by name. Name must match a model name on Hugging Face.
    Example: "WakandaAI/wakanda-whisper-small-rw-v1"
    """
    # download the config and checkpoint from huggingface
    config = hf_hub_download(repo_id=model_name, filename="config.yaml")
    model_path = hf_hub_download(repo_id=model_name, filename="model.pt")
    
    if config:
        with open(config, "r") as f:
                config = yaml.safe_load(f)
    else:
        raise ValueError(f"Model '{model_name}' does not have a valid config.json file.")
    
    language = config.get("language")
    whisper_size = config.get("whisper-size")
    dims = config.get("dims")

    if not whisper_size:
        raise ValueError(f"Model '{model_name}' does not have a valid 'whisper-size' in its config.")
    if not language:
        raise ValueError(f"Model '{model_name}' does not have a valid 'language' in its config.")
    if not dims:
        raise ValueError(f"Model '{model_name}' does not have a valid 'dims' in its config.")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        def transcribe(model, audio, **kwargs):
            if language != "multilingual":
                kwargs['language'] = language
            return transcribe_function(model, audio, without_timestamps=True, **kwargs)
            
        Whisper.transcribe = transcribe

        dims  = ModelDimensions(**dims)
        model = Whisper(dims)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.set_alignment_heads(_ALIGNMENT_HEADS[whisper_size])

        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")