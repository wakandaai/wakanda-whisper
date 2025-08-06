import whisper
import torch


@torch.no_grad()
def batch_greedy_transcribe(model, mel, tokenizer):
    mel = mel.half().to(model.device)
    audio_features = model.encoder(mel)
    sot_tokens = tokenizer.sot_sequence_including_notimestamps
    prev_tokens = torch.tensor(sot_tokens).long().unsqueeze(0).repeat(mel.size(0),1)
    prev_tokens = prev_tokens.to(model.device)
    dones = torch.zeros(mel.size(0)).bool().to(model.device)
    end_id = torch.zeros(mel.size(0)).long().to(model.device)
    
    for i in range(prev_tokens.size(1), 448):
        logits = model.decoder(prev_tokens, audio_features)
        next_token = logits[:, -1, :].argmax(dim=-1)
        dones |= (next_token == tokenizer.eot)
        end_id.masked_fill_(~dones, i)
        prev_tokens = torch.cat([prev_tokens, next_token.unsqueeze(1)], dim=-1)
        if dones.all().item() == True:
            break

    texts = []
    for i in range(mel.size(0)):
        texts.append(
            tokenizer.decode(prev_tokens[i].tolist()[len(sot_tokens):end_id[i].item()+1]).strip()
        )

    return texts


@torch.no_grad()
def transcribe(model, tokenizer, audio):
    """
    Transcribe audio using the model and tokenizer.
    """
    if isinstance(audio, list):
        temp = []
        for a in audio:
            a = whisper.load_audio(a)
            a = whisper.pad_or_trim(a)
            temp.append(whisper.log_mel_spectrogram(a, n_mels=model.dims.n_mels, device=model.device))
        mel = torch.stack(temp, dim=0)
    else:
        audio = whisper.load_audio(audio)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels, device=model.device)
        mel = mel.unsqueeze(0)
    num = mel.size(0)

    texts = batch_greedy_transcribe(model, mel, tokenizer)
    return texts[0] if num == 1 else texts
