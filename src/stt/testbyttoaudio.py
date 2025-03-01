import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import soundfile as sf


def load_audio(file_path):

    audio_input, sample_rate = librosa.load(file_path, sr=16000)  
    return audio_input, sample_rate


def transcribe_audio(file_path):

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

    audio_input, sample_rate = load_audio(file_path)
    

    input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values


    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription

audio_file = "hi.wav"
text = transcribe_audio(audio_file)
print("Transcription: ", text)
