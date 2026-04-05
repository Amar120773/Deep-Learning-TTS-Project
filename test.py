from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf

# Load models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Use random speaker embedding (instead of dataset)
speaker_embeddings = torch.randn(1, 512)

# Input text
text = "Hello, this is my AI text to speech project."

inputs = processor(text=text, return_tensors="pt")

# Generate speech
speech = model.generate_speech(
    inputs["input_ids"],
    speaker_embeddings,
    vocoder=vocoder
)

# Save output
sf.write("output.wav", speech.numpy(), samplerate=16000)

print("Audio generated successfully!")