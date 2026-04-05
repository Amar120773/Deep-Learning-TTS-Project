import gradio as gr
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import numpy as np

# Load models once
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Fixed neutral voice
torch.manual_seed(42)
speaker_embeddings = torch.randn(1, 512)

def generate(text):
    if not text or not text.strip():
        return None

    # Improve naturalness slightly by adding punctuation if missing
    if text[-1] not in [".", "!", "?"]:
        text += "."

    inputs = processor(text=text, return_tensors="pt")

    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings,
        vocoder=vocoder
    )

    audio = speech.cpu().numpy()

    # Normalize safely
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Slight smoothing (reduces harshness)
    audio = audio * 0.95

    # Save output
    sf.write("output.wav", audio, samplerate=16000)

    return "output.wav"


# UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ AI Text-to-Speech")
    gr.Markdown("Generate natural speech from text (Neutral Voice)")

    text = gr.Textbox(
        label="Enter Text",
        placeholder="Type something like: Hello, this is a demo of my AI project.",
        lines=3
    )

    generate_btn = gr.Button("Generate Voice")

    output_audio = gr.Audio(label="Output Audio")

    generate_btn.click(
        fn=generate,
        inputs=text,
        outputs=output_audio
    )

demo.launch(share=True)