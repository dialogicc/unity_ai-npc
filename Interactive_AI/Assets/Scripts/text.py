# %%
#https://huggingface.co/docs/transformers/model_doc/whisper
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa

print("Start AudioToText")

# # Select an audio file and read it:
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# audio_sample = ds[2]["audio"]
# waveform = audio_sample["array"]
# sampling_rate = audio_sample["sampling_rate"]

# Eigene Audioaufnahme laden:
# Lese die Audiodatei und ändere die Abtastrate auf 16.000 Hz
waveform, sampling_rate = librosa.load("/Users/franz/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studium/Master/4. Master/MA/UnityProjects/Interactive_AI/Assets/Audio/audioInput.wav", sr=16000)

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")#openai/whisper-... tiny, base, small, medium, large
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny") 

# Use the model and processor to transcribe the audio:
input_features = processor(
    waveform, sampling_rate=sampling_rate, return_tensors="pt"
).input_features

# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

mic_to_text = transcription[0]

print(mic_to_text)

print("End AudioToText")


# %%
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# %% [markdown]
# https://ollama.com/blog/openai-compatibility

# %%
#OLLAMA
import requests
import re  # Importiere das Modul für reguläre Ausdrücke

print("Start TextGeneration")

def query_ollama(prompt):
    api_url = 'http://localhost:11434/v1/chat/completions'
    
    data = {
        "model": "gemma:2b",
        "messages": [
            {"role": "system", "content": "You are an AI, who answers the users questions."},
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        response_text = response.json()['choices'][0]['message']['content']
        # Entferne alles, was in Klammern steht, einschließlich der Klammern
        cleaned_text = re.sub(r'\(.*?\)', '', response_text)
        return cleaned_text.strip()  # Entferne überflüssige Leerzeichen
        return response_text
    else:
        return f"Fehler: {response.text}"


# Prompt, den du ausführen möchtest
prompt = mic_to_text + "(. Answer in english)"
response = query_ollama(prompt)
print(response)

print("End TextGeneration")


# %% [markdown]
# https://huggingface.co/parler-tts/parler_tts_mini_v0.1

import os
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

print("Start TextToAudio")

try:
    # Gerät festlegen
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")

    # Model und Tokenizer laden
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    # Texteingaben definieren
    prompt = response
    description = "A male speaker."

    # Tokenisierung
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Text-to-Speech Generation
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    # Bestimmen Sie den absoluten Pfad zum Zielordner
    output_folder = os.path.join(os.getcwd(), "Assets", "Audio")

    # Stellen Sie sicher, dass der Zielordner existiert, falls nicht, erstellen Sie ihn
    os.makedirs(output_folder, exist_ok=True)

    # Pfad zur Ausgabedatei im Zielordner
    output_file = os.path.join(output_folder, "audioOutput.wav")

    # Speichern Sie die Audiodatei im Zielordner
    sf.write(output_file, audio_arr, model.config.sampling_rate)

except Exception as e:
    print(f"Fehler bei der Umwandlung: {str(e)}")

print("End TextToAudio")



# %%
import IPython.display as ipd

# Audioausgabe
ipd.Audio(audio_arr, rate=model.config.sampling_rate)


# %%
# Audio speichern
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)


# %% [markdown]
# https://huggingface.co/suno/bark-small

# %%
# from transformers import AutoProcessor, AutoModel

# processor = AutoProcessor.from_pretrained("suno/bark-small")
# model = AutoModel.from_pretrained("suno/bark-small")

# inputs = processor(
#     text=[response],
#     return_tensors="pt",
# )

# speech_values = model.generate(**inputs, do_sample=True)

# %%
# from IPython.display import Audio

# sampling_rate = model.generation_config.sample_rate
# Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)


# %%
# # Sampling-Rate annehmen oder aus der Modellkonfiguration beziehen
# sampling_rate = 22050  # Beispielrate

# # Speichere die Audiodaten als WAV-Datei
# audio_arr = speech_values.cpu().numpy().squeeze()
# scipy.io.wavfile.write("output.wav", rate=sampling_rate, data=audio_arr)

# %%
# import torch
# from transformers import AutoModel, AutoProcessor
# import scipy

# # Setze das Gerät, auf dem das Modell laufen soll
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Lade das Modell und setze es auf das Gerät
# model = AutoModel.from_pretrained("suno/bark-small").to(device)

# # Stelle sicher, dass das Modell in halber Präzision läuft, um Speicher zu sparen und Geschwindigkeit zu verbessern
# #model = model.half()
# model = model.float()  # Setzt das Modell zurück auf volle Präzision

# # Lade den Processor
# processor = AutoProcessor.from_pretrained("suno/bark-small")

# # Dein zu transformierender Text
# response = response

# # Verarbeite den Text für das Modell
# inputs = processor(
#     text=[response],
#     return_tensors="pt",
# )
# inputs = {k: v.to(device) for k, v in inputs.items()}

# # Generiere die Sprachdaten
# speech_values = model.generate(**inputs, do_sample=True)



# %%
# import scipy
# import numpy as np

# # Versuche verschiedene Sampling-Raten
# for rate in [16000, 22050, 44100, 48000]:
#     scipy.io.wavfile.write(f"output_{rate}.wav", rate=rate, data=audio_arr)

# # Überprüfe jede Datei, um zu sehen, welche die korrekte Abspielgeschwindigkeit hat


# %%
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-1.1-2b-it",
#     torch_dtype=torch.bfloat16
# )

# input_text = mic_to_text
# input_ids = tokenizer(input_text, return_tensors="pt")

# outputs = model.generate(**input_ids, max_new_tokens=50)
# print(tokenizer.decode(outputs[0]))


# %%
# import torch
# from transformers import pipeline

# # Pipeline mit float32, wenn BFloat16 nicht unterstützt wird
# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)

# # Verwenden des Tokenizer-Chat-Templates
# messages = [
#     {
#         "role": "system",
#         "content": "You are an english speaking chatbot, give short answers.",
#     },
#     {"role": "user", "content": mic_to_text},
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=40, do_sample=True, temperature=0.5, top_k=30, top_p=0.8)
# print(outputs[0]["generated_text"])



