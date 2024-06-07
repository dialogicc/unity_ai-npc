# %%
#https://huggingface.co/docs/transformers/model_doc/whisper
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa

print("Start AudioToText")

# Load your own audio recording:
# Read the audio file and resample it to 16,000 Hz
waveform, sampling_rate = librosa.load("Assets/Audio/audioInput.wav", sr=16000)

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny") # openai/whisper-... tiny, base, small, medium, large
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
# OLLAMA
import requests
import re  # Import the regular expressions module

print("Start TextGeneration")

def query_ollama(prompt):
    api_url = 'http://localhost:11434/v1/chat/completions'
    
    data = {
        "model": "gemma:2b",
        "messages": [
            {"role": "system", "content": "You are an AI, who answers the users questions."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 30  # Limit the number of tokens
    }
    
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        response_text = response.json()['choices'][0]['message']['content']
        # Remove everything in parentheses, including the parentheses
        cleaned_text = re.sub(r'\(.*?\)', '', response_text)
        # Limit the text to the last complete sentence
        last_full_stop = cleaned_text.rfind('.')
        if last_full_stop != -1:
            cleaned_text = cleaned_text[:last_full_stop + 1]
        return cleaned_text.strip()  # Remove unnecessary spaces
    else:
        return f"Error: {response.text}"

# Prompt you want to execute
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
    # Select device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")

    # Load model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    # Define text inputs
    prompt = response
    description = "A fast speaking male speaker."

    # Tokenization
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Text-to-Speech Generation
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    # Determine the absolute path to the target folder
    output_folder = os.path.join(os.getcwd(), "Assets", "Audio")

    # Ensure the target folder exists, if not, create it
    os.makedirs(output_folder, exist_ok=True)

    # Path to the output file in the target folder
    output_file = os.path.join(output_folder, "audioOutput.wav")

    # Save the audio file in the target folder
    sf.write(output_file, audio_arr, model.config.sampling_rate)

except Exception as e:
    print(f"Error during conversion: {str(e)}")

print("End TextToAudio")

# %%
import IPython.display as ipd

# Audio playback
ipd.Audio(audio_arr, rate=model.config.sampling_rate)

# %%
# Save audio
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



