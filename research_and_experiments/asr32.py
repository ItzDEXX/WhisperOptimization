import torch
import time
import librosa

from transformers import WhisperProcessor , WhisperForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device.upper())

processor= WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

audio_path = "audio.mp3"
print(audio_path)


audio_array, _ = librosa.load(audio_path,sr=16000)
input_features = processor(audio_array,sampling_rate=16000,return_tensors="pt").input_features.to(device)

print("🚀 Running Whisper...")
torch.cuda.synchronize()
start_time = time.time()

predicted_ids = model.generate(input_features)

torch.cuda.synchronize()
end_time = time.time()

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("-" * 40)
print(f"📝 Transcription: {transcription}")
print(f"⏱️ Time Taken: {end_time - start_time:.4f} seconds")
print("-" * 40)

