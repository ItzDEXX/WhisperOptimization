import torch
import time
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper_kernels

device = "cuda"
print(f"DEVICE: {device.upper()}")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")


model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", 
    torch_dtype=torch.float16, 
    attn_implementation="sdpa"
).to(device)

model.eval()
model.generation_config.cache_implementation = "static"


torch.cuda.nvtx.range_push("Compilation")  
print("⏳ Compiling model (this takes ~1 min the first time)...")
model = torch.compile(model, mode="reduce-overhead")
torch.cuda.nvtx.range_pop()               
audio_path = "harvard.wav"
print(f"File: {audio_path}")

audio_array, _ = librosa.load(audio_path, sr=16000)
input_features = processor(
    audio_array, 
    sampling_rate=16000, 
    return_tensors="pt"
).input_features.to(device, dtype=torch.float16)


torch.cuda.nvtx.range_push("Warmup")    
print("🔥 Warming up (building CUDA Graph)...")
with torch.no_grad():
    model.generate(input_features)
torch.cuda.nvtx.range_pop()           


print("🚀 Running Whisper (CUDA Graph Optimized)...")
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("Inference")   
start_time = time.time()

with torch.no_grad():
    predicted_ids = model.generate(input_features)

torch.cuda.synchronize()
end_time = time.time()
torch.cuda.nvtx.range_pop()                

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("-" * 40)
print(f"📝 Transcription: {transcription}")
print(f"⏱️ Time Taken: {end_time - start_time:.4f} seconds")
print("-" * 40)