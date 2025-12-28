import torch
import time
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper_kernels  # <--- YOUR KERNEL

device = "cuda"
print(f"DEVICE: {device.upper()}")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 1. Load Model (FP16)
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", 
    torch_dtype=torch.float16, 
    attn_implementation="sdpa"
).to(device)

model.eval()

# --- FIX 1: SAFER COMPILATION ---
# "max-autotune" was causing the memory overwrite crash.
# We switch to "default" which is stable for manual loops.
print("⏳ Compiling decoder (mode='default')...")
model.model.decoder = torch.compile(model.model.decoder, mode="default")

audio_path = "harvard.wav"
print(f"File: {audio_path}")

audio_array, _ = librosa.load(audio_path, sr=16000)
input_features = processor(
    audio_array, 
    sampling_rate=16000, 
    return_tensors="pt"
).input_features.to(device, dtype=torch.float16)

# --- MARKER: Inference ---
print("🚀 Running Whisper (Manual Loop + Custom Kernel)...")

torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    encoder_outputs = model.model.encoder(input_features)
    
    decoder_input_ids = torch.tensor([[50258]], device=device)
    past_key_values = None 
    
    # C. The Loop
    for i in range(50):
        # --- FIX 2: MARK STEP BEGIN ---
        # This prevents the "memory overwritten" error by telling the compiler 
        # that a new iteration has started.
        torch.compiler.cudagraph_mark_step_begin()

        # 1. Run Decoder
        out = model.model.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            use_cache=True
        )
        
        past_key_values = out.past_key_values
        logits = model.proj_out(out.last_hidden_state)
        
        # --- FIX 3: FLOAT CAST ---
        # Cast to float because your kernel expects float32
        next_token_id = whisper_kernels.fused_top1(logits[:, -1].float())
        
        decoder_input_ids = next_token_id.view(-1, 1)
        
        if next_token_id.item() == 50257:
            break

torch.cuda.synchronize()
end_time = time.time()

transcription = processor.batch_decode(decoder_input_ids, skip_special_tokens=True)[0]

print("-" * 40)
print(f"📝 Transcription: {transcription}")
print(f"⏱️ Time Taken: {end_time - start_time:.4f} seconds")
print("-" * 40)