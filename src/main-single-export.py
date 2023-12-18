import torch


ssml_sample = """
  Hey there! Welcome back to another exciting episode. Let's dive iin!
"""

language = "en"
model_id = "v3_en"
sample_rate = 48000
speaker = "en_117"
device = torch.device("cpu")

res = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language=language,
    speaker=model_id,
)
model = res[0]
model.to(device)

audio_paths = model.save_wav(
    text=ssml_sample,
    speaker=speaker,
    sample_rate=sample_rate,
    audio_path="en_117_24.wav",
)
print(audio_paths)
