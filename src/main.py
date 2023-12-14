import torch


ssml_sample = """
  Ainsi, les salons se parent de bougies et de sapins. 
  En général, les français optent pour des sapins en pot afin de prolonger la magie du moment.
  L'arbre odorant est paré de guirlandes et de boules tandis qu'une étoile chatoyante est ajoutée sur la cime. 
"""

language = "fr"
model_id = "v3_fr"
sample_rate = 48000
speaker = "fr_5"
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
    audio_path="fr_5.wav",
)
print(audio_paths)
