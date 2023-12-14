import torch


ssml_sample = """
  Heinrich kommt aus Paris.
  Das ist die Hauptstadt von Frankreich. 
  In diesem Sommer macht sie einen Sprachkurs in Freiburg. 
  Das ist eine Universitätsstadt im Süden von Deutschland.
  Es gefällt ihr hier sehr gut. Morgens um neun beginnt der Unterricht, um vierzehn Uhr ist er zu Ende.
"""

language = "de"
model_id = "v3_de"
sample_rate = 48000
speaker = ""
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
    audio_path="de-karlsson.wav",
)
print(audio_paths)
