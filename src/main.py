import torch


ssml_sample = """
  Aunque pueda parecer una afirmación un tanto restrictiva, cualquier trabajo científico y profesional no existe si no se comparte con la comunidad, 
  desde su versión académica o desde la vertiente más profesional, ya sea de manera formal o informal. 
  La experiencia profesional o los resultados de una investigación están completos solo cuando sus 
  resultados y conclusiones se comparten con el objetivo del enriquecimiento mutuo a nivel individual 
  (profesional y académico) y a nivel comunitario y social. 
"""

language = "es"
model_id = "v3_es"
sample_rate = 48000
speaker = "es_2"
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
    audio_path="es_2.wav",
)
print(audio_paths)
