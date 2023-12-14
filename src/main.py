import torch

ssml_sample = """
  <speak>
  <p>
      Когда я просыпаюсь, <prosody rate="x-slow">я говорю довольно медленно</prosody>.
      Потом я начинаю говорить своим обычным голосом,
      <prosody pitch="x-high"> а могу говорить тоном выше </prosody>,
      или <prosody pitch="x-low">наоборот, ниже</prosody>.
      Потом, если повезет – <prosody rate="fast">я могу говорить и довольно быстро.</prosody>
      А еще я умею делать паузы любой длины, например две секунды <break time="2000ms"/>.
      <p>
        Также я умею делать паузы между параграфами.
      </p>
      <p>
        <s>И также я умею делать паузы между предложениями</s>
        <s>Вот например как сейчас</s>
      </p>
  </p>
  </speak>
"""

language = "ru"
model_id = "v3_1_ru"
sample_rate = 48000
speaker = "xenia"
device = torch.device("cpu")

res = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language=language,
    speaker=model_id,
)
model = res[0]
model.to(device)

audio_paths = model.save_wav(text=ssml_sample, speaker=speaker, sample_rate=sample_rate)
print(audio_paths)
