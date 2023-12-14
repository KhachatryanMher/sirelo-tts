import torch

ssml_sample = """
  <speak>
  <p>
    When I wake up, <prosody rate="x-slow">I speak quite slowly</prosody>.
    Then I start speaking in my regular voice,
    <prosody pitch="x-high"> or I can speak with a higher pitch </prosody>,
    or <prosody pitch="x-low">on the contrary, lower</prosody>.
    Then, if I'm lucky – <prosody rate="fast">I can speak quite quickly.</prosody>
    I can also make pauses of any length, for example, two seconds <break time="2000ms"/>.
  </p>
  <p>
    Additionally, I can make pauses between paragraphs.
  </p>
  <p>
    <s>And I can also make pauses between sentences</s>
    <s>Like right now</s>
  </p>
</speak>

  """

language = "en"
model_id = "v3_en"
sample_rate = 48000
speaker = "en_0"
device = torch.device("cpu")

model, text = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language=language,
    speaker=model_id,
)
model.to(device)  # gpu or cpu

audio_paths = model.save_wav(text=ssml_sample, speaker=speaker, sample_rate=sample_rate)
print(audio_paths)
