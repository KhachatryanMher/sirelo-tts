import asyncio
import torch


async def generate_audio(index):
  ssml_sample = f"""
    Oh, what an extraordinary day it has been! 
    The sun, with its radiant beams, greeted me as I stepped outside. 
    The birds, oh, the birds, sang a symphony of joy, filling the air with their melodious tunes. 
    I, with a heart full of gratitude, embraced the beauty that surrounded me.
  """

  language = "en"
  model_id = "v3_en"
  sample_rate = 48000
  speaker = f"en_{index}"
  device = torch.device("cpu")

  res = torch.hub.load(
      repo_or_dir="snakers4/silero-models",
      model="silero_tts",
      language=language,
      speaker=speaker,
  )
  model = res[0]
  model.to(device)

  audio_paths = model.save_wav(
    text=ssml_sample,
    sample_rate=sample_rate,
    audio_path=f"en_{index}.wav",
  )
  print(audio_paths)


async def main():
  tasks = [generate_audio(index) for index in range(117)]
  await asyncio.gather(*tasks)


if __name__ == "__main__":
  asyncio.run(main())
