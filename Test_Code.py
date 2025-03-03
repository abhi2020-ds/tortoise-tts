import torch
import torchaudio
import os
from tortoise.api import TextToSpeech
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

tts = TextToSpeech(use_deepspeed=True, kv_cache=True)

from tortoise.utils.text import split_and_recombine_text
from time import time
import os

outpath = "results/longform/"

textfile_path = "./tortoise/data/de.txt"

# Process text
with open(textfile_path, 'r', encoding='utf-8') as f:
    text = ' '.join([l for l in f.readlines()])
    if '|' in text:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        texts = text.split('|')
    else:
        texts = split_and_recombine_text(text)

seed = int(time())

CUSTOM_VOICE_NAME = "tom"

voice_outpath = os.path.join(outpath, CUSTOM_VOICE_NAME)
os.makedirs(voice_outpath, exist_ok=True)

voice_samples, conditioning_latents = load_voice(CUSTOM_VOICE_NAME)

all_parts = []
for j, text in enumerate(texts):
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                              preset="ultra_fast", k=1, use_deterministic_seed=seed)
    gen = gen.squeeze(0).cpu()
    torchaudio.save(os.path.join(voice_outpath, f'{j}.wav'), gen, 24000)
    all_parts.append(gen)

full_audio = torch.cat(all_parts, dim=-1)
torchaudio.save(os.path.join(voice_outpath, 'combined.wav'), full_audio, 24000)