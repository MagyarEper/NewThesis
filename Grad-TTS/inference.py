# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

from speechbrain.inference.vocoders import HIFIGAN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    args = parser.parse_args()
    
    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None
    
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN (SpeechBrain)...')
    vocoder = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz",
        savedir="pretrained_models/tts-hifigan-libritts-16kHz"
    )
    
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 16000 / (y_dec.shape[-1] * 256)}')

            # CRITICAL FIX: Clamp mel to training range before vocoding
            # Training mels are in range [-11.5129, ~-3.27] (compression=True + min_max_norm)
            # Grad-TTS sometimes outputs values outside this range (up to +0.33)
            # This causes vocoder distortion since it never saw such values during training
            y_dec = torch.clamp(y_dec, min=-11.5129, max=-3.0)
            
            # SpeechBrain HiFi-GAN vocoder használata
            # y_dec shape kell: [batch, n_mels, time] vagy [batch, time, n_mels]
            # SpeechBrain expects: [batch, time, n_mels]
            if y_dec.dim() == 2:  # [n_mels, time]
                y_dec = y_dec.transpose(0, 1).unsqueeze(0)  # [1, time, n_mels]
            elif y_dec.shape[1] == 80:  # [batch, n_mels, time]
                y_dec = y_dec.transpose(1, 2)  # [batch, time, n_mels]
            
            audio_tensor = vocoder.decode_batch(y_dec)
            audio = (audio_tensor.cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f'./out/sample_{i}.wav', 16000, audio)

    print('Done. Check out `out` folder for samples.')
