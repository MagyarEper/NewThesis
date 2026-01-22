import torch
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Load vocoder
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir="pretrained_models/tts-hifigan-libritts-16kHz"
)

# Load wav (MONO, 16 kHz)
signal, sr = torchaudio.load("/home/arcdeus/Documents/NewThesis/wavs_16khz/C_001_0001_stove_NULL_AO_2.wav")
assert sr == 16000

signal = signal[0]  # [samples]

# Compute mel EXACTLY as expected by the vocoder
mel, _ = mel_spectogram(
    audio=signal,
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=True
)

# mel shape: [1, T, 80]  ← EZ FONTOS
print(mel.shape)

# Vocoder
with torch.no_grad():
    wav_hat = hifi_gan.decode_batch(mel)

# Save reconstructed audio
torchaudio.save(
    "/home/arcdeus/Documents/NewThesis/reconstructed_wavs/C_001_0001_stove_NULL_AO_2_reconstructed.wav",
    wav_hat.squeeze(1).cpu(),
    16000
)
