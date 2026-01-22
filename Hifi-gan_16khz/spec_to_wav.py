import torch
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from pathlib import Path
import pandas as pd

# HiFi-GAN vocoder betöltése
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz", 
    savedir="pretrained_models/tts-hifigan-libritts-16kHz"
)

# Manifest beolvasása
manifest_path = "/home/arcdeus/Documents/NewThesis/manifest.csv"
df = pd.read_csv(manifest_path)

# Kimeneti mappák
spec_output_dir = Path("/home/arcdeus/Documents/NewThesis/spectrograms")
wav_output_dir = Path("/home/arcdeus/Documents/NewThesis/reconstructed_wavs")
spec_output_dir.mkdir(parents=True, exist_ok=True)
wav_output_dir.mkdir(parents=True, exist_ok=True)

# Néhány wav fájl feldolgozása (pl. első 5)
for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    wav_path = row['wav']
    utt_id = row['utt_id']
    
    # WAV betöltése
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # Ha nem 16kHz, resample
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Mel-spektrogram készítése a HiFi-GAN kompatibilis paraméterekkel
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000
    )
    
    mel_spec = mel_spectrogram_transform(waveform)
    
    # Log scale (opcionális, de általában használt)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    print(f"✓ {utt_id}: mel shape={mel_spec.shape}")
    
    # Mentés .pt fájlként
    torch.save(mel_spec, spec_output_dir / f"{utt_id}.pt")
    
    # MEL → WAV konverzió HiFi-GAN vocoderrel
    # Squeeze ha szükséges (remove channel dim ha van)
    mel_for_vocoder = mel_spec.permute(0, 2, 1)
    mel_for_vocoder = mel_spec.squeeze(0) if mel_spec.dim() == 3 else mel_spec
 
    print("mel_spec:", mel_spec.shape)
    print("mel_for_vocoder:", mel_for_vocoder.shape)

    # HiFi-GAN decode - shape kell: [batch, n_mels, time]
    if mel_for_vocoder.dim() == 2:
        mel_for_vocoder = mel_for_vocoder.unsqueeze(0)  # [1, 80, time]
    
    with torch.no_grad():
        reconstructed_wav = hifi_gan.decode_batch(mel_for_vocoder)
    
    # Mentés rekonstruált wav-ként
    reconstructed_wav = reconstructed_wav.squeeze()  # remove extra dims
    torchaudio.save(
        wav_output_dir / f"{utt_id}_reconstructed.wav",
        reconstructed_wav.unsqueeze(0).cpu(),  # [1, samples]
        sample_rate=16000
    )
    
    print(f"  → Rekonstruált wav shape: {reconstructed_wav.shape}")
    print(f"  → Mentve: {utt_id}_reconstructed.wav\n")

print(f"✓ Kész! {min(5, len(df))} spektrogram és rekonstruált wav elkészítve.")
print(f"Spektrogramok: {spec_output_dir}")
print(f"Rekonstruált wavok: {wav_output_dir}")