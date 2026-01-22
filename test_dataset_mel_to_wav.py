import sys
sys.path.append('/home/arcdeus/Documents/NewThesis/Grad-TTS')

import torch
import torchaudio
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from torch.utils.data import DataLoader
from speechbrain.inference.vocoders import HIFIGAN
from pathlib import Path

print("="*60)
print("DATASET MEL → SPEECHBRAIN HIFI-GAN → WAV ROUND-TRIP TEST")
print("="*60)

# HiFi-GAN vocoder betöltése
print("\n1. HiFi-GAN vocoder betöltése...")
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir="pretrained_models/tts-hifigan-libritts-16kHz"
)
print("✓ Vocoder betöltve")

# Dataset betöltése
print("\n2. Dataset betöltése...")
dataset = TextMelSpeakerDataset(
    filelist_path='/home/arcdeus/Documents/NewThesis/Grad-TTS/resources/filelists/train.txt',
    cmudict_path='/home/arcdeus/Documents/NewThesis/Grad-TTS/resources/cmu_dictionary',
    add_blank=True,
    n_fft=1024,
    n_mels=80,
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    f_min=0.0,
    f_max=8000.0
)
print(f"✓ Dataset méret: {len(dataset)} sample")

# Collate function és DataLoader
collate_fn = TextMelSpeakerBatchCollate()
dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

# Első batch betöltése
print("\n3. Első batch betöltése (5 minta)...")
batch = next(iter(dataloader))

# Debug: nézzük meg a batch struktúráját
print(f"Batch type: {type(batch)}")
if isinstance(batch, tuple):
    print(f"Batch tuple hossz: {len(batch)}")
    for i, item in enumerate(batch):
        print(f"  batch[{i}] type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
    x, x_lengths, y, y_lengths, spk = batch
elif isinstance(batch, dict):
    print(f"Batch dict keys: {batch.keys()}")
    x = batch['x']
    x_lengths = batch['x_lengths']
    y = batch['y']
    y_lengths = batch['y_lengths']
    spk = batch['spk']

print(f"✓ Batch betöltve")
print(f"  y (mel) shape: {y.shape}")
print(f"  y dtype: {y.dtype}")
print(f"  y min: {y.min().item():.4f}, max: {y.max().item():.4f}, mean: {y.mean().item():.4f}")

# Kimeneti mappa
output_dir = Path("/home/arcdeus/Documents/NewThesis/dataset_mel_reconstructed_wavs")
output_dir.mkdir(parents=True, exist_ok=True)

print("\n4. MEL → WAV konverzió (5 minta)...")
print("="*60)

for i in range(5):
    print(f"\nMinta {i+1}/5:")
    
    # Mel kivétele: y[i] shape = [80, T]
    mel = y[i:i+1, :, :y_lengths[i]]  # [1, 80, T]
    print(f"  Mel shape: {mel.shape}")
    
    # SpeechBrain HiFi-GAN [B, channels, time] formátumot vár
    # Már jó formátumban van: [1, 80, T]
    # NEM kell transpose!
    print(f"  Mel for vocoder shape: {mel.shape}")
    
    # HiFi-GAN decode
    with torch.no_grad():
        wav_reconstructed = hifi_gan.decode_batch(mel)
    
    print(f"  Reconstructed wav shape: {wav_reconstructed.shape}")
    
    # Mentés
    output_path = output_dir / f"dataset_sample_{i}_reconstructed.wav"
    torchaudio.save(
        str(output_path),
        wav_reconstructed.squeeze(0).cpu(),  # [1, samples] -> [samples] -> unsqueeze -> [1, samples]
        16000
    )
    
    print(f"  ✓ Mentve: {output_path.name}")

print("\n" + "="*60)
print("✅ KÉSZ! 5 minta rekonstruálva.")
print("="*60)
print(f"\nRekonstruált wav fájlok helye:")
print(f"  {output_dir}")
print("\n📝 KÖVETKEZŐ LÉPÉS:")
print("  1. Hallgasd meg a rekonstruált wav fájlokat")
print("  2. Ellenőrizd hogy érthető beszéd-e")
print("  3. Ha jó minőségű → dataset mel OK, vocoder OK ✅")
print("  4. Ha rossz minőségű → mel paraméterek nem egyeznek ❌")
