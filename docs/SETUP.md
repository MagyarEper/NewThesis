# Hungarian Dysarthria TTS Training Setup

Ez a dokumentum leírja, hogyan kell beállítani és elindítani a GRAD-TTS traininget egy új gépen.

## 1. Előfeltételek

- CUDA-capable GPU (minimum 16GB VRAM ajánlott)
- Python 3.9+
- Conda vagy venv

## 2. Repository clone és environment setup

```bash
# Clone a repository
git clone https://github.com/MagyarEper/NewThesis.git
cd NewThesis

# Conda environment létrehozása
conda create -n grad-tts python=3.9 -y
conda activate grad-tts

# Alapvető csomagok
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn tqdm
pip install librosa soundfile
pip install openpyxl  # xlsx olvasáshoz
pip install tensorboard
pip install speechbrain

# GRAD-TTS specifikus
cd Grad-TTS
pip install -r requirements.txt
cd ..
```

## 3. Adatok előkészítése

### 3.1. Eredeti WAV fájlok és transcript

A Hungarian Dysarthria Database fájljai szükségesek:
- `dsyarthria-hun_transcripts.xlsx` - transkripciók
- Eredeti WAV fájlok (22.05 kHz vagy 44.1 kHz)

### 3.2. Manifest létrehozása

```bash
python seged_szkriptek/manifest.py \
  --xlsx /path/to/dsyarthria-hun_transcripts.xlsx \
  --output manifest.txt \
  --wav-dir wavs_16khz
```

Ez létrehozza a `manifest.txt` fájlt formátumban:
```
/path/to/wavs_16khz/C_001_0001_NULL_NULL_A.wav|A szöveg zárójelek nélkül|0
```

### 3.3. **FONTOS: WAV fájlok 16 kHz-re konvertálása**

**KRITIKUS LÉPÉS:** A GRAD-TTS SpeechBrain HiFi-GAN vocoderrel dolgozik, ami 16 kHz-es bemenetet vár.

```bash
# Összes WAV fájl resample-lése 16 kHz-re
python seged_szkriptek/wav_converter.py \
  --manifest manifest.txt \
  --input-dir /path/to/original_wavs/ \
  --output-dir wavs_16khz
```

**Paraméterek:**
- `--manifest`: A manifest.txt fájl (létrehozva az előző lépésben)
- `--input-dir`: Az eredeti WAV fájlok könyvtára (22.05 vagy 44.1 kHz)
- `--output-dir`: A 16 kHz-es output könyvtár (default: `wavs_16khz`)

**Megjegyzés:** 
- A script automatikusan átugorja a már létező fájlokat
- Körülbelül 10,000 fájl esetén ~10-15 perc
- Győződj meg róla, hogy elegendő tárhely van (~3-4 GB)

### 3.4. WAV fájlok ellenőrzése

```bash
python seged_szkriptek/check_files.py --manifest manifest.txt
```

Ha vannak hiányzó fájlok, a script automatikusan eltávolítja őket a manifestből.

### 3.5. Train/Valid/Test split készítése

```bash
python seged_szkriptek/train_test_split.py \
  --manifest manifest.txt \
  --output-dir Grad-TTS/resources/filelists/libri-tts \
  --seed 42
```

Ez létrehozza:
- `train.txt` (~80%)
- `valid.txt` (~10%)
- `test.txt` (~10%)

A split beszélőnként történik (speaker-level split).

## 4. Training indítása

```bash
cd Grad-TTS

# Training indítása
export CUDA_VISIBLE_DEVICES=0
python train_multi_speaker.py
```

### Training paraméterek (params.py)

A jelenlegi konfiguráció 16GB VRAM-ra optimalizált:

```python
# Data params
n_spks = 39              # 39 beszélő
sample_rate = 16000      # 16 kHz (SpeechBrain HiFi-GAN miatt)
n_mels = 80
hop_length = 256

# Model params (16GB VRAM-ra optimalizált)
n_enc_channels = 128     # Encoder channels
filter_channels = 512    # Filter channels
filter_channels_dp = 192 # Duration predictor filter channels
n_enc_layers = 5         # Encoder layers
dec_dim = 48            # Decoder dimension

# Training params
batch_size = 12         # 16GB VRAM-ra optimalizált
n_epochs = 500          # Reális epoch szám
learning_rate = 1e-4
save_every = 25         # Checkpoint mentés 25 epochonként
```

## 5. TensorBoard monitoring

Másik terminálon:

```bash
conda activate grad-tts
cd NewThesis/Grad-TTS
tensorboard --logdir=logs/hungarian_dysarthria --port=6006
```

Nyisd meg böngészőben: `http://localhost:6006`

## 6. Checkpointok

A checkpointok itt lesznek mentve:
```
Grad-TTS/logs/hungarian_dysarthria/
  ├── grad_XXX.pt       # Epoch checkpoints
  └── events.out.*      # TensorBoard logs
```

## 7. Inference (tesztelés)

A training után próbáld ki a modellt:

```python
cd Grad-TTS
python inference.py
```

## Hibaelhárítás

### "CUDA out of memory"
- Csökkentsd a `batch_size`-t `params.py`-ban (pl. 12 → 8)

### "WAV file not found"
- Futtasd újra `check_files.py`-t
- Ellenőrizd hogy a `wavs_16khz/` könyvtárban vannak-e a fájlok

### "Sample rate mismatch"
- **Ellenőrizd hogy MINDEN WAV fájl 16 kHz-es!**
- Futtasd újra a `wav_converter.py`-t

### "Import error: speechbrain"
```bash
pip install speechbrain
```

## Hasznos parancsok

```bash
# Manifest statisztikák
wc -l manifest.txt
wc -l Grad-TTS/resources/filelists/libri-tts/*.txt

# WAV fájlok száma
ls wavs_16khz/*.wav | wc -l

# Sample rate ellenőrzés (random minta)
soxi wavs_16khz/C_001_0001_NULL_NULL_A.wav | grep "Sample Rate"

# GPU használat monitoring
watch -n 1 nvidia-smi
```

## Várható training idő

- **16GB VRAM GPU (pl. RTX 4080):**
  - ~667 iterations/epoch
  - ~3-5 perc/epoch
  - 500 epoch: **~25-40 óra** (1-2 nap)

## Fontos fájlok áttekintése

```
NewThesis/
├── manifest.txt                    # Fő manifest (10,002 utterances)
├── wavs_16khz/                     # 16 kHz WAV fájlok (FONTOS!)
├── seged_szkriptek/
│   ├── manifest.py                 # Manifest generálás
│   ├── check_files.py              # WAV ellenőrzés
│   ├── train_test_split.py         # Dataset split
│   └── wav_converter.py            # 16 kHz konverzió (FONTOS!)
└── Grad-TTS/
    ├── params.py                   # Training konfiguráció
    ├── train_multi_speaker.py      # Training script
    ├── data.py                     # SpeechBrain mel_spectogram
    ├── inference.py                # Inference script
    └── resources/filelists/libri-tts/
        ├── train.txt               # Training split
        ├── valid.txt               # Validation split
        └── test.txt                # Test split
```
