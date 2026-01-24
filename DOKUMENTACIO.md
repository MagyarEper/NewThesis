# Magyar Dysarthria TTS Projekt Dokumentáció

## 1. Projekt Áttekintés

Ez a projekt egy multi-speaker text-to-speech (TTS) rendszert valósít meg magyar dysarthria beszéd szintetizálására.

**Cél:** Képes legyen intelligibilis beszédet generálni magyar smart home parancsokból, 39 különböző dysarthria beszélő karakterisztikáit megőrizve.

## 2. Adathalmaz

### Hungarian Dysarthria Database
- **Forás:** Magyar dysarthria beszédadatbázis
- **Beszélők száma:** 39 beszélő (dysarthria különböző súlyossági fokozataival)
- **Utterance-ek száma:** 10,002 felvétel
- **Tartalom:** Smart home parancsok (pl. "Kapcsold ki a villanyt", "Nyisd ki az ablakot")
- **Eredeti mintavételi frekvencia:** 22.05 kHz / 44.1 kHz (változó)
- **Feldolgozott mintavételi frekvencia:** 16 kHz (SpeechBrain HiFi-GAN kompatibilitás miatt)

### Adatelőkészítés
1. **WAV konverzió 16 kHz-re:** Minden eredeti felvétel újra-mintavételezve 16 kHz-re
2. **Manifest generálás:** `manifest.txt` fájl létrehozása (pipe-separated: `wav_path|text|speaker_id`)
3. **Fájl validálás:** Hiányzó/hibás fájlok szűrése
4. **Train/Valid/Test split:** 80-10-10 arányú felosztás beszélőnként
   - Train: 7,988 utterance (38 beszélő)
   - Valid: 998 utterance (37 beszélő)
   - Test: 1,016 utterance (37 beszélő)

## 3. Modell Architektúra

### 3.1. Grad-TTS (Acoustic Model)

**Típus:** Gradient-based diffusion TTS model

**Komponensek:**
- **Text Encoder:** 
  - Phoneme reprezentáció (CMU dictionary alapján)
  - Channels: 128
  - Layers: 5
  - Filter channels: 512
  - Attention heads: 2
  
- **Duration Predictor:**
  - Filter channels: 192
  - Phoneme időtartam predikció
  
- **Diffusion Decoder:**
  - Mel-spectrogram generálás
  - Dimension: 48
  - Beta range: [0.05, 20.0]
  - PE scale: 1000

- **Speaker Embedding:**
  - 39 beszélő (speaker ID: 0-38)
  - Embedding dim: 64

**Paraméterek száma:** 7,210,200 (7.2M)

### 3.2. HiFi-GAN (Vocoder)

**Típus:** Neural vocoder (mel-spectrogram → waveform)

**Model:** SpeechBrain pretrained HiFi-GAN
- **Forrás:** `speechbrain/tts-hifigan-libritts-16kHz`
- **Mintavételi frekvencia:** 16 kHz
- **Input:** Mel-spectrogram (80 bins)
- **Output:** 16 kHz waveform

## 4. Training Paraméterek

### Optimalizálás
- **Batch size:** 12 (16GB VRAM-re optimalizálva)
- **Learning rate:** 1e-4
- **Optimizer:** Adam (implicit a train scriptben)
- **Epoch-ok száma:** 500
- **Training idő:** ~15 óra (NVIDIA RTX 3060 12GB)
- **Iteráció/epoch:** 665
- **Sebesség:** ~6.2 iteration/sec

### Mel-Spectrogram Paraméterek
- **n_fft:** 1024
- **n_mels:** 80
- **hop_length:** 256
- **win_length:** 1024
- **f_min:** 0 Hz
- **f_max:** 8000 Hz
- **Mel scale:** Slaney
- **Normalization:** Min-max energy norm

### Loss Funkciók
1. **Duration Loss:** Phoneme időtartam predikció hibája
2. **Prior Loss:** Text encoder és alignment loss
3. **Diffusion Loss:** Mel-spectrogram generálás hibája

## 5. Training Eredmények

### Konvergencia (Epoch 500)
- **Duration Loss:** 0.616
- **Prior Loss:** 1.201
- **Diffusion Loss:** 0.099

### Megfigyelések
- **Konvergencia:** Epoch ~400 körül stabilizálódtak a loss értékek
- **Plateau:** Epoch 481-500 között minimális változás (±0.005)
- **Következtetés:** A modell teljesen konvergált, további training nem javítana jelentősen

### GPU Használat
- **GPU:** NVIDIA GeForce RTX 3060 (12GB)
- **VRAM használat:** ~2.5 GB / 12 GB (20%)
- **GPU kihasználtság:** ~76%
- **Hőmérséklet:** 60°C (stabil)
- **Power draw:** 115W / 170W

## 6. Inference Paraméterek

### Optimális Beállítások
A sanity check tesztek alapján az alábbi paraméterek adták a legjobb eredményeket:

```bash
python sanity_check.py \
  --checkpoint logs/hungarian_dysarthria/grad_500.pt \
  --length-scale 1.0 \
  --temperature 1.2 \
  --timesteps 20
```

**Paraméterek magyarázata:**
- **length_scale:** 1.0 (eredeti: 0.91)
  - Időtartam szorzó: 1.0 = természetes hosszúság
  - Kisebb érték → rövidebb beszéd (levágott végek)
  - Nagyobb érték → hosszabb beszéd
  
- **temperature:** 1.2 (eredeti: 1.5)
  - Sampling hőmérséklet
  - Alacsonyabb → stabilabb, konzisztensebb
  - Magasabb → változatosabb, de zajosabb

- **timesteps:** 20 (eredeti: 10)
  - Diffusion reverse lépések száma
  - Több lépés → simább, jobb minőségű mel-spectrogram
  - RTF (Real-Time Factor): ~0.10 (még mindig real-time képes)

### Generálási Sebesség
- **Timesteps=10:** RTF ~0.06 (16x gyorsabb mint real-time)
- **Timesteps=20:** RTF ~0.10 (10x gyorsabb mint real-time)

## 7. Minőségi Értékelés (Sanity Check)

### Tesztelési Protokoll
- **Fix mondatok:** 10 smart home parancs
- **Tesztelt beszélők:** Speaker 0, 5, 10
- **Generált fájlok:** 30 WAV (3 speaker × 10 mondat)

### Eredmények
- **Audio minőség:** Stabil, nincs collapse vagy túlzott zaj
- **Intelligibilitás:** ~70% jó minőségű, ~30% változó/nehezen érthető
- **Megfigyelések:**
  - Néhány utterance nagyon jó minőségű
  - Néhány utterance alig érthető
  - Változékonyság oka: CMU dictionary angol phoneme-eket használ magyar szövegre
  
### Korlátok
1. **CMU Dictionary:** Angol phoneme-ek, nem magyar → pontatlan kiejtés
2. **Dysarthria változékonyság:** Egyes beszélők/phoneme-ek nehezebben modellezhetők
3. **Konvergált modell:** További training nem javít jelentősen

## 8. Fájlstruktúra

```
NewThesis/
├── Grad-TTS/                          # Fő TTS model
│   ├── model/                         # Model architektúra
│   │   ├── tts.py                     # Grad-TTS implementáció
│   │   ├── diffusion.py               # Diffusion komponens
│   │   ├── text_encoder.py            # Text encoder
│   │   └── monotonic_align/           # Duration alignment (Cython)
│   ├── data.py                        # Dataset betöltés
│   ├── params.py                      # Training paraméterek
│   ├── train_multi_speaker.py         # Multi-speaker training script
│   ├── sanity_check.py                # Inference teszt script
│   ├── inference.py                   # Általános inference
│   ├── logs/hungarian_dysarthria/     # Training logok és checkpointok
│   │   ├── grad_25.pt ... grad_500.pt # Model checkpointok (25 epochonként)
│   │   └── train.log                  # Loss értékek
│   └── resources/
│       ├── cmu_dictionary             # Phoneme dictionary
│       └── filelists/libri-tts/       # Train/valid/test splitek
├── seged_szkriptek/                   # Adatelőkészítő scriptek
│   ├── manifest.py                    # Manifest generálás
│   ├── check_files.py                 # Fájl validálás
│   ├── train_test_split.py            # Dataset split
│   └── wav_converter.py               # 16 kHz konverzió
├── wavs_16khz/                        # 16 kHz WAV fájlok (10,002 fájl)
├── manifest.txt                       # Dataset manifest
├── SETUP.md                           # Setup útmutató
└── DOKUMENTACIO.md                    # Ez a fájl
```

## 9. Reprodukálhatóság

### Environment
```bash
conda create -n grad-tts2 python=3.9
conda activate grad-tts2
pip install torch torchaudio
pip install speechbrain
pip install -r Grad-TTS/requirements.txt
```

### Cython Build
```bash
cd Grad-TTS/model/monotonic_align
python setup.py build_ext --inplace
```

### Training Újrafuttatása
```bash
cd Grad-TTS
python train_multi_speaker.py
```

### Inference Teszt
```bash
python sanity_check.py \
  --checkpoint logs/hungarian_dysarthria/grad_500.pt \
  --output-dir sanity_check_output \
  --length-scale 1.0 \
  --temperature 1.2 \
  --timesteps 20 \
  --speakers 0 5 10
```

## 10. Következő Lépések (2. Rész: Evaluáció)

**MEGJEGYZÉS:** Ez a dokumentáció az első fél (training) eredményeit tartalmazza. A második rész az alábbi elemeket fogja tartalmazni:

### Tervezett Evaluációs Metrikák
- [ ] **Objektív metrikák:**
  - Mel-Cepstral Distortion (MCD)
  - F0 RMSE
  - Duration accuracy
  
- [ ] **Szubjektív értékelés:**
  - Mean Opinion Score (MOS)
  - Intelligibility tests
  - Speaker similarity ratings

- [ ] **Beszélőnkénti/phoneme-enkénti analízis:**
  - Mely beszélők/phoneme-ek működnek a legjobban
  - Mely esetekben romlik a minőség
  - Dysarthria súlyosság vs. szintézis minőség korrelációja

- [ ] **Összehasonlítás baseline-nal:**
  - Eredeti dysarthria felvételek vs. szintetizált beszéd
  - Dysarthria vs. egészséges kontroll beszélők

### További Fejlesztési Lehetőségek
1. **Magyar phoneme dictionary:** CMU helyett magyar-specifikus phoneme mapping
2. **Több training adat:** További felvételek gyűjtése nehéz beszélőktől
3. **Fine-tuning:** Beszélőnkénti külön fine-tuning rossz esetekre
4. **Alternative vocoders:** HiFi-GAN+ vagy más modernebb vocoders tesztelése

---

**Utolsó frissítés:** 2026. január 24.  
**Projekt státusz:** Training befejezve, evaluáció folyamatban
