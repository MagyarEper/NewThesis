# Whisper Fine-tune Kísérlet Terv

## Cél
A referencia paper (SPECOM 2025, Leung et al.) módszertanának adaptálása magyar dysarthriás adatra:
**Megvizsgálni, hogy a Grad-TTS által generált szintetikus dysarthriás adatokkal javítható-e a Whisper ASR felismerése.**

---

## Adatok összefoglalása

| Split | Utterances | Speakers |
|-------|-----------|----------|
| Train | 7,988 | 38 |
| Val | 998 | 38 |
| Test | 1,016 | 38 |
| Összesen | 10,002 | 38 |

- **Nyelv**: Magyar
- **Domain**: Smart home parancsok (ablak, fűtés, TV, stb.)
- **Sampling rate**: 16 kHz
- **Speaker utterance range**: 7 (C_012) – 281 (C_041)

---

## Kísérletek

### Experiment 0: Baseline (pre-trained Whisper, fine-tune nélkül)
- **Cél**: Mekkora WER-t ad a Whisper magyar dysarthriás beszédre fine-tune nélkül?
- **Modell**: `openai/whisper-small` (244M param) – ez reális a 4GB GPU-nkhoz
- **Adat**: Test set valódi audió (1,016 utt)
- **Metrika**: WER (összesített + speaker-enkénti)
- **Megjegyzés**: Korábbi tesztünk whisper-base-zel 95.3% WER-t adott

### Experiment 1: Fine-tune valódi adattal (Real baseline)
- **Cél**: Mennyire javul a WER, ha valódi dysarthriás adattal fine-tune-olunk?
- **Modell**: `openai/whisper-small`
- **Train adat**: Train split valódi audió (7,988 utt)
- **Val adat**: Val split valódi audió (998 utt)
- **Test adat**: Test split valódi audió (1,016 utt)
- **Metrika**: WER

### Experiment 2: Fine-tune szintetikus adattal (Synthetic only)
- **Cél**: Elég-e tisztán szintetikus adat a fine-tune-hoz?
- **Modell**: `openai/whisper-small`
- **Train adat**: Train split szövegeiből Grad-TTS-sel generált audió (7,988 utt)
- **Val adat**: Val split valódi audió (998 utt)
- **Test adat**: Test split valódi audió (1,016 utt)
- **Metrika**: WER

### Experiment 3: Data Augmentation (Real + Synthetic)
- **Cél**: Javít-e a szintetikus adat hozzáadása a valódihoz?
- **Modell**: `openai/whisper-small`
- **Train adat**: Train split valódi + szintetikus (15,976 utt, 1:1 arány)
- **Val adat**: Val split valódi audió (998 utt)
- **Test adat**: Test split valódi audió (1,016 utt)
- **Metrika**: WER

---

## Összehasonlítás

| Kísérlet | Train adat | Várt kimenet |
|----------|-----------|-------------|
| Exp 0 | – (pre-trained) | ~95% WER (baseline) |
| Exp 1 | Valódi (7,988) | Jelentős javulás |
| Exp 2 | Szintetikus (7,988) | Paper szerint: hasonló v. jobb mint Exp 1 |
| Exp 3 | Valódi+Szint (15,976) | Paper szerint: legjobb |

---

## Szükséges scriptek

### 1. `generate_full_train_set.py` – Szintetikus train adat generálása
- Input: `train_manifest.csv` szövegei + speaker ID-k
- Modell: `logs/hungarian_dysarthria/grad_500.pt`
- Output: `synthetic_train_wavs/` mappa wav fájlokkal
- Output: `synthetic_train_manifest.csv`
- **Futtatás**: Remote GPU szerveren (szükséges: ~8000 inference)

### 2. `whisper_finetune.py` – Whisper fine-tune script
- Framework: HuggingFace Transformers + PEFT (LoRA) a memóriához
- Modell: `openai/whisper-small`
- Hyperparaméterek (grid search a paper alapján):
  - Learning rate: 1e-5
  - Warmup steps: 500
  - Epochs: 10-30
  - Batch size: 8 (gradient accumulation ha kell)
  - FP16 mixed precision
- Best checkpoint: legalacsonyabb val WER
- **Futtatás**: Remote GPU szerveren

### 3. `whisper_evaluate.py` – Kiértékelés
- Input: Fine-tuned modell + test manifest
- Output: WER (összesített, speaker-enkénti, esetleg severity-group ha van ilyen info)
- Whisper text normalizer használata (a paper alapján)

### 4. `combined_manifest.py` – Manifest fájlok összeállítása
- Valódi + szintetikus manifest kombinálása Exp 3-hoz

---

## Hardver stratégia

| Feladat | Hol? | Miért? |
|---------|------|--------|
| Szintetikus adat generálás | Remote GPU (ha elérhető) VAGY helyi 4GB GPU | Grad-TTS inference nem igényel sok VRAM |
| Whisper fine-tune | Remote GPU | whisper-small + LoRA ~6-8GB VRAM kell |
| Kiértékelés (inference) | Helyi GPU is elég | Inference kis VRAM |

**Helyi GPU**: RTX 3050 Ti, 4GB – Inferenciára elég, fine-tune-ra kevés
**Remote GPU**: 152.66.178.131 – Jelenleg nem elérhető, ellenőrizni kell

### LoRA alternatíva (4GB GPU-hoz)
Ha a remote nem elérhető, `whisper-small` + LoRA (r=8) + FP16 + gradient checkpointing
akár a helyi 4GB GPU-n is elfér. Ez ~2-3GB VRAM.

---

## Implementáció sorrendje

1. **Szintetikus train adat generálása** (`generate_full_train_set.py`)
   - Ez a leghosszabb lépés (~8000 utterance Grad-TTS inference)
   - A meglévő `generate_test_set.py` alapján

2. **Whisper baseline** (Exp 0) – gyors, helyi GPU-n is megy
   
3. **Whisper fine-tune script** (`whisper_finetune.py`)
   - HuggingFace Trainer API
   - LoRA adapter

4. **Exp 1** futtatás (valódi adat)

5. **Exp 2** futtatás (szintetikus adat)

6. **Exp 3** futtatás (kombinált adat)

7. **Kiértékelés és összehasonlítás**

---

## Eltérések a referencia papertől

| Paper | Mi | Megjegyzés |
|-------|-----|-----------|
| TORGO (angol, 8 speaker) | Magyar Dysarthria DB (38 speaker) | Más nyelv, több speaker |
| LOSO kiértékelés | Globális train/val/test split | LOSO túl drága 38 speakerre (38× fine-tune) |
| Whisper medium/large/large-v2 | Whisper small (+LoRA) | VRAM korlát miatt |
| Grad-TTS + Matcha-TTS | Csak Grad-TTS | Egy TTS modellünk van |
| ~6 óra adat | ~10k utterance (becsült 10+ óra) | Több adatunk van |

---

## Kérdések / Döntések

1. **LOSO vs. globális split?**
   - Paper: LOSO (minden speaker-re külön fine-tune) → 38 speaker esetén ez 38×3 = 114 fine-tune futás
   - **Javaslat**: Globális split (egyszerűbb, gyorsabb). Indoklás: 38 speaker sokkal több mint a paper 8 speakerje, a cross-speaker generalizáció amúgy is cél.

2. **Whisper méret?**
   - **Javaslat**: `whisper-small` (244M) + LoRA. Ha a remote elérhető, lehet `whisper-medium`-ot is próbálni.

3. **Magyar nyelv beállítás?**
   - Whisper fine-tune-nál `language="hu"`, `task="transcribe"` beállítás kell.
