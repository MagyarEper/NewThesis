# Kísérleti Eredmények — Whisper Fine-tune Magyar Dysarthriás Beszédre

## Összefoglaló

A projekt célja: **megvizsgálni, hogy a Grad-TTS által szintetikusan generált dysarthriás beszédadatokkal javítható-e a Whisper ASR felismerési pontossága magyar dysarthriás beszélőkön.**

A módszertan a SPECOM 2025 referencia paper (Leung et al.) adaptációja magyar nyelvre.

---

## Adathalmaz

**Magyar Dysarthria Database** — smart home parancsok (ablak, fűtés, TV, stb.)

| Split | Utterance-ek | Beszélők |
|-------|-------------|----------|
| Train | 7,988 | 38 |
| Val | 998 | 38 |
| Test | 1,016 | 38 |
| **Összesen** | **10,002** | **38** |

- **Nyelv**: Magyar
- **Sampling rate**: 16 kHz
- **Speaker eloszlás**: Egyenetlen — C_012: 7 utt, C_041: 281 utt

---

## TTS Modell (Szintetikus Adat Generálás)

- **Modell**: Grad-TTS (Popov et al., 2021)
- **Checkpoint**: `grad_500.pt` (500 epoch, 7.2M paraméter)
- **Architektúra**: dec_dim=48, n_enc_channels=128, n_enc_layers=5, 38 multi-speaker
- **Vocoder**: HiFi-GAN (LibriTTS 16kHz pre-trained)
- **Generálási paraméterek**: timesteps=10, temperature=1.2, length_scale=1.0

### TTS Minőség (korábban kiértékelve)

| Metrika | Érték | Megjegyzés |
|---------|-------|-----------|
| MCD | ~8-12 dB | Átlagos TTS minőség |
| STOI | ~0.10 | Alacsony, de a paper szerint nem korrelál WER-rel |
| ESTOI | ~0.08 | Alacsony |
| PPG-D | – | Gyenge-közepes korreláció WER-rel (ρ=0.25-0.50) |

> **Fontos megjegyzés**: A referencia paper szerint az objektív TTS metrikák (MCD, STOI, ESTOI) **nem korrelálnak** a downstream ASR WER-rel. Csak a PPG-D mutat gyenge-közepes korrelációt. Ezért az alacsony STOI nem jelenti feltétlenül, hogy a szintetikus adat használhatatlan ASR augmentációra.

---

## ASR Modell Konfiguráció

- **Base model**: `openai/whisper-small` (244M paraméter)
- **Fine-tune módszer**: LoRA (Low-Rank Adaptation)
  - Rank: r=16, alpha=32
  - Target modulok: q_proj, v_proj, k_proj, o_proj, fc1, fc2
  - Dropout: 0.05
  - Tanítható paraméterek: 5,603,328 (2.27%)
- **Nyelv**: `language="hu"`, `task="transcribe"`
- **Optimalizáció**:
  - Learning rate: 1e-4
  - Warmup: 500 step
  - Epochs: 15
  - Batch size: 8 (gradient accumulation: 2 → effektív batch: 16)
  - FP16 mixed precision
  - Gradient checkpointing
- **Model selection**: Best validation WER (`load_best_model_at_end=True`)

---

## Kísérletek és Eredmények

### Experiment 0: Baseline (pre-trained Whisper, fine-tune nélkül)

- **Cél**: Mekkora WER-t ad a Whisper magyar dysarthriás beszédre fine-tune nélkül?
- **Modell**: `openai/whisper-small` (pre-trained)
- **Teszt adat**: 1,016 utterance (valódi)

| Metrika | Érték |
|---------|-------|
| **Overall WER** | **94.60%** |
| **Overall CER** | **58.53%** |
| Átlag speaker WER | 93.74% |
| Legjobb speaker | C_011: 54.9% |
| Legrosszabb speaker | C_018: 178.8% |

> A >100% WER azt jelenti, hogy a Whisper több szót hallucinál, mint amennyi a referenciában van (sok insertion).

<details>
<summary>Per-speaker WER (Exp 0)</summary>

| Speaker | WER | Utt. szám |
|---------|-----|-----------|
| C_001 | 85.7% | 4 |
| C_002 | 73.8% | 11 |
| C_003 | 90.9% | 14 |
| C_004 | 116.7% | 13 |
| C_005 | 61.1% | 12 |
| C_006 | 82.2% | 27 |
| C_007 | 72.4% | 27 |
| C_008 | 87.8% | 28 |
| C_009 | 95.1% | 14 |
| C_010 | 86.9% | 28 |
| C_011 | 54.9% | 17 |
| C_013 | 162.4% | 28 |
| C_014 | 96.2% | 23 |
| C_015 | 81.7% | 32 |
| C_016 | 95.0% | 29 |
| C_017 | 84.3% | 31 |
| C_018 | 178.8% | 29 |
| C_020 | 150.0% | 29 |
| C_021 | 89.0% | 35 |
| C_022 | 93.1% | 33 |
| C_023 | 105.4% | 29 |
| C_024 | 77.6% | 30 |
| C_025 | 78.5% | 32 |
| C_026 | 91.4% | 31 |
| C_027 | 72.6% | 32 |
| C_028 | 77.6% | 30 |
| C_029 | 75.0% | 32 |
| C_030 | 89.9% | 32 |
| C_031 | 97.3% | 32 |
| C_032 | 77.1% | 31 |
| C_033 | 106.8% | 35 |
| C_036 | 143.9% | 33 |
| C_037 | 93.8% | 35 |
| C_038 | 67.4% | 34 |
| C_040 | 82.2% | 34 |
| C_041 | 97.1% | 36 |
| C_042 | 97.1% | 34 |

</details>

---

### Experiment 1: Fine-tune valódi adattal (Real baseline)

- **Cél**: Mennyire javul a WER, ha valódi dysarthriás adattal fine-tune-olunk?
- **Train adat**: 7,988 utterance (valódi)
- **Val adat**: 998 utterance (valódi)
- **Teszt adat**: 1,016 utterance (valódi)

| Metrika | Érték |
|---------|-------|
| **Best Val WER** | **10.68%** |
| **Overall Test WER** | **11.26%** |
| **Overall Test CER** | **5.06%** |
| Átlag speaker WER | 12.76% |
| Legjobb speaker | C_040: 0.0% (!) |
| Legrosszabb speaker | C_001: 42.9% (4 utt) |

> **Javulás a baseline-hoz képest: 94.60% → 11.26% (test)** — a fine-tune drasztikusan javítja a felismerést.
> 
> Néhány speaker szinte tökéletesen felismert (C_040: 0%, C_036: 2.0%, C_037: 2.9%), míg a leggyengébb speakerek kevés train adattal rendelkeznek (C_001: 4 utt → 42.9%).

<details>
<summary>Per-speaker WER (Exp 1)</summary>

| Speaker | WER | Utt. szám |
|---------|-----|-----------|
| C_001 | 42.9% | 4 |
| C_002 | 18.5% | 11 |
| C_003 | 8.0% | 14 |
| C_004 | 40.5% | 13 |
| C_005 | 13.9% | 12 |
| C_006 | 4.9% | 27 |
| C_007 | 12.9% | 27 |
| C_008 | 17.7% | 28 |
| C_009 | 19.8% | 14 |
| C_010 | 13.7% | 28 |
| C_011 | 3.9% | 17 |
| C_013 | 7.6% | 28 |
| C_014 | 6.0% | 23 |
| C_015 | 12.0% | 32 |
| C_016 | 11.3% | 29 |
| C_017 | 7.9% | 31 |
| C_018 | 10.0% | 29 |
| C_020 | 21.7% | 29 |
| C_021 | 7.6% | 35 |
| C_022 | 19.6% | 33 |
| C_023 | 7.1% | 29 |
| C_024 | 8.6% | 30 |
| C_025 | 11.6% | 32 |
| C_026 | 14.6% | 31 |
| C_027 | 12.2% | 32 |
| C_028 | 17.2% | 30 |
| C_029 | 9.5% | 32 |
| C_030 | 11.6% | 32 |
| C_031 | 8.6% | 32 |
| C_032 | 5.1% | 31 |
| C_033 | 20.8% | 35 |
| C_036 | 2.0% | 33 |
| C_037 | 2.9% | 35 |
| C_038 | 7.3% | 34 |
| C_040 | 0.0% | 34 |
| C_041 | 16.1% | 36 |
| C_042 | 16.7% | 34 |

</details>

---

### Experiment 2: Fine-tune szintetikus adattal (Synthetic only)

- **Cél**: Elég-e tisztán szintetikus (Grad-TTS) adat a fine-tune-hoz?
- **Train adat**: 7,988 utterance (szintetikus — Grad-TTS, timesteps=10, temp=1.2)
- **Val adat**: 998 utterance (valódi)
- **Teszt adat**: 1,016 utterance (valódi)

| Metrika | Érték |
|---------|-------|
| **Best Val WER** | **20.82%** |
| **Overall Test WER** | **19.91%** |
| **Overall Test CER** | **8.54%** |
| Átlag speaker WER | 22.14% |
| Legjobb speaker | C_040: 5.9% |
| Legrosszabb speaker | C_001: 66.7% (4 utt) |

> **Javulás a baseline-hoz képest: 94.60% → 19.91%** — a tisztán szintetikus adat is jelentősen javítja a felismerést!
>
> Összehasonlítás Exp 1-gyel: a szintetikus adat önmagában gyengébb (19.91%) mint a valódi (11.26%), de a különbség mérsékelt (~8.7 pp). Ez arra utal, hogy a Grad-TTS szintetikus adatnak van értéke.

<details>
<summary>Per-speaker WER (Exp 2)</summary>

| Speaker | WER | Utt. szám |
|---------|-----|-----------|
| C_001 | 66.7% | 4 |
| C_002 | 30.8% | 11 |
| C_003 | 18.2% | 14 |
| C_004 | 57.1% | 13 |
| C_005 | 13.9% | 12 |
| C_006 | 7.4% | 27 |
| C_007 | 20.2% | 27 |
| C_008 | 23.8% | 28 |
| C_009 | 40.7% | 14 |
| C_010 | 23.8% | 28 |
| C_011 | 6.9% | 17 |
| C_013 | 21.0% | 28 |
| C_014 | 28.6% | 23 |
| C_015 | 36.6% | 32 |
| C_016 | 35.2% | 29 |
| C_017 | 9.6% | 31 |
| C_018 | 14.7% | 29 |
| C_020 | 23.9% | 29 |
| C_021 | 16.2% | 35 |
| C_022 | 32.3% | 33 |
| C_023 | 16.7% | 29 |
| C_024 | 21.3% | 30 |
| C_025 | 24.3% | 32 |
| C_026 | 15.1% | 31 |
| C_027 | 20.3% | 32 |
| C_028 | 18.4% | 30 |
| C_029 | 10.0% | 32 |
| C_030 | 18.5% | 32 |
| C_031 | 17.6% | 32 |
| C_032 | 10.9% | 31 |
| C_033 | 39.6% | 35 |
| C_036 | 7.7% | 33 |
| C_037 | 6.2% | 35 |
| C_038 | 8.8% | 34 |
| C_040 | 5.9% | 34 |
| C_041 | 22.0% | 36 |
| C_042 | 28.4% | 34 |

</details>

---

### Experiment 3: Data Augmentation (Real + Synthetic)

- **Cél**: Javít-e a szintetikus adat hozzáadása a valódihoz?
- **Train adat**: 15,976 utterance (7,988 valódi + 7,988 szintetikus)
- **Val adat**: 998 utterance (valódi)
- **Teszt adat**: 1,016 utterance (valódi)
- **Epochok**: 10 (a dupla adatmennyiség miatt — hasonló lépésszám mint Exp 1-2 15 epochpal)

| Metrika | Érték |
|---------|-------|
| **Best Val WER** | **10.10%** |
| **Overall Test WER** | **10.84%** |
| **Overall Test CER** | **3.49%** |
| Átlag speaker WER | 12.54% |
| Legjobb speaker | C_040: 0.0% |
| Legrosszabb speaker | C_001: 47.6% (4 utt) |

> **A fő eredmény: Exp 3 (10.84%) < Exp 1 (11.26%) — a szintetikus adataugmentáció javítja az ASR-t!**
>
> A javulás 0.42 százalékpont (94.60% → 11.26% → 10.84%). Bár a javulás mérsékelt, konzisztens:
> a kombinált modell 23/37 beszélőnél jobb vagy egyenlő az Exp 1-gyel.

<details>
<summary>Per-speaker WER (Exp 3)</summary>

| Speaker | WER | Utt. szám |
|---------|-----|-----------|
| C_001 | 47.6% | 4 |
| C_002 | 18.5% | 11 |
| C_003 | 6.8% | 14 |
| C_004 | 42.9% | 13 |
| C_005 | 15.3% | 12 |
| C_006 | 4.3% | 27 |
| C_007 | 13.5% | 27 |
| C_008 | 16.5% | 28 |
| C_009 | 17.3% | 14 |
| C_010 | 12.5% | 28 |
| C_011 | 3.9% | 17 |
| C_013 | 11.5% | 28 |
| C_014 | 3.0% | 23 |
| C_015 | 9.4% | 32 |
| C_016 | 14.5% | 29 |
| C_017 | 5.6% | 31 |
| C_018 | 10.0% | 29 |
| C_020 | 20.0% | 29 |
| C_021 | 6.7% | 35 |
| C_022 | 14.8% | 33 |
| C_023 | 7.7% | 29 |
| C_024 | 10.3% | 30 |
| C_025 | 15.5% | 32 |
| C_026 | 9.7% | 31 |
| C_027 | 13.7% | 32 |
| C_028 | 16.1% | 30 |
| C_029 | 9.5% | 32 |
| C_030 | 13.2% | 32 |
| C_031 | 8.0% | 32 |
| C_032 | 2.9% | 31 |
| C_033 | 22.2% | 35 |
| C_036 | 1.5% | 33 |
| C_037 | 1.9% | 35 |
| C_038 | 5.7% | 34 |
| C_040 | 0.0% | 34 |
| C_041 | 14.1% | 36 |
| C_042 | 17.2% | 34 |

</details>

---

## Összehasonlító táblázat

| Kísérlet | Train adat | Train méret | Val WER | Test WER | Test CER |
|----------|-----------|-------------|---------|----------|----------|
| **Exp 0** | – (pre-trained) | – | – | **94.60%** | **58.53%** |
| **Exp 1** | Valódi | 7,988 | 10.68% | **11.26%** | **5.06%** |
| **Exp 2** | Szintetikus | 7,988 | 20.82% | **19.91%** | **8.54%** |
| **Exp 3** | Valódi + Szintetikus | 15,976 | 10.10% | **10.84%** | **3.49%** |

> **Megjegyzés**: A CER-nél az Exp 3 javulás kifejezettebb: 5.06% → 3.49% (−1.57 pp), míg WER-nél 11.26% → 10.84% (−0.42 pp). Ez arra utal, hogy a szintetikus augmentáció elsősorban a karakterszintű pontosságot javítja — kevesebb betűhiba, de a szóhatárok hasonlóan jók.

---

## Severity-csoportos elemzés

A beszélők egy része a témavezető (Lívia) által súlyossági besorolást kapott az MSZNY2026 adatbázisban. A 38 beszélőből **15-nél** áll rendelkezésre besorolás:

| Súlyosság | N | Exp 0 | Exp 1 (valódi) | Exp 2 (szint.) | Exp 3 (daug) | Δ(1→3) |
|-----------|---|-------|----------------|----------------|--------------|--------|
| **Enyhe** | 7 | 109,4% | 8,3% | 16,7% | **8,1%** | **−0,2 pp** |
| **Középsúlyos** | 5 | 88,0% | 18,3% | 27,7% | **18,2%** | **−0,2 pp** |
| **Súlyos** | 2 | 102,0% | 14,7% | 28,6% | **15,1%** | +0,4 pp |
| Control | 1 | 72,6% | 12,2% | 20,3% | 13,7% | +1,5 pp |
| Nem besorolt | 22 | 90,3% | 12,8% | 22,1% | **12,4%** | **−0,4 pp** |

#### CER súlyossági csoportonként

| Súlyosság | N | Exp 0 | Exp 1 (valódi) | Exp 2 (szint.) | Exp 3 (daug) | Δ(1→3) |
|-----------|---|-------|----------------|----------------|--------------|--------|
| **Enyhe** | 7 | 68,9% | 3,1% | 8,7% | **2,8%** | **−0,3 pp** |
| **Középsúlyos** | 5 | 57,1% | 6,2% | 12,8% | 7,2% | +1,0 pp |
| **Súlyos** | 2 | 71,3% | 5,5% | 14,9% | **5,5%** | ±0,0 pp |
| Control | 1 | 26,9% | 3,2% | 7,5% | 3,8% | +0,6 pp |
| Nem besorolt | 22 | 53,9% | 5,9% | 8,7% | **3,8%** | **−2,0 pp** |

**Megfigyelések:**
- Az **enyhe** és **középsúlyos** csoportokban a data augmentáció (Exp 3) konzisztensen javít az Exp 1-hez képest (−0,2 pp mindkettőnél)
- A **súlyos** csoportban (mindössze 2 beszélő!) minimális romlás (+0,4 pp) — de a kis mintaméret miatt ez nem szignifikáns
- A Control speaker (C_027) esetében a szintetikus adat nem segít — ez várt eredmény, hiszen a kontroll beszélő beszéde normális
- A **nem besorolt** 22 beszélőnél is −0,4 pp javulás, ami az összesített trendet tükrözi
- A severity-grouped eredmények konzisztensek a SPECOM referencia paper megfigyelésével: a szintetikus augmentáció leginkább az enyhe-közepes súlyosságú beszélőknél hatékony

> **Megjegyzés**: A severity besorolás a témavezető szubjektív értékelése alapján készült, és csak 15/38 beszélőre áll rendelkezésre. A súlyos csoportban mindössze 2 beszélő van, ezért ennek a csoportnak az eredménye nem általánosítható.

<details>
<summary>Severity mapping (speaker → besorolás)</summary>

| Speaker | Severity |
|---------|----------|
| C_001 | Középsúlyos |
| C_003 | Középsúlyos |
| C_006 | Enyhe |
| C_010 | Enyhe |
| C_016 | Enyhe |
| C_018 | Enyhe |
| C_026 | Középsúlyos |
| C_027 | Control |
| C_029 | Középsúlyos |
| C_031 | Súlyos |
| C_033 | Súlyos |
| C_036 | Enyhe |
| C_040 | Enyhe |
| C_041 | Enyhe |
| C_042 | Középsúlyos |

*Forrás: MSZNY2026.xlsx, "Lívia" oszlop (HUN sheet)*

</details>

---

## Eltérések a referencia papertől (Leung et al., SPECOM 2025)

| Szempont | Paper | Mi |
|----------|-------|-----|
| Nyelv | Angol (TORGO) | Magyar |
| Beszélők | 8 | 38 |
| Kiértékelés | LOSO (Leave-One-Speaker-Out) | Globális train/val/test split |
| ASR modell | Whisper medium/large/large-v2 | Whisper small + LoRA |
| TTS modellek | Grad-TTS + Matcha-TTS | Csak Grad-TTS |
| Adat méret | ~6 óra | ~10k utterance (~10+ óra becsült) |
| Fine-tune | Teljes modell | LoRA (r=16, 2.27% paraméter) |

### Indoklások:
- **Globális split LOSO helyett**: 38 speakerre LOSO 38×3=114 fine-tune futást igényelne — nem kivitelezhető
- **Whisper small + LoRA**: VRAM korlát (4GB lokális, ~16GB remote) — a small+LoRA elfér
- **Csak Grad-TTS**: Egy TTS modellünk van, Matcha-TTS nincs betanítva

---

## Futtatási környezet

| Feladat | Gép | Specifikáció |
|---------|-----|-------------|
| Exp 0 baseline | Lokális | RTX 3050 Ti 4GB |
| Exp 1 fine-tune | Remote (deep07) | GPU szerver |
| TTS generálás | Remote (deep07) | GPU szerver |
| Exp 2-3 fine-tune | Remote (deep07) | GPU szerver |

---

## Fájlok

| Fájl | Leírás |
|------|--------|
| `whisper_finetune.py` | Whisper LoRA fine-tune script |
| `whisper_evaluate.py` | WER kiértékelő (overall + per-speaker) |
| `generate_test_set.py` | Grad-TTS szintetikus adat generáló (bármilyen split) |
| `run_whisper_experiments.sh` | Összes kísérlet futtatása egyben |
| `EXPERIMENT_PLAN.md` | Kísérlet terv |
| `results/exp0_baseline.csv` | Exp 0 részletes eredmények |
| `results/exp0_baseline_summary.txt` | Exp 0 összefoglaló |

---

*Utolsó frissítés: 2026-04-09*
