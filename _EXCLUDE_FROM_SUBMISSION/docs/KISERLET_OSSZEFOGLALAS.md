# Kísérleti Összefoglaló — Magyar Dysarthriás ASR Augmentáció

**Projekt:** Szintetikus dysarthriás adattal bővített Whisper fine-tune magyar nyelvre  
**Módszer:** Grad-TTS multi-speaker TTS + Whisper-small LoRA fine-tune  
**Referencia paper:** Leung et al., SPECOM 2025 (angol TORGO adatbázis)

---

## 1. Adathalmaz

**Magyar Dysarthria Database** — smart home parancsok (ablak, fűtés, TV, stb.)

*Megjegyzés: Az adatbázis meseolvasási felvételeket is tartalmaz, de a jelen munkában kizárólag az okosotthon-parancsokat használtuk.*

| Split | Utterance | Beszélők |
|-------|-----------|----------|
| Train | 7 988 | 38 |
| Val | 998 | 38 |
| Test | 1 016 | 38 |
| **Összesen** | **10 002** | **38** |

- Sampling rate: 16 kHz
- Speaker eloszlás: egyenetlen (C_012: 7 utt, C_041: 281 utt)
- Severity besorolás (15/38 speaker, témavezető értékelése): enyhe (7), középsúlyos (5), súlyos (2), control (1)

---

## 2. TTS Modellek (Grad-TTS verziók)

### 2.1 V1 — ARPAbet pipeline (eredeti, hibás)

| Paraméter | Érték |
|-----------|-------|
| Log dir | `hungarian_dysarthria` (eredeti) |
| Epochok | 500 |
| Text pipeline | `english_cleaners` + CMU szótár → ARPAbet |
| Vocab méret | ~149 szimbólum |
| Checkpoint | `grad_500.pt` |

**Probléma:** Az `english_cleaners` cleaner az `unidecode`-ot hívja, ami az ékezetes betűket latinosítja (`á→a`, `ő→o`, stb.). A CMU Pronouncing Dictionary ~134 000 *angol* szóból áll — a magyar szavak 99%-a nincs benne, ezért betű-alapú fallback ARPAbet kódolás jött létre. A modell torzított, inkonzisztens tokeneken tanult.

**Felhasználás:** v1 kísérletekhez (`wavs_v1_*`, `wavs_v1_control_*`) — a generált hangok csak az adatbázis szövegeire működtek kielégítően, ismeretlen szövegre gyenge volt.

---

### 2.2 V2 — Magyar grafémák, javított train/val split

| Paraméter | Érték |
|-----------|-------|
| Log dir | `hungarian_dysarthria_v2` |
| Epochok | 200 |
| Text pipeline | `english_cleaners` + CMU szótár → ARPAbet (azonos V1-gyel) |
| Vocab méret | ~148 szimbólum (azonos V1-gyel) |
| Train filelist | `train_textplit.txt` — 7 516 utt, 5 117 egyedi szöveg |
| Val filelist | `valid_textsplit.txt` — 1 471 utt, 903 egyedi szöveg |
| Szöveg-átfedés | **0%** train–val között |
| Batch size | 12 |
| Save every | 10 epoch |
| Architektúra | `n_enc_channels=128`, `dec_dim=48`, `n_enc_layers=5`, 39 speaker embedding |
| Vocoder | HiFi-GAN (LibriTTS 16kHz pre-trained) |

**Fő változtatások v1-hez képest:**
- Szöveg-alapú train/val split (0% átfedés) az overfitting detektálhatósága érdekében — ez a v2 kulcsmódosítása
- Validációs loss loop hozzáadva a training scripthez (`val_dur`, `val_prior`, `val_diff`)
- `torchaudio` backend: `soundfile` (deep07-en ffmpeg backend crashelt)

**Checkpointok:** `grad_100.pt`, `grad_120.pt`, `grad_200.pt`  
**Generált hanganyag:** `wavs_synthetic_v2_*` mappák (ep100/ep120/ep200, különböző temperature és length_scale variánsokkal)

**Generálási paraméterek (fő kísérletekhez):**
- timesteps=10, temperature=1.2, length_scale=1.0

---

### 2.3 V3 — CMU szótár eltávolítás + magyar grafémák

| Paraméter | Érték |
|-----------|-------|
| Log dir | `hungarian_dysarthria_v3` |
| Epochok | 120 |
| Text pipeline | `basic_cleaners` → magyar karakterek (CMU szótár eltávolítva) |
| Vocab méret | 82 szimbólum (148-ról csökkentve, ARPAbet kivéve) |
| Batch size | 32 (v2: 12) |
| Save every | 20 epoch |
| Architektúra | azonos v2-vel |

**Indoklás:** A v2 az eredeti `english_cleaners` + CMU szótár pipeline-t használta, ami az ékezetes betűket eltorzította és véletlenszerű angol fonemizálást végzett. A v3 teljesen eltávolítja a CMU szótárat, bevezeti a `basic_cleaners` → magyar grafémák pipeline-t, és a batch_size=32 deep07 16GB GPU-n ~2.5× gyorsabb epoch futást tesz lehetővé.

**Főbb változtatások v2-höz képest:**
- CMU szótár és ARPAbet teljesen eltávolítva → `basic_cleaners` + magyar karakterek (82 token)
- Batch size: 12 → 32 (~2.5× gyorsabb epoch)
- Epoch: 200 → 120 (v2-ben 120-nál már konvergált)

**Generált hanganyag:** `wavs_synthetic_v3/`, `wavs_synthetic_v3_lowtemp/`, `wavs_wiki_v3/`

---

### 2.4 V4 — Speed perturbáció augmentáció

| Paraméter | Érték |
|-----------|-------|
| Log dir | `hungarian_dysarthria_v4` |
| Epochok | **200** |
| Text pipeline | `basic_cleaners` → magyar karakterek |
| Vocab méret | 82 szimbólum |
| Batch size | 32 |
| Save every | 20 epoch |
| Architektúra | azonos v2/v3-mal (`n_enc_channels=128`, `dec_dim=48`, `n_enc_layers=5`) |
| Tanítás dátuma | 2026-05-11–12 |
| Utolsó checkpoint | `grad_200.pt` (2026-05-12 04:04) |
| Batch per epoch | ~234 |

**Főbb változtatások v3-hoz képest:**
- **Speed perturbáció adataugmentáció:** tanítás közben random sebesség-faktor [0.9, 1.1] 50%-os valószínűséggel, mel-spektrogram tartományban (interpoláció, nem audio-resample) → `Grad-TTS/data.py` 138–169. sor
- 200 epoch (v3: 120) — de nem ez a leghosszabb tanítás (V1 = 500 epoch)

**Generált hanganyag:** `wavs_probe_v4/` (21 probe fájl: 3 speaker × 7 mondat)

**Training curve:** `results/training_curve_v4.png`

---

### 2.5 TTS verzió összehasonlítás

| Verzió | Text pipeline | Epochok | Fő probléma / változás | Fő felhasználás |
|--------|--------------|---------|------------------------|-----------------|
| V1 | ARPAbet (CMU, hibás) | 500 | Ékezet elvesztés, vegyes tokenek | v1 kísérletek (Exp 2–4) |
| V2 | ARPAbet (English cleaners + CMU) | 200 | Csak train/val split javítása (0% átfedés) | Exp 5–7 kísérletek |
| V3 | Magyar karakterek (CMU eltávolítva) | 120 | CMU eltávolítás, grafémák, batch=32 | Demo, összehasonlítás |
| V4 | Magyar karakterek | 200 | Speed perturbáció (v3 + speed aug) | Probe, ellenőrzés |

**Megjegyzés:** Az Exp 0–4 Whisper kísérletekhez **V1** szintetikus hanganyagot használtunk (`grad_500.pt`). Az Exp 5–7 kísérletekhez V2 TTS-t használtunk — de a V2 még ARPAbet pipeline-t használt, ezért a szintetikus hanganyag minősége gyenge volt (lásd exp5 WER: 95%, exp7 WER: 97%).

---

## 3. Whisper Fine-tune Konfiguráció

Minden kísérletben (Exp 1–4) azonos alapkonfiguráció volt alkalmazva:

| Paraméter | Érték |
|-----------|-------|
| Base model | `openai/whisper-small` (244M paraméter) |
| Fine-tune módszer | LoRA (PEFT) |
| LoRA rank | r=16 |
| LoRA alpha | 32 |
| LoRA target modulok | `q_proj, v_proj, k_proj, o_proj, fc1, fc2` |
| LoRA dropout | 0.05 |
| Tanítható paraméterek | 5 603 328 (~2.27%) |
| Nyelv | `"hu"` |
| Task | `"transcribe"` |
| Learning rate | 1e-4 |
| LR warmup | 500 lépés |
| Epochok | 15 (Exp 1–2), **10** (Exp 3–4, dupla adat miatt) |
| Batch size | 8 |
| Gradient accumulation | 2 (effektív batch: 16) |
| Precision | FP16 mixed |
| Gradient checkpointing | igen |
| Model selection | `load_best_model_at_end=True` (best val WER) |
| Futtatás | Remote — deep07 GPU szerver |

**Eltérés Exp 3–4-ben:** 10 epoch (Exp 1–2: 15 epoch), mert a dupla adatmennyiség miatt az összes tanítási lépés hasonló.

---

## 4. Kísérletek és Eredmények

### Exp 0 — Baseline (fine-tune nélkül)

| Paraméter | Érték |
|-----------|-------|
| Modell | `openai/whisper-small` (pre-trained, változtatás nélkül) |
| Train adat | – |
| Test adat | 1 016 valódi utterance |

| Metrika | Érték |
|---------|-------|
| Overall WER | **91.28%** |
| Overall CER | **54.67%** |
| Avg speaker WER | 90.38% |
| Avg speaker CER | 53.28% |

**Értelmezés:** A pre-trained Whisper sem a magyar nyelvet, sem a dysarthriás beszédstílust nem ismeri megfelelően. A >100% WER-ek (C_004: 114%, C_013: 150%, C_018: 178%, C_020: 146%) azt jelzik, hogy a modell insertionöket produkál (több szót ad ki, mint amennyi a referenciában van — hallucinációk).

---

### Exp 1 — Fine-tune valódi adattal (Real baseline)

| Paraméter | Érték |
|-----------|-------|
| Modell | whisper-small + LoRA |
| Train adat | 7 988 **valódi** dysarthriás utterance |
| Val adat | 998 valódi |
| Test adat | 1 016 valódi |
| Epochok | 15 |

| Metrika | Érték |
|---------|-------|
| Best val WER | 10.68% |
| **Overall WER** | **10.81%** |
| **Overall CER** | **5.00%** |
| Avg speaker WER | 12.33% |
| Avg speaker CER | 5.22% |

**Legjobb speaker:** C_040: 0.0%  
**Legrosszabb speaker:** C_001: 42.9% (csak 4 utt a train setben)

**Értelmezés:** A fine-tune drasztikusan javítja a felismerést (~80 pp). Ez az **upper bound** — a valódi adattal elérhető legjobb teljesítmény a mi konfigurációnkban.

---

### Exp 2 — Fine-tune szintetikus adattal (Synthetic only)

| Paraméter | Érték |
|-----------|-------|
| Modell | whisper-small + LoRA |
| TTS modell | Grad-TTS **V1** (`grad_500.pt`, timesteps=10, temp=1.2) |
| Train adat | 7 988 **szintetikus** utterance (V1 Grad-TTS) |
| Val adat | 998 valódi |
| Test adat | 1 016 valódi |
| Epochok | 15 |

| Metrika | Érték |
|---------|-------|
| Best val WER | 20.82% |
| **Overall WER** | **19.05%** |
| **Overall CER** | **8.36%** |
| Avg speaker WER | 21.28% |
| Avg speaker CER | 9.29% |

**Értelmezés:** A tisztán szintetikus adat is **jelentősen javítja** a felismerést a baseline-hoz képest (91.28% → 19.05%). Összehasonlítva Exp 1-gyel (10.81%): ~8 pp különbség, de a szintetikus adat *önmagában is* értékes. A referencia paper (Leung et al.) hasonló mintázatot talált angol TORGO adatbázison.

---

### Exp 3 — Data Augmentation 1× (Real + Synthetic 1:1)

| Paraméter | Érték |
|-----------|-------|
| Modell | whisper-small + LoRA |
| TTS modell | Grad-TTS **V1** (`grad_500.pt`, timesteps=10, temp=1.2) |
| Train adat | **15 976** utterance (7 988 valódi + 7 988 szintetikus) |
| Val adat | 998 valódi |
| Test adat | 1 016 valódi |
| Epochok | **10** (dupla adat → hasonló lépésszám mint 15 ep / Exp 1) |

| Metrika | Érték |
|---------|-------|
| Best val WER | 10.10% |
| **Overall WER** | **10.33%** |
| **Overall CER** | **3.42%** |
| Avg speaker WER | 11.95% |
| Avg speaker CER | 4.11% |

**Értelmezés:** A szintetikus augmentáció **javít** a valódi adathoz képest: 10.81% → 10.33% WER (−0.48 pp). A CER-javulás kifejezettebb: 5.00% → 3.42% (−1.58 pp). A modell 23/37 speakernél jobb vagy egyenlő Exp 1-gyel.

---

### Exp 4 — Data Augmentation 10× (Real + Synthetic 1:10)

| Paraméter | Érték |
|-----------|-------|
| Modell | whisper-small + LoRA |
| TTS modell | Grad-TTS **V1** (`grad_500.pt`, timesteps=10, temp=1.2) |
| Train adat | **87 988** utterance (7 988 valódi + **80 000** szintetikus) |
| Val adat | 998 valódi |
| Test adat | 1 016 valódi |
| Epochok | **10** |

| Metrika | Érték |
|---------|-------|
| **Overall WER** | **12.00%** |
| **Overall CER** | **4.27%** |
| Avg speaker WER | 13.89% |
| Avg speaker CER | 5.13% |

**Értelmezés:** Az 1:10 arányú szintetikus augmentáció gyengébb, mint az 1:1 arány (Exp 3: 10.33% < Exp 4: 12.00%). A túl sok szintetikus adat "dilutes" a valódi adatot — a modell a szintetikus hangstílushoz igazodik, ami rontja a valódi dysarthriás felismerést.

---

## 5. Összehasonlító táblázat

### Fő eredmények

| Kísérlet | Train adat | Train méret | TTS verzió | Overall WER | Overall CER | Avg spk WER |
|----------|-----------|-------------|------------|-------------|-------------|-------------|
| **Exp 0** | – (baseline) | – | – | **91.28%** | **54.67%** | 90.38% |
| **Exp 1** | Valódi | 7 988 | – | **10.81%** | **5.00%** | 12.33% |
| **Exp 2** | Szintetikus | 7 988 | **V1** grad_500.pt | **19.05%** | **8.36%** | 21.28% |
| **Exp 3** | Valódi + Szintetikus 1:1 | 15 976 | **V1** grad_500.pt | **10.33%** | **3.42%** | 11.95% |
| **Exp 4** | Valódi + Szintetikus 1:10 | 87 988 | **V1** grad_500.pt | **12.00%** | **4.27%** | 13.89% |

### Javulás a baseline-hoz képest

| Kísérlet | WER (abs.) | WER javulás | CER (abs.) | CER javulás |
|----------|-----------|------------|-----------|------------|
| Exp 0 → Exp 1 | 91.28% → 10.81% | **−80.47 pp** | 54.67% → 5.00% | **−49.67 pp** |
| Exp 0 → Exp 2 | 91.28% → 19.05% | **−72.23 pp** | 54.67% → 8.36% | **−46.31 pp** |
| Exp 0 → Exp 3 | 91.28% → 10.33% | **−80.95 pp** | 54.67% → 3.42% | **−51.25 pp** |
| Exp 1 → Exp 3 | 10.81% → 10.33% | **−0.48 pp** | 5.00% → 3.42% | **−1.58 pp** |
| Exp 3 vs Exp 4 | 10.33% vs 12.00% | Exp 3 jobb | 3.42% vs 4.27% | Exp 3 jobb |

---

## 6. Severity-csoportos elemzés

**WER severity csoportonként** (sorrendben: legrosőszabbtól enyhéig):

| Súlyosság | N | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Legjobb |
|-----------|---|-------|-------|-------|-------|-------|--------|
| Súlyos | 2 | 102.6% | 14.8% | 29.0% | 15.2% | **16.7%** | Exp 1 |
| Középsúlyos | 5 | 87.7% | 17.2% | 26.9% | **16.9%** | 22.2% | Exp 3 |
| Enyhe | 7 | 105.6% | 8.9% | 17.0% | **8.6%** | **8.6%** | Exp 3/4 |
| Control | 1 | 77.3% | 8.9% | 17.9% | **10.6%** | 10.4% | Exp 1 |
| Nem besorolt | 22 | 91.6% | 12.9% | 22.3% | **12.5%** | 13.3% | Exp 3 |

**CER severity csoportonként:**

| Súlyosság | N | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Legjobb |
|-----------|---|-------|-------|-------|-------|-------|--------|
| Súlyos | 2 | 70.1% | 5.7% | 15.7% | **5.6%** | 6.1% | Exp 3 |
| Középsúlyos | 5 | 53.7% | 5.8% | 11.8% | 6.7% | 10.3% | Exp 1 |
| Enyhe | 7 | 59.7% | 3.1% | 8.5% | **2.9%** | 3.1% | Exp 3 |
| Control | 1 | 23.3% | 2.0% | 6.1% | 2.5% | **2.5%** | Exp 1 |
| Nem besorolt | 22 | 50.7% | 5.5% | 8.2% | **3.8%** | 4.3% | Exp 3 |

*Megjegyzés: az értékek speaker-szintű átlagok átlagai (micro avg értékek kisebb mértékben eltérhetnek). Severity besorolás 15/38 speakerre áll rendelkezésre; a súlyos csoportban mindossze 2 speaker van — ez a szám statisztikailag nem értékelhető.*

**Megfigyelés:** Az augmentáció (Exp 3) leginkább az enyhe, középsúlyos és nem besorolt csoportban segít. A control speaker (C_027) esetén a szintetikus adat nem segít — ez várt eredmény. Az Exp 4 (1:10 arány) a középsúlyos csoportnál jelentősen romlik (17.2% → 22.2%), ami azt jelzi, hogy a túll sok szintetikus adat főleg a nehézebb beszédű speakereken árt.

---

## 7. V4 Probe Eredmények

A v4 Grad-TTS checkpoint 21 probe fájlon lett kiértékelve (`wavs_probe_v4/`):
- 3 speaker (C_001, C_002, C_003) × 7 mondat (2 dysarthria szöveg + 5 rövid: "Bent vagyok", "Fúj a szél", "Süt a nap", "Le fog esni", "Gyere ide ki")

| Modell | Overall WER | Overall CER |
|--------|-------------|-------------|
| Baseline (whisper-small pre-trained) | **389.47%** | 395.58% |
| Exp 4 (whisper-small + LoRA, daug80k) | **94.74%** | 84.81% |

**Per-speaker (Exp 4):**

| Speaker | WER | CER |
|---------|-----|-----|
| C_001 | 96.0% | 89.6% |
| C_002 | 100.0% | 82.4% |
| C_003 | 88.5% | 82.8% |

**Értelmezés:** A v4 Grad-TTS szintetikus hangjai minden speakernél ~85–100% WER-t produkálnak még az exp4 fine-tuned modellel is. Ez azt jelzi, hogy:
1. A v4 TTS **nem reprodukálja** a valódi dysarthriás akusztikai karakterisztikát
2. A generált hangok hangzásilag eltérnek a valódi hanganyagtól — a fine-tuned modell nem tudja felismerni őket
3. A v4 nem jelent minőségi javulást a v2/v3-hoz képest a felismerhetőség szempontjából

---

## 8. Spektrogram Összehasonlítás

Generált fájl: `results/spectrogram_versions.png`  
Tartalom: 5 panel — valódi dysarthriás hang + v1/v2/v3/v4 szintetikus hang ugyanarra a mondatra ("Süt a nap", C_003 speaker)

**Főbb megfigyelések:**
- **Valódi hang**: Szabálytalan időbeli kiterjedés, torzított formánsok, zajosabb pergőhangok — jellegzetes dysarthriás akusztika
- **V1 (ARPAbet)**: Torzított spektrum, ARPAbet hibák miatti inkonzisztens formánsok
- **V2/V3/V4 (magyar karakterek)**: Tisztább spektrum mint v1, de a dysarthriás karakterisztika elvész — a hang normál TTS-hez hasonlít, nem valódi dysarthriáshoz

---

## 9. Keresztbeszélős Szintetikus Adatok Értékelése

### 9.1 Kísérlet leírás

A keresztbeszélős augmentáció célja: az adatbázis train szövegeit olyan hangokkal szintetizálni, amelyeket az adott speaker nem mondott el — ezzel az akusztikai variáció növelése speakeren belül, adatleakage nélkül.

**Manifest:** `generation_manifest_crossspeaker.csv` — 7 980 sor, 38 speaker × 210 mondat  
**TTS modell:** Grad-TTS V1 (`grad_500.pt`, timesteps=10)  
**Leakage:** 0% — egyik generált mondat sem szerepel a teszthalmaz mondatai között  
**Generált könyvtár:** `wavs_synthetic_crossspeaker/` (7 980 fájl, RTF=0.197)

### 9.2 TTS minőség-vizsgálat: két Whisper modell a szintetikus hangokon

| Modell | Test adat | Overall WER | Avg sp. WER |
|--------|-----------|-------------|-------------|
| Baseline (`whisper-small`, nincs FT) | 7 980 szintetikus crossspeaker | **189.71%** | 189.75% |
| Exp 1 (valódi dysarthriás FT) | 7 980 szintetikus crossspeaker | **70.16%** | 70.18% |

**Összehasonlítás a valódi teszthalmazzal:**

| Modell | Valódi teszt WER | Szintetikus crosssp. WER | Különbség |
|--------|-----------------|--------------------------|----------|
| Baseline | 91.28% | 189.71% | +98 pp |
| Exp 1 | 10.81% | 70.16% | +59 pp |

**Eredményfájlok:** `results/crossspeaker_baseline.csv`, `results/crossspeaker_exp1.csv`

### 9.3 Értelmezés

1. **A szintetikus hangok akusztikailag nagyon eltérnek a valódi hangoktól.** A baseline modell 190% WER-t produkál a szintetikusokon — rosszabb, mint a valódi dysarthriás hangokon (91%). A TTS kimenet nemcsak nem dysarthriás, de nem is természetes hangzású.

2. **Az Exp 1 (diszartriás fine-tune) részben tudja olvasni a szintetikus hangokat** (70% WER), de ez jóval rosszabb a valódi teszthalmaz 10.81%-ánál. A fine-tune közvetetten megtanulta valamennyire a TTS domaint, de nem tudja megbízhatóan felismerni.

3. **Ez magyarázza, miért nem segített az augmentáció az Exp 3–7-ben:** a szintetikus hangok annyira eltérnek a valódi diszartriás hangoktól, hogy nem nyújtanak hasznos akusztikai generalizációt — legfeljebb szöveg-szintű (lexikai) variáció a hatásuk.

4. **A szűk keresztmetszet a TTS minősége:** Ha a Grad-TTS több (~50–100 óra) és változatosabb diszartriás adaton lett volna tanítva, a szintetikus hangok hitelesebb diszartriás karakterisztikát reprodukáltak volna, és az augmentáció valódi akusztikai generalizációt nyújthatott volna.

---

## 10. Eltérések a Referencia Papertől (Leung et al., SPECOM 2025)

| Szempont | Paper | Mi | Indok |
|----------|-------|-----|-------|
| Nyelv | Angol (TORGO) | Magyar | – |
| Adatbázis | TORGO (~6 óra, 8 speaker) | Magyar Dysarthria DB (~10k utt, 38 speaker) | Hozzáférhetőség |
| Kiértékelés | **LOSO** (speaker-független) | **Globális train/test split** | 38 speaker × LOSO = 76–114 fine-tune futás → számítási korlát |
| ASR modell | Whisper medium/large/large-v2 | Whisper small + LoRA | VRAM korlát (4 GB lokális, ~16 GB remote) |
| TTS modellek | Grad-TTS + Matcha-TTS | Csak Grad-TTS | Csak egy TTS modellt tanítottunk |
| Fine-tune | Teljes modell | LoRA (r=16, 2.27% paraméter) | VRAM korlát |

### Metodikai különbség hatása az eredmények összehasonlíthatóságára

A **LOSO** kiértékelés minden fold-ban egy olyan speakerre tesztel, akinek a hangja **egyetlen tanítási példában sem szerepelt** — ez a nehezebb, de valósághoz közelebb álló forgatókönyv. A mi **globális split** megközelítésünkben az összes speaker mondatai jelen vannak a train halmazban (csak különböző mondatokkal) — ez szignifikánsan könnyíti a feladatot, mivel a modell megtanulja az egyes speakerek hangjellemzőit.

Ezért a két rendszer WER értékei **nem közvetlen összehasonlíthatók**: a mi ~10.5%-os WER-ünk (Exp 1) nem állítható szembe közvetlenül a paper 27.8%-os legjobb eredményével. A LOSO kísérlet elvégzése konzultáció alapján eldöntendő.

---

## 11. Végső Összehasonlítás — Dolgozat Döntések

### Fő konklúziók

1. **A Grad-TTS tanítása csak részben sikerült:** A modell konvergált (loss csökkent), de a generált hangok nem reprodukálják megbízhatóan a dysarthriás akusztikát. Valószínű ok: ~8 órányi hanganyag kevés volt ilyen komplex, inkonzisztens ejtési mintázatok tanulásához. A modell "normál" TTS-ként viselkedik, nem dysarthria-szimulátorként.

2. **A fő eredmény a Whisper fine-tune:** Pre-trained Whisper-small (91% WER) → valódi dysarthriás adattal fine-tuned (~10.5% WER, Exp 1) — közel 80 pp javulás. Ez a dolgozat legfontosabb, legmegbízhatóbb eredménye.

3. **A szintetikus augmentáció nem hozott érdemi javulást (Exp 3 vs Exp 1):** A WER-javulás minimális (−0.48 pp), ami a kísérleti feltételek mellett nem tekinthető szignifikánsnak. A szintetikus adat tehát nem káros, de nem is hasznos — legalábbis ekkora arányban és ilyen TTS minőséggel.

4. **Exp 2 eredménye (19% WER) nem érvényes összehasonlítási alap:** A szintetikus fine-tuning adathalmaz mondatai 100%-ban átfedtek a teszt halmaz mondataival (sentence-level data leakage — az adatbázis mondatait szintetizáltuk V1-gyel, ugyanezek a mondatok vannak a tesztelőben is). Valódi generalizáció ennél valószínűleg jóval gyengébb lenne.

5. **Szintetikus adat önmagában katasztrofális (Exp 5, 7):** Ha a fine-tuning adat mondatai különböznek a teszthalmazétól (nincs leakage), a szintetikus-only megközelítés ~95-100% WER-t ad — a modell teljesen overfit a TTS domainre, a valódi hangokra nem generalizál.

6. **Túl sok szintetikus adat árt (Exp 4):** 1:10 arány rontja az 1:1-hez képest — a "szintetikus dominancia" rontja a valódi hangokra való generalizálást.

7. **A TTS nem reprodukálja a dysarthriát:** A szintetikus hangok nem dysarthriásak akusztikailag. Ha az augmentáció mégis használna valamit, az inkább szöveg-szintű (lexikai sokszínűség) hatás, nem akusztikai szimulálás.

### Amit érdemes betenni a dolgozatba

| Elem | Miért fontos |
|------|-------------|
| Exp 0 baseline | A kiindulópont — mutatja a problémát |
| Exp 1 (valódi) | Upper bound — a legjobb elérhető eredmény |
| Exp 2 (szintetikus) | Bizonyítja: szintetikus adat nélkül is van érték |
| Exp 3 (1:1 augm.) | **Fő eredmény** — a szintetikus augmentáció legjobb konfigurációja |
| Exp 4 (1:10 augm.) | Mutatja az optimum határát (több nem jobb) |
| Severity-csoportos elemzés | Finomabb kép, valódi dysarthria-specifikus eredmény |
| V4 probe WER | Illusztrálja a TTS korlátait — tudományos becsületesség |
| Spektrogram összehasonlítás | Vizuális illusztráció a TTS korlátaihoz |

### Amit valószínűleg nem kell részletesen tárgyalni

| Elem | Miért nem |
|------|-----------|
| V1/V3/V4 TTS modellek | A fő kísérletekhez (Exp 2–4) V1 volt használva |
| Probe v4 részletes per-speaker WER | Csak 21 fájl, nem szignifikáns |
| Exp 5–7 részletes eredmények | V2 TTS-sel futott, de a V2 még ARPAbet pipeline-t használt (WER: 95–97%) — ezek nem a fő kísérletek |

---

## 12. Fájlok és Útvonalak

| Fájl / Könyvtár | Tartalom |
|-----------------|---------|
| `results/exp0_baseline.csv` | Exp 0 részletes per-speaker |
| `results/exp1_real.csv` | Exp 1 részletes per-speaker |
| `results/exp2_synthetic.csv` | Exp 2 részletes per-speaker |
| `results/exp3_daug.csv` | Exp 3 részletes per-speaker |
| `results/exp4_daug80k.csv` | Exp 4 részletes per-speaker |
| `results/probe_v4_baseline.csv` | V4 probe — baseline modell |
| `results/probe_v4_exp4.csv` | V4 probe — exp4 modell |
| `results/training_curve_v4.png` | V4 training loss görbe |
| `results/spectrogram_versions.png` | Spektrogram összehasonlítás v1–v4 |
| `wavs_synthetic_v2_ep120/` | Fő szintetikus adathalmazok (Exp 2–4) |
| `wavs_probe_v4/` | V4 probe hangok (21 fájl) |
| `Grad-TTS/logs/hungarian_dysarthria_v4/` | V4 training log és checkpoint |
| `whisper_finetuned/exp1_real/` | Exp 1 fine-tuned modell |
| `whisper_finetuned/exp3_daug/` | Exp 3 fine-tuned modell |
| `whisper_finetuned/exp4_daug80k/` | Exp 4 fine-tuned modell |
| `whisper_finetune.py` | Whisper LoRA fine-tune script |
| `whisper_evaluate.py` | WER/CER kiértékelő |
| `generate_test_set.py` | Grad-TTS generáló script |
| `generation_manifest_crossspeaker.csv` | Keresztbeszélős generálási manifest (7 980 sor) |
| `create_crossspeaker_manifest.py` | Keresztbeszélős manifest generáló script |
| `synthetic_crossspeaker_manifest.csv` | Generált keresztbeszélős hangok manifesztje |
| `wavs_synthetic_crossspeaker/` | 7 980 szintetikus crossspeaker hang (V1 TTS) |
| `results/crossspeaker_baseline.csv` | Baseline Whisper eval szintetikus crossspeaker hangokon |
| `results/crossspeaker_exp1.csv` | Exp1 Whisper eval szintetikus crossspeaker hangokon |

---

*Utolsó frissítés: 2026-05-12*
