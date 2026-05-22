# V1/V2 vs. V3 Grad-TTS pipeline — részletes összehasonlítás

---

## 1. A két pipeline célja

Mindkét verzió ugyanazt a feladatot oldja meg: a **Magyar Dysarthria Adatbázison** (10 002 felvétel, 38 felhasznált speaker) betanít egy Grad-TTS multi-speaker TTS modellt, amely aztán szintetikus dysarthriás hangokat tud generálni tetszőleges szövegre és tetszőleges (tanult) hangszínre.

A különbség abban van, **hogyan konvertálják a szöveget a modell bemenetévé**, és **hogyan osztják fel az adatot** train/val halmazokra.

---

## 2. Szöveg → bemeneti reprezentáció

### V1/V2: ARPAbet fonémák (CMU Pronouncing Dictionary)

```
Magyar szöveg  →  Unidecode  →  angol betűk  →  CMU dict  →  ARPAbet  →  modell
  "ablakot"    →  "ablakot"  →  "ablakot"    →  (ismeretlen)  →  fallback betűk
  "Kapcsold"   →  "Kapcsold" →  "Kapcsold"   →  (ismeretlen)  →  ???
```

- **ARPAbet**: egy 39 foném alapú *angol* hangkészlet (pl. `AH0`, `B`, `L`, `EY1`, `K`, ...)
- **CMU Pronouncing Dictionary**: ~134 000 *angol* szó kiejtési szótára
- **Probléma**: A magyar szavak 99%-a nincs benne a CMU dict-ben
  - Ismeretlen szavaknál a kód betű-alapú fallbacket alkalmaz (pl. `K → K`, `A → AH0`)
  - Ezért a fonetizálás hibás, de *konzisztens* — minden egyes tanításnál ugyanúgy hibás
  - A modell megtanulja a hibás fonéma→hang leképezést, és az inferenciánál is ugyanúgy kell input
- **Vocab méret**: **149 szimbólum** (ARPAbet + blank + egyéb speciális tokenek)
- **Következmény**: A modell működik a *tanult mondatokon* (memorización keresztül), de ismeretlen szövegekre gyenge, mert a fonéma-szekvenecia torzult

### V3: Magyar grafémák (karakterek)

```
Magyar szöveg  →  lowercase + normalizálás  →  karakterek  →  modell
  "Ablakot"    →  "ablakot"                 →  a,b,l,a,k,o,t
  "Kapcsold"   →  "kapcsold"                →  k,a,p,c,s,o,l,d
```

- **Nincs szótár, nincs fonetizálás** — a modell közvetlenül karakterekből tanul
- **Vocab méret**: **83 szimbólum** (magyar ábécé összes karaktere + speciális tokenek)
- A modell maga tanulja meg a karakter→akusztika leképezést a hanganyagból (implicit fonetika)
- **Következmény**: Elméletileg helyesebb magyar szövegre, de több epochot igényel, mert a semmiből kell megtanulnia a fonológiát

---

## 3. Adatfelosztás (train/val split)

### Közös alap

Mindkét verzió ugyanabból a 10 002 felvételből indul ki (38 felhasznált speaker × ~263 utt átlag).
A fájlok `|`-szeparált formátumban vannak: `wav_elérési_út|szöveg|speaker_index`

### V1/V2 split: speaker-split (véletlenszerű)

**Fájlok:** `Grad-TTS/resources/filelists/train.txt`, `valid.txt`

| Halmaz | Utterance | Speaker |
|--------|-----------|---------|
| Train  | 7 998     | 39      |
| Val    | 1 000     | 37      |
| **Val arány** | **11,1%** | |

**Módszer:** Az utterance-ek véletlenszerűen kerültek train-be vagy val-ba, **speaker-szintű stratifikáció nélkül**.

**Szöveg átfedés train∩val:** **379 mondat**
- Ugyanaz a mondat (pl. `"kapcsold be a lámpát"`) szerepelhet train-ben C_007 hangján ÉS val-ban C_015 hangján
- A val loss azt méri: mennyire jól szintetizálja a *már látott szövegeket* ismerős (de esetleg más) hangokon

**Csak train-ben lévő speakerek:** C_012 (7 utt), C_034
- C_012-nek olyan kevés felvétele van, hogy valószínűleg véletlenül kimaradt a val-ból

### V3 split: text-split (szöveg alapján szétválasztott)

**Fájlok:** `Grad-TTS/resources/filelists/train_textplit.txt`, `valid_textsplit.txt`

| Halmaz | Utterance | Speaker |
|--------|-----------|---------|
| Train  | 7 524     | 39      |
| Val    | 1 474     | 38      |
| **Val arány** | **16,4%** | |

**Módszer:** A *mondatokat* (szövegeket) osztották szét, nem az utterance-eket. Ha egy mondat val-ba kerül, akkor **minden speaker összes felvétele** erről a mondatról val-ba kerül.

**Szöveg átfedés train∩val:** **1 mondat** (gyakorlatilag nulla)
- A val halmaz teljesen új, soha nem látott szövegeket tartalmaz
- A val loss azt méri: mennyire jól generalizál a modell *ismeretlen szövegekre* (de ismert hangokon)

**Csak train-ben lévő speakerek:** C_034
- C_012 itt már megjelenik a val-ban is (van elég mondatjából val-ba kerülőre)

### Összehasonlítás

| Jellemző | V1/V2 (speaker-split) | V3 (text-split) |
|----------|----------------------|-----------------|
| Split alapja | Véletlenszerű utterance | Mondat-szintű |
| Train utt | 7 998 | 7 524 |
| Val utt | 1 000 | 1 474 |
| Val arány | 11,1% | 16,4% |
| Szöveg átfedés | **379 mondat** | **1 mondat** |
| Val mit mér | Ismert szöveg szintézise | Ismeretlen szöveg szintézise |
| Val loss szintje | Alacsonyabb (könnyebb feladat) | Magasabb (nehezebb feladat) |
| Összes egyedi mondat | 6 016 | 6 016 |

**Fontos:** A val loss közvetlenül nem hasonlítható össze V1/V2 és V3 között, mert más nehézségű feladatot mér.

---

## 4. Modell architektúra különbségek

| Jellemző | V1/V2 | V3 |
|----------|-------|-----|
| Szöveges input | ARPAbet fonémák | Magyar karakterek |
| Vocab méret | 149 | 83 |
| n_spks | 39 | 39 |
| dec_dim | 48 | 48 |
| n_enc_channels | 192 (V1) / 128 (V2) | 128 |
| Tanítási epochok | 500 (V1), 200 (V2) | 120 → 200 (folyamatban) |
| Checkpoint | `grad_500.pt` (V1), `grad_120/200.pt` (V2) | `grad_120.pt` → `grad_200.pt` |

---

## 5. Checkpoint helyek

```
Grad-TTS/logs/
├── hungarian_dysarthria/          # V1
│   └── grad_500.pt                # 500 epoch, ARPAbet, 192 enc channel
├── hungarian_dysarthria_v2/       # V2
│   ├── grad_120.pt                # Legjobb checkpoint (WER=90.8% unseen)
│   └── grad_200.pt                # Végső checkpoint
└── hungarian_dysarthria_v3/       # V3
    └── grad_120.pt                # Jelenlegi (WER=100%, undertrained)
                                   # → deep07-en folytatódik 200 epochig
```

---

## 6. Inferencia pipeline

### V1 inferencia (generate_v1_samples.py)

```
CSV manifest (utt_id, text, speaker) 
    → text: Unidecode → CMU dict → ARPAbet fonémák
    → V1 vocab (148 szimbólum) indexelés
    → Grad-TTS encoder → MAS → diffúziós dekóder (mel-spektrogram)
    → HiFi-GAN vocoder (SpeechBrain, LibriTTS 16kHz)
    → WAV fájl
```

Külön script szükséges (`generate_v1_samples.py`), mert a V1 vocab/szimbólumkészlet rekonstruálva van a git history-ból — a jelenlegi `params.py` már V3-as.

### V2/V3 inferencia (generate_test_set.py)

```
CSV manifest (utt_id, text, speaker)
    → text: lowercase + Grad-TTS/text/ modul (V2: phonemizer, V3: karakter)
    → Grad-TTS encoder → MAS → diffúziós dekóder
    → HiFi-GAN vocoder
    → WAV fájl
```

Paraméterek: `--timesteps` (50 az optimális), `--temperature` (1.0 default), `--length-scale`

---

## 7. Kiértékelés

A TTS modellek minőségét **nem val loss-szal**, hanem **downstream WER-rel** értékeljük:

1. Generálunk hangokat a val/unseen szövegekre a betanított modellel
2. Whisper-small (openai/whisper-small, CPU) átiratozza a hangokat
3. WER/CER kiszámítása a referencia szöveggel szemben

**V2 legjobb eredmény** (ep120, t50, unseen szövegek):
- WER = 90.8%, CER = 71.2%

**V3 jelenlegi eredmény** (ep120, t50, unseen szövegek):
- WER = 100%, CER = 100% — teljesen érthetetlen (undertrained)

A V3 várható jobb eredmény ~200 epochnál, mert a loss görbe ep120-nál még konvergált.

---

## 8. Miért V3 a "helyesebb" megközelítés?

1. **Nincs angol szótártól való függés** — a CMU dict nem tartalmaz magyar szavakat
2. **Nincs information leakage a fonémán keresztül** — a V1/V2-ben az angol ejtési szabályok torzítják a representációt
3. **Text-split val** — a V3 val halmaz valóban *generalizálást* mér, nem memorizálást
4. **Hosszabb konvergencia** — ezért kell ~200 epoch a V3-nak szemben a V1 500 epochával (ahol a könnyebb feladat és az ismerős szövegek miatt a val loss hamarabb látszólag stagnált)

A V3 az adaptáció egyik kulcslépése: az eredeti Grad-TTS angliai kutatási kontextusból (*english_cleaners*, CMU dict) átültetés a valódi magyar fonetikai térbe.
