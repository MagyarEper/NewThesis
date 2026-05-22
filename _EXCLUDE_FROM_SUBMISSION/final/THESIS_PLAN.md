# Szakdolgozat Befejezési Terv
**Dátum:** 2026-05-17  
**Státusz:** Aktív

---

## Kulcsinformációk

### Verifikált WER/CER számok (ezeket kell használni mindenhol)

| Kísérlet | Overall WER | Overall CER | Avg sp. WER | Megjegyzés |
|----------|-------------|-------------|-------------|------------|
| Exp 0 – Baseline | 91.28% | 54.67% | 90.38% | |
| Exp 1 – Valódi | 10.81% | 5.00% | 12.33% | |
| Exp 2 – Szintetikus | 19.05% | 8.36% | 21.28% | ⚠️ LEAKAGE — jelölni kell |
| Exp 3 – DAug 1:1 | 10.33% | 3.42% | 11.95% | Fő augmentációs eredmény |
| Exp 4 – DAug 1:10 | 12.00% | 4.27% | 13.89% | Túl sok szintetikus adat ront |
| Cross-sp. Baseline | 189.71% | – | 189.75% | Szintetikus hangokon |
| Cross-sp. Exp1 | 70.16% | – | 70.18% | Szintetikus hangokon |

**Forrás:** `docs/KISERLET_OSSZEFOGLALAS.md`

### Az új konklúzió-narratíva

**Régi (helytelen):** „A TTS domain-specifikusan jól működött, külső hangokra gyenge."  
**Új (helyes):** „A Grad-TTS tanítása adatigény szempontjából elégtelen volt (~8 óra, 38 speaker). A szintetikus hangok nem reprodukálják a diszartriás akusztikai karakterisztikát, ezért az augmentáció nem hozott szignifikáns javulást."

**Bizonyítékok:**
- Cross-speaker eval: baseline 189.71% WER szintetikuson (vs 91.28% valódin) → a TTS kimenet akusztikailag rosszabb mint a diszartriás hang
- Exp1 fine-tuned modell 70.16% WER szintetikuson (vs 10.81% valódin) → még a fine-tuned modell is alig tudja olvasni
- Exp3 vs Exp1: 10.33% vs 10.81% (csak 0.48 pp javulás) → nem szignifikáns
- Exp4 (1:10 arány): 12.00% WER → több szintetikus adat ront
- Spektrogram-összehasonlítás → nincs diszartriás karakterisztika a szintetikusban

---

## Fázis 1 — 06_eredmenyek.md

**Prioritás: KRITIKUS — a számok most rosszak**

1. WER/CER számok cseréje a fenti táblázat szerint (régi: 94.60% → 91.28%, 11.26% → 10.81%, stb.)
2. Exp 2 elemzéséhez megjegyzés hozzáadása:
   > *„Megjegyzés: az Exp 2 eredménye adatleakage miatt nem tekinthető érvényes összehasonlítási alapnak — a szintetikus tanítóhalmaz mondatai 100%-ban megegyeztek a teszthalmaz mondataival (V1 TTS az adatbázis összes mondatát látta tanítás közben, köztük a tesztmondatokat is). A valódi generalizációs teljesítmény ennél várhatóan lényegesen gyengébb lenne."*
3. **Exp 4 szekció hozzáadása** (12.00% WER, 1:10 arány ront)
4. **Cross-speaker intelligibility evaluation szekció hozzáadása:**
   - Mit mértünk: 7 980 szintetikus hang (V1 TTS, train mondatok, teszt-átfedés nélkül) + 2 Whisper modell
   - Eredmény: baseline 189.71%, Exp1 70.16%
   - Értelmezés: a TTS kimenet akusztikailag nem reprodukálja a diszartriás karakterisztikát
5. **Severity-csoportos elemzés kitöltése** (forrás: `docs/KISERLET_OSSZEFOGLALAS.md` 6. szekció):

| Súlyosság | N | Exp 0 | Exp 1 | Exp 3 | Legjobb |
|-----------|---|-------|-------|-------|--------|
| Súlyos | 2 | 102.6% | 14.8% | 15.2% | Exp 1 |
| Középsúlyos | 5 | 87.7% | 17.2% | 16.9% | Exp 3 |
| Enyhe | 7 | 105.6% | 8.9% | 8.6% | Exp 3 |
| Control | 1 | 77.3% | 8.9% | 10.6% | Exp 1 |
| Nem besorolt | 22 | 91.6% | 12.9% | 12.5% | Exp 3 |

6. **Leung et al. összehasonlítás kitöltése:**
   - Módszertani különbség explicit: LOSO (paper) vs globális split (mi) → WER-ek nem közvetlen összehasonlíthatók
   - Paper legjobb: 27.8% WER (LOSO, whisper-large); mi: 10.81% (globális split, whisper-small) — a könnyebb kiértékelési protokoll magyarázza a különbséget

---

## Fázis 2 — 07_osszegzes.md

**Prioritás: KRITIKUS — tézispontok és limitációk nem tükrözik az új konklúziót**

### Tézispontok átírása

| # | Régi | Probléma | Új |
|---|------|----------|----|
| T1 | Grad-TTS alkalmazható, 38 speaker, RTF 0.053 | Félrevezető: technikai tény, de minőség kérdéses | **A fine-tune 91.28% → 10.81% WER-t hoz** — ~8k valódi felvétel elegendő a drámai javuláshoz |
| T2 | LoRA fine-tune drámai javulás | Helyes | Megtartjuk, számokat frissítjük |
| T3 | Szintetikus adat önállóan is hasznos (19.91%) | **Érvénytelen** — Exp2 leakage + Exp5/7 ~95% WER (no leakage) | **Törlés vagy átírás**: szintetikus-only leakage nélkül ~95% WER → valódi adat nélkül NEM hasznos |
| T4 | Augmentáció tovább javít (10.84% vs 11.26%) | Félrevezető — 0.48 pp nem szignifikáns | **Átírás**: a szintetikus augmentáció nem hozott szignifikáns javulást a TTS elégtelen minősége miatt |

### Limitációk átírása

| Régi limitáció | Teendő |
|---|---|
| CMU kiejtési szótár | **Pontosítás**: V1-specifikus probléma; V3 megoldotta (basic_cleaners, magyar grafémák); a javulás **elmaradt** → a szűk keresztmetszet nem a szótár volt |
| Egyetlen domén | Megtartjuk |
| Kis modell (whisper-small) | Megtartjuk |
| Szintetikus adat minősége (CV OOV) | **Teljes csere**: TTS elégtelen tanítóadatra visszavezethető gyenge minőség |

**Új limitáció szöveg (TTS adatigény):**
> A Grad-TTS szintetikus hangok minősége nem érte el a diszartriás augmentációhoz szükséges szintet. A cross-speaker intelligibility kísérlet alapján a szintetikus hangok WER-je baseline modellel 189.71% volt — rosszabb mint a valódi diszartriás teszthalmaz 91.28%-a. Az irodalomban megbízható speaker-adaptációhoz ~50–100 óra tanítóadat szükséges; jelen munkában ez ~8 óra volt 38 speakertől. A TTS fejlesztési iterációk (V1→V4) sem hozták meg az áttörést: a fő korlát az adatmennyiség volt, nem a szöveg-előfeldolgozási pipeline.

**TTS verzióhistória összefoglaló (a limitációkba, ~1 bekezdés):**
- V1: ARPAbet/CMU pipeline, 500 epoch — ékezet-torzítás, vegyes tokenek
- V2: szövegalapú train/val split, egyébként azonos V1-gyel — 200 epoch
- V3: CMU eltávolítva, magyar grafémák (basic_cleaners) — szöveg-szempontból javult, de diszartriás akusztikai reprodukció nem javult
- V4: speed perturbáció augmentáció — nincs mért javulás a felismerhetőségben
- Fő korlát mindvégig: ~8 óra tanítóadat 38 speakertől

### Jövőbeli irányok frissítése

- **Több diszartriás TTS tanítóadat** — ez a legfontosabb; ~50–100 óra várhatóan érdemi javulást hozna
- **Modern low-resource TTS** (XTTS, VALL-E) — kevesebb adattal is jobb speaker-adaptáció
- **LOSO kísérlet** — nem végeztük el (76 fine-tune futás, ~5–11 nap GPU-idő), de speaker-független általánosíthatósághoz szükséges lenne
- **Magyar G2P** (espeak-ng) — megmarad jövőbeli irányként
- **Más célnyelvek** — a módszertan alacsony erőforrású nyelvekre alkalmazható

---

## Fázis 3 — 04_modszerek.md

1. Grad-TTS szakaszhoz rövid **„TTS fejlesztési iterációk"** alszekció (~1 bekezdés, a fenti verzióhistória alapján)
2. Egyértelműsítés: **melyik verzióval futottak a kísérletek** (Exp 0–4: V1 grad_500.pt; Exp 5–7: V2)
3. Hardware specifikáció: deep07, 16 GB GPU

---

## Fázis 4 — 05_kiserlet.md

1. **Hardware specifikáció** kitöltése (deep07: ~16 GB GPU, nohup alapú futtatás)
2. **Exp 4 leírás** hozzáadása (80k szintetikus, 1:10 arány, 10 epoch)
3. **Cross-speaker augmentáció kísérlet** hozzáadása:
   - 7 980 szintetikus hang, 38 speaker × 210 mondat
   - V1 TTS, 0% teszt-átfedés, train mondatok cross-speaker generálva
   - Célja: TTS minőség-mérés, nem Whisper fine-tune
4. **LOSO TODO** → átalakítani: „nem végeztük el, metodikai korlátként jelölve"
5. Exp 2 leakage megjegyzés itt is

---

## Fázis 5 — 01_bevezetes.md

**Ellenőrizni és javítani:**
1. Baseline WER szám: ha 94.60% szerepel → javítani 91.28%-ra
2. Kutatási kérdések frissítése:
   - Q1 („Grad-TTS szintetizál-e akusztikailag hű diszartriás hangot?") → az eredmény: **nem**, ez a fő korlát — ezt már a bevezetésben fel kell vezetni 1-2 mondatban, hogy az olvasó tudja mire számítson
3. 1-2 mondat hozzáadása: a TTS tanítás adatigénye nem triviális → motiválja miért nem egyszerű a szintetikus augmentáció

---

## Fázis 6 — 02_irodalom.md

**Már megvan:** dizartria, adatbázisok, TTS történet, Grad-TTS, HiFi-GAN, Whisper, LoRA, SpecAugment, Leung et al.

**Javasolt kiegészítések (hiányzó building blockok):**

### 1. Mel-spektrogram és STFT (legfontosabb, jelenleg egyáltalán nincs)
- STFT → spektrogram → mel-szűrőbank → log-mel
- Ez a közös „kapocs" a TTS pipeline (Grad-TTS mel-t generál) és az ASR pipeline (Whisper log-mel-t kap) között
- Rövid alfejezet: 2.2.0 vagy 2.2 elejére illesztve

### 2. Transformer architektúra (2.3 ASR szekcióba)
- Encoder-decoder, self-attention alapelvek
- Whisper erre épül de a dolgozat nem magyarázza el
- 1 bekezdés elég, a részletek nem szükségesek

### 3. Speaker embedding (2.2 Grad-TTS szekcióba)
- x-vector / d-vector elvek röviden
- Hogyan kódolja a modell a hangszínt → hogyan kondicionálja a szintézist egy adott diszartriás hangszínre
- Jelenleg csak megemlítve van, nem kifejtve

### 4. G2P és szöveg-előfeldolgozás kihívásai magyarul (2.2 TTS szekcióba)
- CMU szótár angol-specifikus korlátai, OOV kezelés
- Magyar grafémák → basic_cleaners alternatíva
- Ez közvetlenül indokolja a V1→V3 fejlesztési iterációt, amit a limitációkban leírunk

---

## Döntések és kizárások

| Elem | Döntés |
|------|--------|
| Exp 2 | Megmarad de ⚠️ leakage-jelöléssel — módszertanilag tanulságos |
| LOSO | Nem végezzük el — csak korlát/jövőbeli irányként |
| V4 TTS | Csak limitációkban szerepel, nem önálló kísérletként |
| Severity-elemzés | Bekerül az eredményekbe |
| Exp 5–7 | Csak megemlítve (V2 TTS, ~95% WER), nem részletezve |
| Cross-speaker kísérlet | Bekerül mint TTS minőség-mérő kísérlet, nem augmentációs kísérletként |

---

## Fájlok

| Fájl | Teendő | Prioritás |
|------|--------|-----------|
| `Dolgozat/06_eredmenyek.md` | Számcsere + Exp4 + cross-speaker + severity + Leung | KRITIKUS |
| `Dolgozat/07_osszegzes.md` | Tézispontok + limitációk teljes átírás | KRITIKUS |
| `Dolgozat/04_modszerek.md` | TTS verzióhistória + hardware | KÖZEPES |
| `Dolgozat/05_kiserlet.md` | Exp4 + cross-speaker + LOSO kezelés | KÖZEPES |
| `Dolgozat/01_bevezetes.md` | WER szám + Q1 frissítés | ALACSONY |
| `Dolgozat/02_irodalom.md` | Mel-spektrogram + Transformer + Speaker emb. + G2P | ALACSONY |
| `docs/KISERLET_OSSZEFOGLALAS.md` | Forrás — nem módosítandó |
