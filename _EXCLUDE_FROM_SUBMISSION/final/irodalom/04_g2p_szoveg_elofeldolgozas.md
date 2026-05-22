# Irodalom vázlat: G2P és szöveg-előfeldolgozás

**Beillesztés helye:** `Dolgozat/02_irodalom.md` — 2.2 TTS szekció, Grad-TTS pipeline leírása részeként  
**Méret:** ~250–400 szó  
**Prioritás:** KÖZEPES — közvetlenül indokolja a V1→V3 TTS fejlesztési iterációt

---

## Mit kell lefedni

### 2.x. Grafémáról fonémára (G2P) és szöveg-előfeldolgozás TTS-ben

**Szükséges tartalom:**

1. **Mit jelent a G2P? (1-2 mondat)**  
   A szöveg-hangzóvá alakítás első lépése a szöveges bemenetet fonetikai reprezentációvá (fonémaszekvenciává) alakítja. Ez a grapheme-to-phoneme (G2P) konverzió: a leírt szóból meghatározza a kiejtést.

2. **Miért nem triviális? (1 bekezdés)**  
   Az angolban és más nyelvekben a helyesírás és kiejtés nem feltétlenül egyezik meg (pl. „read" kiejtése kontextusfüggő). TTS rendszerekben két fő megközelítés létezik:
   - **Szótáralapú G2P:** előre definiált szótárból keresi ki a fonémákat (pl. CMU Pronouncing Dictionary — 133 000 angol szó ARPAbet fonémáival). Ismeretlen szavaknál (OOV) karakterenkénti visszatérés vagy hibás kiejtés.
   - **Szabályalapú / neurális G2P:** tanult modell (pl. espeak-ng, Phonetisaurus) az ismeretlen szavakra is ad kiejtést.

3. **Magyar nyelv sajátosságai (1 bekezdés)**  
   A magyar fonetikailag következetes: nagyrészt minden betű ugyanúgy ejtendő (eltérően az angoltól). Ezért:
   - A CMU szótár teljesen alkalmatlan: nem tartalmaz magyar szavakat, és az ARPAbet fonémakészlet sem fedi le a magyar hangokat (pl. ő, ű, cs, sz, zs)
   - Az espeak-ng nyílt forráskódú TTS motor natív magyar G2P-t tartalmaz, és IPA-alapú kimenettel dolgozik
   - A `basic_cleaners` megközelítés (V3 TTS): a szöveget karakterszinten kezeli, minden betűt közvetlenül a modellnek ad át — kerüli a G2P lépést, ami egyszerűbb, de elveszíti a fonetikai normalizálást

4. **Kapcsolat a dolgozattal (1 mondat)**  
   A V1 TTS CMU-alapú pipeline-ja ékezetes karaktereknél és ismeretlen szavaknál torzított tokeneket adott; a V3-ban a CMU elhagyásával és magyar grafémás (`basic_cleaners`) feldolgozással ez kiküszöbölhető volt — bár az akusztikai minőségre ez önmagában nem volt elegendő.

---

## Kulcsszavak kereséshez

- `grapheme to phoneme conversion TTS`
- `CMU Pronouncing Dictionary limitations`
- `espeak-ng multilingual TTS frontend`
- `text normalization TTS pipeline low-resource language`

## Ajánlott irodalom (keress rájuk!)

| Forrás | Mit kell megnézni |
|--------|------------------|
| **Lenzo (1998)** — *CMU Pronouncing Dictionary* | Az eredeti dokumentáció; hivatkoznod kell ha CMU-t megemlíted |
| **Bisani & Ney (2008)** — *"Joint-sequence models for grapheme-to-phoneme conversion", Speech Communication* | A neurális/szekvenica-modell alapú G2P referencia |
| **Dutoit (1997)** — *"An Introduction to Text-to-Speech Synthesis"* | Klasszikus könyv, a TTS frontend (szövegnormalizálás, G2P) leírása |
| **espeak-ng dokumentáció / Sproat et al.** | Ha az espeak-ng-t mint alternatívát megemlíted |
| **Szaszák György et al.** — magyar TTS kutatások (BMGE, SZTAKI) | Magyar G2P specifikus munkák — érdemes keresni rá, van ilyen |
| **Popov et al. (2021)** — *Grad-TTS* | Leírja, hogy a Grad-TTS melyik text frontend-et használja |

## Megjegyzés

Ez a szekció leginkább a limitációk és V1→V3 fejlesztési iteráció kontextusát adja meg. Nem kell mélyen belemenni a G2P algoritmikájába — elég, hogy az olvasó megértse: miért volt probléma a CMU szótár magyar szövegen, és mi volt az alternatíva.
