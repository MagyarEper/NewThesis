# Dolgozat javítások nyilvántartása

## Státuszok: ⏳ Folyamatban | ✅ Kész | 🔴 Kritikus | 🟡 Fontos | 🟢 Kisebb

---

## Témavezető visszajelzés — 1. kör

### 🔴 J1 — "keresztbeszélős" fogalom kifejtése + átnevezése

**Probléma:** A "keresztbeszélős kiértékelés" fogalom nincs definiálva. A "forrás–célpont
beszélőpár" leírás félrevezető: a szöveg speaker-independent (standardizált mondatok,
amelyeket az adott beszélő nem vett fel), ezért nincs igazi forrás-oldal, csak célbeszélő.

**Érintett fájlok és sorok:**
- `thesis/chapters/06_eredmenyek.tex` L195, L198, L201–203, L208, L226
- `thesis/chapters/04_modszerek.tex` L112, L147
- `thesis/chapters/07_osszegzes.tex` L37, L41
- `thesis/chapters/absztrakt.tex` L14
- `thesis/chapters/tartalmi_osszefoglalo.tex` L40

**Teendő:**
1. Szekciócím: `Keresztbeszélős intelligibilitás-értékelés` → `Szintetikus hangok érthetőség-értékelése`
2. Első bekezdés újraírása (L201–203): eltávolítani a "forrás–célpont" leírást,
   helyette egyértelműen leírni: az adatbázis 37 dizartriás célbeszélőjére
   szintetizálták az általuk nem rögzített mondatokat (speaker-independent szövegek
   a tanítókészletből, teszthalmaztól eltérők; összesen 7 980 felvétel).
3. Minden `keresztbeszélős X` csere:

| Régi | Új |
|------|----|
| `keresztbeszélős intelligibilitás-értékelés` | `szintetikus érthetőség-értékelés` |
| `keresztbeszélős intelligibilitás-kísérlet` | `szintetikus érthetőség-kísérlet` |
| `keresztbeszélős baseline WER-t` | `a szintetikus hangokon baseline WER-t` |
| `keresztbeszélős kísérlet` | `szintetikus érthetőség-kísérlet` |
| `Keresztbeszélős kísérletekkel` | `Szintetikus érthetőség-kísérletekkel` |
| `keresztbeszélős intelligibilitás-eredményekkel` | `szintetikus érthetőségi eredményekkel` |
| `keresztbeszélős kiértékelés` | `szintetikus érthetőség-értékelés` |

**Státusz:** ⏳

---

### 🟡 J2 — "intelligibilitás" → "érthetőség" (hunglish szó)

**Probléma:** A témavezető kéri a magyar megfelelő következetes használatát.

**Érintett sorok:**
- `thesis/chapters/04_modszerek.tex` L147
- `thesis/chapters/06_eredmenyek.tex` L198, L208
- `thesis/chapters/07_osszegzes.tex` L41
- `thesis/chapters/tartalmi_osszefoglalo.tex` L40

(Megjegyzés: J1 és J2 összefonódik — egyszerre implementálható.)

**Státusz:** ⏳

---

### 🔴 J3 — Hiányzó gépi tanulás alapfogalmak szekció

**Probléma:** Nincs bevezető szekció az ML alapfogalmakról (gépi tanulási feladat,
tanítás, tanítóadatbázis, neurális háló architektúra stb.) — a TTS/ASR részek ezt
feltételezik, de soha nem definiálják.

**Elhelyezés:** `thesis/chapters/02_irodalom.tex`, a jelenlegi
`\section{Hangjelreprezentáció: STFT és mel-spektrogram}` ELÉ illesztve
(vagyis új 2.2 szekció lesz, a többi tolódik 2.3-ra stb.)

**Tartalom (kb. 1–1,5 oldal):**
- Gépi tanulási feladat: bemenet → kimenet leképezés, paraméteres modell, predikciós cél
- Tanítás: paraméterek hangolása veszteségfüggvény minimalizálásával (gradiens-ereszkedés)
- Tanítóadatbázis: annotált adatok halmaza; train / validation / test felosztás
- Neurális háló architektúra: rétegek, paraméterek (súlyok), aktivációs függvények, mélység
- Transzfer-tanulás és finomhangolás — különösen fontos, mert a Whisper és a Grad-TTS
  is előtanított modellekből indul a dolgozatban
- (Opcionálisan: túlillesztés / generalizáció, 1-2 mondat)

**Státusz:** ⏳

---

### 🔴 J4 — Téves/hiányos beszélő-kategorizáció

**Probléma (két szint):**
1. Belső ellentmondás: prezentáció-előkészítő fájlok szerint 34 dizartriás + 4 kontroll
   (C_007, C_008, C_011, C_012); tézis szerint 37 dizartriás + 1 kontroll (C_027) —
   ezek egymásnak ellentmondanak.
2. Kategorizálási alaphiba: 23 nem besorolt beszélőt a tézis hallgatólagosan dizartriásnak
   kezeli. Az MSZNY2026 Excel alapján ahol nincs kategória = "nem kategorizált", nem
   "default dizartriás"; köztük más jellegű beszédsérülésű is lehet.

**Érintett helyek:**
- `thesis/chapters/02_irodalom.tex` L64: "37 dizartriás és 1 kontroll"
- `thesis/chapters/03_adatok.tex` L8, L40: "37 dizartriás és 1 kontroll"
- `thesis/chapters/03_adatok.tex` L52: "sokszínű dizartriás ejtésprofilt foglal magában"
  → semlegesebb megfogalmazás
- `thesis/chapters/04_modszerek.tex` L31: "38 különböző dizartriás hangszínt"
  → "38 különböző hangszínt"
- `thesis/chapters/05_kiserlet.tex` L98: táblázatban "37 dizartriás + 1 kontroll"
- `thesis/chapters/06_eredmenyek.tex` L84, L124

**⚠️ BLOKKOLT: pontos kategóriaszámok kellenek az MSZNY2026 Excelből**

**Státusz:** ⏳

---

### 🔴 J5 — E/T 1 → E/1 (többes szám → egyes szám 1. személy, ~104 előfordulás)

**Probléma:** A dolgozat egyéni munkát ír le, de számos helyen többes szám első személyt
használ (tanítottuk, végeztük, mértük...), ami csapatmunkát sugall.

**NEM változtatandó (általános/generikus alany):**
- `01_bevezetes.tex`: "generálhatunk", "tanítunk be" (motivációs, hipotetikus mondatok)
- `02_irodalom.tex`: "szennyezünk", "megtanítjuk" (diffúziós modell általános leírása)

**Cserék fejezetek szerint:**

`thesis/chapters/absztrakt.tex`
- "vizsgáljuk" → "vizsgálom" (vagy: "E dolgozat azt vizsgálja...")
- "igazoltuk" → "igazoltam"

`thesis/chapters/tartalmi_osszefoglalo.tex`
- "generáltunk" → "generáltam"
- "finomhangoltuk" → "finomhangoltam"
- "hasonlítottuk" → "hasonlítottam"

`thesis/chapters/03_adatok.tex`
- "resample-ltük" → "resample-ltam" (2×)
- "osztottuk fel" → "osztottam fel"
- "soroltuk" → "soroltam"
- "osztottuk" → "osztottam"
- "mentettük" → "mentettem"
- "alakítottuk" → "alakítottam"
- "képeztük le" → "képeztem le"
- "rendeltük" → "rendeltem"
- "alkalmaztuk" → "alkalmaztam"
- "normalizáltuk" → "normalizáltam"

`thesis/chapters/04_modszerek.tex`
- "előkészítjük" → "előkészítem"
- "tanítottuk" → "tanítottam"
- "optimalizáltunk" → "optimalizáltam"
- "hangoltuk" → "hangoltam"
- "kaptuk" → "kaptam"
- "megállapítottuk" → "megállapítottam"
- "változtattuk" → "változtattam"
- "egyszerűsítettük" → "egyszerűsítettem"
- "elhagytuk" → "elhagytam"
- "társítottunk" → "társítottam"
- "megmértük" → "megmértem"
- "alkalmaztuk" → "alkalmaztam"
- "töltöttük" → "töltöttem"
- "választottuk" → "választottam"
- "adaptáltuk" → "adaptáltam"
- "állítottuk be" → "állítottam be"
- "végeztük" → "végeztem"
- "álltunk vissza" → "álltam vissza"
- "mértük" → "mértem"
- "kiszámítottuk" → "kiszámítottam"

`thesis/chapters/05_kiserlet.tex`
- "hasonlítottunk össze" → "hasonlítottam össze"
- "finomhangoltuk" → "finomhangoltam" (3×)
- "adtunk" → "adtam"
- "legyártottuk" → "legyártottam"
- "állítottunk elő" → "állítottam elő"
- "gyűjtöttük" → "gyűjtöttem"
- "szintetizáltunk" → "szintetizáltam"

`thesis/chapters/06_eredmenyek.tex`
- "soroltuk" → "soroltam"
- "végeztük" → "végeztem" (2×)
- "értékeltük" → "értékeltem" (2×)
- "futtattuk" → "futtattam"

`thesis/chapters/07_osszegzes.tex`
- "vizsgáltuk" → "vizsgáltam"
- "tanítottuk be" → "tanítottam be"
- "hasonlítottunk össze" → "hasonlítottam össze"
- "Igazoltuk" → "Igazoltam"
- "Megmutattuk" → "Megmutattam" (2×)
- "mértünk" → "mértem"
- "végeztünk" → "végeztem"
- "változtattunk" → "változtattam"

**Státusz:** ⏳

---

### 🔴 J6 — LoRA választásának explicit megindoklása hiányzik

**Probléma:** A bíráló szerint a LoRA alkalmazása "ad hoc döntésnek tűnik" — nincs
egyetlen összefüggő bekezdés, amely megmagyarázza, miért tértünk el Leung et al.
teljes finomhangolásától, és miért pont a LoRA a jobb megoldás.

**Ami már megvan (szétszórva, de nem összekötve):**
- `02_irodalom.tex` L216–228: LoRA-alszekció (overfitting korlátozott adaton)
- `05_kiserlet.tex` L108–116: "választás a kérdésfelvetésből következik"
- `05_kiserlet.tex` L91–103: összehasonlítótáblázat (Leung = teljes modell, jelen = LoRA 2,27%)

**Ami hiányzik:** az összekötő, explicit érvelés a módszertanban.

**Teendő:** `thesis/chapters/04_modszerek.tex` — a jelenlegi LoRA-bevezető bekezdés
(L157–162: "A modellt paraméterhatékony LoRA-finomhangolással adaptáltuk...")
UTÁN illesztendő 2–3 mondat:

> "Leung és munkatársai \cite{leung2024specom} teljes finomhangolást alkalmaztak
> Whisper-medium/-large modelleken; jelen munkában ezzel szemben a LoRA-t három
> okból választottam: (1) a $\approx$8 órányi tanítóadat és a 247\,M paraméteres
> modell aránya teljes FT esetén magas túltanulási kockázatot hordoz; (2) a LoRA
> paraméterhatékonysága (2{,}27\%) lehetővé tette, hogy a hat kísérleti konfigurációt
> reprodukálható körülmények között, egységes hardveren futtassam; (3) az elért
> 10{,}81\%-os WER (Exp~1) utólag megerősíti, hogy a paraméterhatékony adaptáció
> elegendő volt a feladathoz. A módszertani eltérések áttekintéséhez ld.\ a
> \ref{tab:elteresek}. táblázatot."

**Státusz:** ⏳

---

### 🔴 J7 — "Keresztbeszélős kiértékelés" szekció teljes újraírása

**Probléma:** A szekció kétszeresen félrevezető: (1) a "forrás–célpont" hangszín-konverziót
sugall, holott arról szó sincs; (2) a WER itt TTS-minőséget mér közvetve — ezt soha
nem mondja ki. A bíráló "egyáltalán nem érti", miközben a fő állítások ebből következnek.

**Mit csinál a kísérlet valójában** (`create_crossspeaker_manifest.py` alapján):
- Minden dysarthriás hangszínre ($X$) összegyűjtjük azokat a tréning mondatokat,
  amelyeket $X$ maga NEM rögzített (teszthalmaz kizárva → nincs leakage)
- $X$ speaker-embeddingével szintetizálja a Grad-TTS V1 ezeket → 37 × ~215 = ~7 980 felvétel
- **Nem hangszín-konverzió.** Minden speakernél a hiányzó mondatait szintetizálják
  saját hangjával

**A ki nem fejtett logikai lánc:**
1. Ha Grad-TTS tökéletes lenne → Exp 1 ASR ~10,81% WER-t kapna szintetikus hangon is
2. Ehelyett 70,16% → a szintetikus hang akusztikailag messze van a valóditól
3. 189,71% baseline WER → az általános ASR is összeomlik ezen → nem természetes hangzású
4. Következmény: ezzel magyarázható az Exp 2–4 gyenge augmentációs hatása

**Szükséges változtatások:**
1. Szekciócím → `Szintetikus hangminőség ASR-alapú értékelése`
2. Bevezető mondatok: eltávolítani "forrás–célpont" teljesen, helyette:
   *"minden hangszín saját speaker-embeddingével szintetizáltuk az általa nem
   rögzített tréningmondatokat (test-mentes szövegkészletből)"*
3. Explicit mondat: *"A WER itt nem az ASR teljesítményét értékeli, hanem
   a szintetikus hangok akusztikai realizmusát méri közvetve: minél magasabb
   az ASR hibaarány, annál kevésbé valószerű a szintetikus hang"*
4. A 4-lépéses logikai lánc explicit kifejtése a következtetés bekezdésben

**Státusz:** ⏳

---

### 🔴 J8 — Kísérletek rendszerezése: Exp0–5 elnevezés és struktúra

**Probléma:** Az Exp0–5 kódok fejlesztési maradványok, nem olvasónak tervezett nevek.
Inkonzisztens: V1–V4, szöveg-diszjunkt, súlyossági csoportok mind más logika szerint
vannak jelölve. A bíráló: "zavaros, nehezen követhető, hogy mi mikor miért történik."

**(A) Átnevezési javaslat** — leíró labelek bevezetése az Exp kódok mellé/helyett:

| Jelenlegi | Leíró label |
|-----------|-------------|
| Exp 0 | Baseline |
| Exp 1 | FT-Valódi |
| Exp 2 | FT-Szintetikus |
| Exp 3 | FT-Valódi+Szint |
| Exp 4 | FT-Valódi+SzintNagy |
| Exp 5 | FT-CsakSzintNagyDomén |

(A kódok megtarthatók zárójelben, pl. "FT-Valódi (Exp 1)".)

**(B) Javasolt fejezetsorrend az eredmények fejezetben:**
1. Összesített ASR eredmények (főtáblázat)
2. Kísérletenként elemzés
3. Részletes elemzések (szöveg-diszjunkt, súlyossági csoportok)
4. Szintetikus hangminőség értékelése (→ J7 szekció)
5. Összehasonlítás a referencia cikkel

**Státusz:** ⏳

---

## Korábbi munkamenetben elvégzett javítások ✅

- ✅ 6.4 akusztikai regularitás mondat enyhítve
- ✅ 7.1 Tézispont 4 → látott szövegek WER 28,42% kvantifikálva
- ✅ 6.5 zárómondat feltételessé téve
- ✅ 4. fejezet spektrogram-felirat enyhítve
- ✅ 1.3 geng2020 konkrét kontextus hozzáadva
- ✅ 6.3 "különböző beszélőket" → "ugyanazokat a 37 beszélőt (különböző felvételekkel)"
- ✅ 2. fejezet "szignifikánsan" → "érdemi mértékben"
- ✅ 6.4 dupla "ez" javítva
- ✅ Exp 5 "gyakorlatilag azonos" → "3,47 pp-tal magasabb / kismértékben rontott"
- ✅ V2–V4 keresztbeszélős WER mérési alap hozzáadva (04, 07 fejezet)
- ✅ Tézispont 4 (07): "semmilyen javulást nem hozott" → "+3,47 pp rontott"
