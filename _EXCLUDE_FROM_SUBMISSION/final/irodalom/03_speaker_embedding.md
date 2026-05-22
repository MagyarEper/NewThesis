# Irodalom vázlat: Speaker Embedding

**Beillesztés helye:** `Dolgozat/02_irodalom.md` — 2.2 TTS szekció, Grad-TTS leírás részeként  
**Méret:** ~250–400 szó  
**Prioritás:** KÖZEPES — jelenleg csak megemlítve, de nem kifejtve

---

## Mit kell lefedni

### 2.x. Speaker embedding: hangszín kódolása neurális hálózattal

**Szükséges tartalom:**

1. **Motiváció (1-2 mondat)**  
   A multi-speaker TTS modelleknek meg kell tudniuk különböztetni a különböző hangszíneket. A speaker embedding egy fix méretű vektor, amely tömören kódolja egy adott személy akusztikai azonosságát — hangmagasságát, rezonanciáit, artikulációs stílusát.

2. **Hogyan tanítják? (1 bekezdés)**  
   A speaker embeddingeket általában előre betanított speaker verification modellek állítják elő:
   - **d-vector:** mély neurális hálózat (DNN) speaker verification célra tanítva; az utolsó rejtett réteg aktivációja az embedding (Variani et al., 2014)
   - **x-vector:** TDNN (Time-Delay Neural Network) alapú, PLDA backend; standard baseline a speaker ID feladatokban (Snyder et al., 2018)
   - **ECAPA-TDNN:** modernebb, attention-alapú pooling; ma az egyik legerősebb nyílt forráskódú speaker encoder

3. **Alkalmazás a Grad-TTS-ben (1 bekezdés)**  
   A Grad-TTS a szintézist speaker embeddingre kondicionálja: minden tanítási példánál a megfelelő speaker embedding hozzáadódik a modell belső reprezentációjához (condition vector). Generáláskor meg kell adni a célspeaker embeddingét — így a modell képes a megtanult hangszínekkel szintetizálni. A Magyar Dizartria Adatbázis 38 speakerjéhez 38 különböző embedding tartozik.

4. **Korlátok (1 mondat)**  
   A speaker embedding képes kódolni a normál hangszín-különbségeket, de a dizartriás ejtési sajátosságok (torzult artikuláció, rendellenes ritmus) nem minden esetben reprezentálódnak megbízhatóan — ez is hozzájárul a TTS minőségi korlátaihoz.

---

## Kulcsszavak kereséshez

- `speaker embedding d-vector x-vector TTS`
- `multi-speaker TTS speaker conditioning`
- `speaker verification embedding neural network`
- `Grad-TTS speaker embedding conditioning`

## Ajánlott irodalom (keress rájuk!)

| Forrás | Mit kell megnézni |
|--------|------------------|
| **Variani et al. (2014)** — *"Deep Neural Networks for Small Footprint Text-Dependent Speaker Verification", ICASSP* | A d-vector fogalmának bevezetése |
| **Snyder et al. (2018)** — *"X-Vectors: Robust DNN Embeddings for Speaker Recognition", ICASSP* | Az x-vector leírása — SpeechBrain is ezt implementálja |
| **Jia et al. (2018)** — *"Transfer Learning from Speaker Verification to Multi-Speaker TTS", NeurIPS* | **Legfontosabb** — ez vezette be a speaker embedding → TTS kondicionálást; a "generalized end-to-end loss" alapcikke; Tacotron-alapú, de az elv ugyanez |
| **Popov et al. (2021)** — *Grad-TTS, ICML* | A Grad-TTS speaker kondicionálásának leírása a Supplementary Materialban vagy a főszövegben |
| **Desplanques et al. (2020)** — *"ECAPA-TDNN", Interspeech* | Modern speaker encoder; SpeechBrain alapértelmezett modellje |

## Opcionális ábra

Egy egyszerű diagram: `[Wavefájl] → [Speaker Encoder] → [Speaker Embedding vektor] → [Grad-TTS kondicionálás]`


## Saját szöveg

2.x. Beszélő-beágyazás (Speaker Embedding): az egyedi hangkarakter kódolása

A korszerű többszereplős beszédszintézis alapvető követelménye, hogy a modell képes legyen különbséget tenni a különböző beszélők akusztikai jellemzői között, és akár ismeretlen hangszíneket is reprodukálni tudjon. Ezt a feladatot a beszélő-beágyazás (speaker embedding) valósítja meg, amely egy rögzített dimenziójú numerikus vektor formájában tömören kódolja az egyén akusztikai identitását, így a hangmagasságot, a rezonanciákat és az egyedi artikulációs stílust.

A speaker embeddingeket jellemzően előre betanított beszélő-azonosító (speaker verification) modellek segítségével állítják elő, amelyek célja a beszélők közötti távolság maximalizálása a vektortérben. A technológia fejlődésének egyik korai mérföldköve a d-vector volt, amely egy mély neurális hálózat (DNN) utolsó rejtett rétegének aktivációját használta reprezentációként (Variani et al., 2014). Ezt követte a robusztusabb x-vector architektúra, amely időkésleltetéses neurális hálózatra (TDNN) épül, és hosszú ideig a beszélő-felismerési feladatok standard bázismodelljévé vált (Snyder et al., 2018). Napjaink egyik legkorszerűbb eljárása az ECAPA-TDNN, amely csatornafüggő figyelem-mechanizmussal (attention) és aggregált hierarchikus jellemzőkkel éri el a kiemelkedő pontosságot (Desplanques et al., 2020).

A beszélő-beágyazások TTS rendszerekben való alkalmazásának elméleti alapjait Jia és munkatársai (2018) fektették le, igazolva, hogy a transzfer tanulás módszerével a beszélő-felismerésre tanított modellek tudása átvihető a szintézis folyamatába. A dolgozatban alkalmazott Grad-TTS modell (Popov et al., 2021) ezt az elvet követve a szintézist a speaker embeddingre kondicionálja. A gyakorlatban minden tanítási példához egy kinyert embedding vektor kapcsolódik, amely „feltételként” (condition vector) hozzáadódik a modell belső reprezentációihoz. Ez teszi lehetővé, hogy a modell a Magyar Dizartria Adatbázisban szereplő 38 különböző beszélő hangszínét külön-külön elsajátítsa.

Ugyanakkor a speaker embeddingek alkalmazása korlátokkal is jár, különösen patológiás beszéd esetén. Bár a vektorok hatékonyan kódolják a normál hangszín-különbségeket, a dizartriára jellemző egyedi ejtési sajátosságok — mint a torzult artikuláció vagy a rendellenes ritmus — nem minden esetben reprezentálódnak megbízhatóan a standard embedding-térben. Ez a korlát közvetlen hatással lehet a generált beszéd természetességére és a betegspecifikus jellemzők pontos visszaadására.
Felhasznált hivatkozások:

    Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification. Interspeech 2020, 3830-3834.

    Jia, Y., Zhang, Y., Weiss, R. J., Wang, Q., Shen, J., Ren, F., Nguyen, P., Pang, R., Wu, Y., & Wu, Z. (2018). Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis. Advances in Neural Information Processing Systems (NeurIPS), 31.

    Popov, V., Vovk, I., Gogoryan, V., Sadekova, T., & Kudinov, M. (2021). Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech. Proceedings of the 38th International Conference on Machine Learning (ICML).

    Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Manohar, N. (2018). X-vectors: Robust DNN Embeddings for Speaker Recognition. 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5329-5333.

    Variani, E., Lei, X., McDermott, E., Moreno, I. L., & Gonzalez-Dominguez, J. (2014). Deep neural networks for small footprint text-dependent speaker verification. 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 4052–4056.