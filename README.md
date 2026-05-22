
# NewThesis

Ez a projekt magyar diszartriás beszédszintézis és automatikus beszédfelismerés (ASR) kísérletek implementációját tartalmazza.
A fő cél: Grad-TTS alapú szintetikus beszéd generálása, majd Whisper modellek finomhangolása és kiértékelése magyar nyelven.

## Fő tartalom

- **Python szkriptek**:  
	Adat-előkészítés, manifest generálás, szintetikus beszéd generálás, ASR finomhangolás és kiértékelés.
- **Grad-TTS/**:  
	A Grad-TTS modell forráskódja (csak a futtatáshoz szükséges részek).
- **Shell scriptek**:  
	Függőségek telepítése, pipeline futtatása.
- **README.md, requirements.txt, evaluation_requirements.txt**:  
	Dokumentáció és szükséges csomagok listája.

## Fő pipeline lépések

1. **Függőségek telepítése**  
	 `bash install_evaluation_deps.sh`

2. **Szintetikus beszéd generálása**  
	 `python generate_test_set.py --checkpoint <ckpt> --manifest <manifest.csv> --output-dir <output_wavs>`

3. **Whisper ASR finomhangolás**  
	 `python whisper_finetune.py --train-manifest <train.csv> --val-manifest <val.csv> --experiment <type> --output-dir <output_dir>`

4. **ASR kiértékelés**  
	 `python whisper_evaluate.py --test-manifest <test.csv> --model-path <model_dir> --output-csv <results.csv>`

## Mi NINCS benne?

- Nagy adathalmazok, modellek, logok, segéd/teszt szkriptek, prezentációk, eredmények: ezek az `_EXCLUDE_FROM_SUBMISSION` mappába kerültek, nem részei a beadásnak.
