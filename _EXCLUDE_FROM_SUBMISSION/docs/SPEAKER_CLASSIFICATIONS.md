# Magyar Dizartria Adatbázis – Beszélők besorolása (MSZNY2026.xlsx alapján)

Forrás: MSZNY2026.xlsx – HUN lap (`id`, `diagnosis`, `sex`, `age`, `Lívia` oszlopok)

A tézisben szereplő manifest.csv **38 beszélőt** tartalmaz (C_001–C_042, hiányzók: C_019, C_034, C_035, C_039).

## A 38 manifest-beszélő adatai

| ID    | Diagnózis               | Nem    | Kor | Súlyossági besorolás (Lívia) | A tézis adathalmazában |
|-------|------------------------|--------|-----|------------------------------|------------------------|
| C_001 | Parkinson              | férfi  | 73  | Középsúlyos                  | ✓                      |
| C_002 | Parkinson              | férfi  | 69  | —                            | ✓                      |
| C_003 | stroke                 | férfi  | 54  | Középsúlyos                  | ✓                      |
| C_004 | aphasia, apraxia       | —      | 53  | —                            | ✓                      |
| C_005 | aphasia, apraxia       | férfi  | 67  | —                            | ✓                      |
| C_006 | stroke                 | férfi  | 66  | Enyhe                        | ✓                      |
| C_007 | —                      | férfi  | 54  | —                            | ✓                      |
| C_008 | —                      | —      | 60  | —                            | ✓                      |
| C_009 | dysarthria             | —      | 69  | —                            | ✓                      |
| C_010 | dysarthria             | —      | 64  | Enyhe                        | ✓                      |
| C_011 | —                      | férfi  | 52  | —                            | ✓                      |
| C_012 | —                      | —      | 49  | —                            | ✓                      |
| C_013 | aphasia                | férfi  | 63  | —                            | ✓                      |
| C_014 | Parkinson              | férfi  | 80  | —                            | ✓                      |
| C_015 | dysarthria             | —      | 59  | —                            | ✓                      |
| C_016 | dysarthria, stroke     | férfi  | 40  | Enyhe                        | ✓                      |
| C_017 | dysarthria             | férfi  | 70  | —                            | ✓                      |
| C_018 | dysarthria (stroke)    | férfi  | 66  | Enyhe                        | ✓                      |
| C_019 | dysarthria (stroke)    | férfi  | 72  | —                            | ✗ (nem szerepel)       |
| C_020 | dysarthria             | férfi  | 77  | —                            | ✓                      |
| C_021 | dysarthria             | férfi  | 54  | —                            | ✓                      |
| C_022 | dysarthria             | férfi  | 58  | —                            | ✓                      |
| C_023 | dysarthria             | férfi  | 64  | —                            | ✓                      |
| C_024 | dysarthria             | nő     | 74  | —                            | ✓                      |
| C_025 | dysarthria             | nő     | 65  | —                            | ✓                      |
| C_026 | dysarthria             | nő     | 70  | Középsúlyos                  | ✓                      |
| C_027 | dysarthria             | férfi  | 66  | **Kontroll**                 | ✓                      |
| C_028 | dysarthria             | nő     | 49  | —                            | ✓                      |
| C_029 | dysarthria             | férfi  | 33  | Középsúlyos                  | ✓                      |
| C_030 | dysarthria             | férfi  | 47  | —                            | ✓                      |
| C_031 | dysarthria             | nő     | 39  | Súlyos                       | ✓                      |
| C_032 | dysarthria             | férfi  | 76  | —                            | ✓                      |
| C_033 | dysarthria             | férfi  | 40  | Súlyos                       | ✓                      |
| C_034 | dysarthria             | férfi  | 71  | —                            | ✗ (nem szerepel)       |
| C_035 | dysarthria             | férfi  | 52  | —                            | ✗ (nem szerepel)       |
| C_036 | dysarthria             | nő     | 46  | Enyhe                        | ✓                      |
| C_037 | dysarthria             | nő     | 44  | —                            | ✓                      |
| C_038 | dysarthria, apraxia    | nő     | 58  | —                            | ✓                      |
| C_039 | dysarthria             | nő     | 42  | —                            | ✗ (nem szerepel)       |
| C_040 | dysarthria             | nő     | 36  | Enyhe                        | ✓                      |
| C_041 | dysarthria             | férfi  | 29  | Enyhe                        | ✓                      |
| C_042 | dysarthria             | nő     | 49  | Középsúlyos                  | ✓                      |

## Összefoglalás (a 38 manifest-beszélőre)

| Kategória        | Darabszám |
|-----------------|-----------|
| Dizartriás       | 37        |
| Kontroll (C_027) | 1         |
| **Összesen**     | **38**    |

### Súlyossági besorolás (Lívia értékelése, az MSZNY2026-os részhalmazra)

| Súlyosság    | Db |
|-------------|-----|
| Enyhe        | 7  |
| Középsúlyos  | 5  |
| Súlyos       | 2  |
| Kontroll     | 1  |
| Nem értékelt | 23 |

### Diagnózisok megoszlása (a 38 manifest-beszélőre)

- Dysarthria (általános): 26
- Parkinson: 3 (C_001, C_002, C_014)
- Stroke: 1 (C_003)
- Dysarthria + stroke: 3 (C_016, C_018, C_019*)
- Aphasia / apraxia: 3 (C_004, C_005, C_013)
- Dysarthria + apraxia: 1 (C_038)
- Ismeretlen: 3 (C_007, C_008, C_011, C_012)

> **Megjegyzés:** A C_027 diagnózisa "dysarthria", de a súlyossági értékelés szerint "Kontroll" — valószínűleg ez az adatbázis egyetlen egészséges kontroll-beszélője.
