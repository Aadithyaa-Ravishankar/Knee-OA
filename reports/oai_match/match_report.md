# OAI feature-label match

Source: `KXR_SQ_BU00.txt` (baseline, Boston University reading). Column names confirmed against `kxr_sq_bu00_Contents.pdf`. `READPRJ` 42 is treated as project 37 when breaking ties (`kXR_SQ_BU_Descrip.pdf`).

- Images: 30017
- Matched: 29358 (97.8%)
- Unmatched: 0
- Filename has no OAI id: 659
- KL agreement of the saved row (KL_oai == folder KL): 99.8% (29295 / 29358)
- KL agreement if a matching grade is not allowed to win the tie: 99.6% (8246 / 8281 knees)
- Knees that also have an OAI reading with a different KL: 73
- OAI rows: 16666; knees with more than one row: 4954
- Rows coded READPRJ 42 (compared as 37): 0

## Match rate

| Group | n | matched | rate |
|---|---:|---:|---:|
| dataset=Dataset | 9786 | 9786 | 100.0% |
| dataset=Dataset_combined | 10445 | 9786 | 93.7% |
| dataset=kneeKL299 | 9786 | 9786 | 100.0% |
| split=auto_test | 4578 | 4578 | 100.0% |
| split=test | 4968 | 4968 | 100.0% |
| split=train | 17993 | 17334 | 96.3% |
| split=val | 2478 | 2478 | 100.0% |
| KL_folder=0 | 11571 | 11571 | 100.0% |
| KL_folder=1 | 5310 | 5310 | 100.0% |
| KL_folder=2 | 7966 | 7734 | 97.1% |
| KL_folder=3 | 4079 | 3858 | 94.6% |
| KL_folder=4 | 1091 | 885 | 81.1% |

## KL agreement

The saved `KL_oai` keeps an OAI reading that matches the folder grade when one exists, so the crosstab can look perfect. The independent check above ignores that preference. `auto_test` files are named `<ID>_1` (right) and `<ID>_2` (left), the same side codes as the OAI file. Osteophyte, sclerosis, cyst, and attrition grades are often missing when the OAI KL is 0 or 1; the data dictionary says those features were not read on knees without definite osteoarthritis. Joint-space grades are filled in.

| KL_folder | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 11556 | 6 | 6 | 0 | 3 |
| 1 | 3 | 5298 | 6 | 3 | 0 |
| 2 | 9 | 9 | 7710 | 6 | 0 |
| 3 | 0 | 0 | 6 | 3846 | 6 |
| 4 | 0 | 0 | 0 | 0 | 885 |

Knees whose OAI file also contains a different KL (up to 20):

- 9001897L: folder 0, other OAI grades [1]
- 9066677L: folder 0, other OAI grades [1]
- 9268724L: folder 0, other OAI grades [1]
- 9362660L: folder 0, other OAI grades [1]
- 9656390L: folder 0, other OAI grades [1]
- 9816138R: folder 0, other OAI grades [1]
- 9002316L: folder 1, other OAI grades [0]
- 9197064R: folder 1, other OAI grades [0]
- 9204538R: folder 1, other OAI grades [0]
- 9226514L: folder 1, other OAI grades [0]
- 9226514R: folder 1, other OAI grades [0]
- 9327968R: folder 1, other OAI grades [0]
- 9346242R: folder 1, other OAI grades [0]
- 9362264L: folder 1, other OAI grades [0]
- 9373786R: folder 1, other OAI grades [0]
- 9416851L: folder 1, other OAI grades [0]
- 9505140L: folder 1, other OAI grades [2]
- 9621208R: folder 1, other OAI grades [0]
- 9626197R: folder 1, other OAI grades [0]
- 9634064R: folder 1, other OAI grades [0]

## Missing features among matched images

| Feature | missing |
|---|---:|
| ost_fem_med | 10947 |
| scl_fem_med | 11520 |
| cyst_fem_med | 11520 |
| jsn_med | 0 |
| chon_med | 10947 |
| ost_tib_med | 10947 |
| scl_tib_med | 11520 |
| cyst_tib_med | 11520 |
| att_tib_med | 11520 |
| ost_fem_lat | 10947 |
| scl_fem_lat | 11520 |
| cyst_fem_lat | 11520 |
| jsn_lat | 0 |
| chon_lat | 10947 |
| ost_tib_lat | 10947 |
| scl_tib_lat | 11520 |
| cyst_tib_lat | 11520 |
| att_tib_lat | 11520 |

## Feature distribution, one row per knee (ID, SIDE)

### KL 0 (n=3253)

- ost_max: `{"0": 780, "1": 20, "2": 1, "nan": 2452}`
- jsn_med: `{"0": 3253, "nan": 0}`
- jsn_lat: `{"0": 3253, "nan": 0}`

### KL 1 (n=1495)

- ost_max: `{"0": 375, "1": 495, "2": 5, "nan": 620}`
- jsn_med: `{"0": 889, "1": 606, "nan": 0}`
- jsn_lat: `{"0": 1432, "1": 63, "nan": 0}`

### KL 2 (n=2175)

- ost_max: `{"0": 26, "1": 1462, "2": 411, "3": 276, "nan": 0}`
- jsn_med: `{"0": 998, "1": 1175, "2": 2, "nan": 0}`
- jsn_lat: `{"0": 1915, "1": 260, "nan": 0}`

### KL 3 (n=1086)

- ost_max: `{"1": 360, "2": 337, "3": 389, "nan": 0}`
- jsn_med: `{"0": 203, "1": 16, "2": 867, "nan": 0}`
- jsn_lat: `{"0": 845, "1": 9, "2": 232, "nan": 0}`

### KL 4 (n=251)

- ost_max: `{"1": 5, "2": 30, "3": 216, "nan": 0}`
- jsn_med: `{"0": 82, "1": 4, "2": 2, "3": 163, "nan": 0}`
- jsn_lat: `{"0": 156, "1": 1, "2": 4, "3": 90, "nan": 0}`

## Same patient, left and right

Ignoring the auto_test copy: every patient with both knees keeps L and R in the same split, and in the same Pipeline fold when a fold is assigned. Train/val fold conflicts: 0.

`auto_test` is the same knees as `test` (4578 of 4578 images, per dataset copy included). The folder grade differs between those two copies for 21 knees. The 659 files with no OAI id are the `gp_` images (659).


## Which row was kept when a knee had several OAI readings

| Rule | knees |
|---|---:|
| fewest_nan | 2126 |
| fewest_nan+lowest_readprj | 119 |
| kl_agree | 52 |
| kl_agree+fewest_nan | 1 |
| lowest_readprj | 2389 |
| only_one | 3594 |

## Spot check (seed 0)

- `9746090R.png` (kneeKL299/train) folder KL 3, OAI KL 3, readprj 15, rule fewest_nan, candidates 2, nonzero features: scl_fem_med=1.0, jsn_med=2.0, ost_tib_med=1.0, scl_tib_med=1.0
- `9634937L.png` (kneeKL299/train) folder KL 2, OAI KL 2, readprj 15, rule fewest_nan, candidates 2, nonzero features: ost_tib_med=1.0
- `9183590_1.png` (Dataset_combined/auto_test) folder KL 1, OAI KL 1, readprj 15, rule fewest_nan, candidates 3, nonzero features: jsn_med=1.0
- `9508335R.png` (Dataset_combined/train) folder KL 3, OAI KL 3, readprj 15, rule only_one, candidates 1, nonzero features: ost_fem_med=3.0, scl_fem_med=2.0, jsn_med=2.0, ost_tib_med=3.0, scl_tib_med=2.0, ost_fem_lat=2.0, ost_tib_lat=1.0
- `9891691L.png` (Dataset/test) folder KL 2, OAI KL 2, readprj 15, rule fewest_nan, candidates 3, nonzero features: jsn_med=1.0, ost_fem_lat=1.0
- `9534424R.png` (Dataset/train) folder KL 0, OAI KL 0, readprj 15, rule lowest_readprj, candidates 3, all listed features 0 or missing
- `9239515L.png` (Dataset/train) folder KL 0, OAI KL 0, readprj 15, rule lowest_readprj, candidates 2, all listed features 0 or missing
- `9663428_1.png` (Dataset/auto_test) folder KL 1, OAI KL 1, readprj 15, rule fewest_nan, candidates 3, nonzero features: jsn_med=1.0
- `9435233L.png` (Dataset/train) folder KL 3, OAI KL 3, readprj 15, rule only_one, candidates 1, nonzero features: ost_fem_med=1.0, scl_fem_med=2.0, jsn_med=2.0, scl_tib_med=2.0
- `9951122R.png` (Dataset/train) folder KL 0, OAI KL 0, readprj 15, rule lowest_readprj, candidates 3, all listed features 0 or missing

## Unmatched files

659 files. Full list is in `match_report.json`.

- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (1).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (10).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (100).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (101).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (102).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (103).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (104).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (105).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (106).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (107).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (108).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (109).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (11).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (110).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (111).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (112).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (113).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (114).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (115).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (116).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (117).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (118).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (119).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (12).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (120).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (121).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (122).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (123).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (124).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (125).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (126).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (127).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (128).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (129).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (13).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (130).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (131).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (132).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (133).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (134).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (135).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (136).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (137).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (138).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (139).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (14).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (140).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (141).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (142).png`
- `D:\NN & DL\Knee-OA\Dataset_combined\train\2\gp_E1_MildG2 (143).png`
- … 609 more in the json
