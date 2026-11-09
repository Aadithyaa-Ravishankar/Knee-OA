# Kaggle: soft joint-band attention + spatial aux screen

Locked champion stays: `r50_dual_cascade_sqrt_aux`.

New run: **`r50_dual_cascade_sqrt_aux_attn_spatial`**

- **Part A:** `--attn-band-weight 0.1` soft joint-band / frame-border prior  
- **Part B:** `--spatial-aux` + `--aux-weight 0.75` (med/lat map heads)

## PC

```powershell
python kaggle/make_bundle.py --code
```

Upload `kaggle/bundle/knee-oa-code.zip` to the Kaggle code dataset.

Use notebook `kaggle/knee_oa_cv_train.ipynb` (CONFIG already set for this screen).

## After download

```powershell
python src/training/stage6_screen_joint_focus.py `
  --runs-dir outputs/cv/runs `
  --candidates r50_dual_cascade_sqrt_aux r50_dual_cascade_sqrt_aux_attn_spatial `
  --out reports/pipeline/screen_attn_spatial.json
```

Win rule: OOF QWK ≈≥ champion, **lower** global frame-border CAM mass, **higher** joint-band mass. Only then locked test.
