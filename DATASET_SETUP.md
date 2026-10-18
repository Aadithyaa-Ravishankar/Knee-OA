# Dataset Setup Guide

> **For teammates:** Follow this guide exactly to download and place the dataset so all scripts work without any changes.

---

## Which Dataset?

**Name:** Knee Osteoarthritis Dataset with Severity Grading  
**Source:** Kaggle  
**Direct link:** https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity

---

## Method 1 — Download via Browser (Easiest)

### Step 1 — Create / Log in to Kaggle

1. Go to https://www.kaggle.com
2. Sign in or create a free account

### Step 2 — Accept the dataset terms

1. Open the dataset page:  
   👉 https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity
2. Click the **Download** button (top right)
3. If prompted, accept the rules

### Step 3 — Download the zip

You will download a file called:
```
archive.zip   (~1–2 GB)
```

### Step 4 — Extract the zip

After extracting, you should see a folder structure like:
```
archive/
├── train/
│   ├── 0/   ← KL Grade 0 images
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
│   ├── 0/ … 4/
├── test/
│   ├── 0/ … 4/
└── auto_test/
    ├── 0/ … 4/
```

### Step 5 — Place the dataset in the correct location

The dataset must be placed at this **exact absolute path**:

```
/Users/<your-username>/Desktop/KNEE OA/dataset/
```

So the final structure should look like:

```
/Users/<your-username>/Desktop/KNEE OA/
├── dataset/               ← PUT THE EXTRACTED FOLDER HERE
│   ├── train/
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── 2/
│   │   ├── 3/
│   │   └── 4/
│   ├── val/
│   │   ├── 0/ … 4/
│   ├── test/
│   │   ├── 0/ … 4/
│   └── auto_test/
│       ├── 0/ … 4/
│
└── knee_oa_xai/           ← This is the cloned repo
    ├── src/
    ├── configs/
    └── ...
```

> ⚠️ **Important:** The folder must be named exactly `dataset` — not `archive`, not `knee-osteoarthritis-dataset`. Rename it if needed.

---

## Method 2 — Download via Kaggle CLI (Faster)

### Step 1 — Install the Kaggle CLI
```bash
pip install kaggle
```

### Step 2 — Get your API key
1. Go to https://www.kaggle.com/settings
2. Scroll to **API** section → click **Create New Token**
3. This downloads `kaggle.json`
4. Move it to `~/.kaggle/kaggle.json`
5. Set permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 3 — Download and extract

Run these commands from your Desktop:
```bash
cd ~/Desktop/KNEE\ OA

# Download
kaggle datasets download -d shashwatwork/knee-osteoarthritis-dataset-with-severity

# Unzip (creates a folder called 'archive' or similar)
unzip knee-osteoarthritis-dataset-with-severity.zip -d dataset_temp

# Rename to the correct folder name
mv dataset_temp dataset    # adjust folder name if different after unzipping
```

---

## Step 6 — Verify the Setup

After placing the dataset, run this quick verification from the project root:

```bash
cd "/Users/<your-username>/Desktop/KNEE OA/knee_oa_xai"
source .venv/bin/activate

python3 -c "
from pathlib import Path
base = Path('/Users/$(whoami)/Desktop/KNEE OA/dataset')
splits = ['train', 'val', 'test', 'auto_test']
grades = [0, 1, 2, 3, 4]
total = 0
for s in splits:
    for g in grades:
        d = base / s / str(g)
        n = len(list(d.glob('*.png'))) if d.exists() else 0
        total += n
        print(f'  {s}/KL-{g}: {n} images')
print(f'\nTotal images found: {total}')
print('✅ All good!' if total == 9786 else '❌ Something is wrong — expected 9786 images')
"
```

**Expected output:**
```
  train/KL-0: 2286 images
  train/KL-1: 1046 images
  ...
Total images found: 9786
✅ All good!
```

---

## Step 7 — Set Up the Python Environment

```bash
cd "/Users/<your-username>/Desktop/KNEE OA/knee_oa_xai"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version — works on all machines)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# If you have an M1/M2/M3 Mac and want GPU acceleration:
# pip install torch torchvision   (standard install — uses MPS Metal backend)
```

---

## Path Reference Summary

| Variable | Value |
|----------|-------|
| Dataset root | `/Users/<your-username>/Desktop/KNEE OA/dataset/` |
| Project root | `/Users/<your-username>/Desktop/KNEE OA/knee_oa_xai/` |
| Config file | `knee_oa_xai/configs/config.yaml` |
| Audit script | `knee_oa_xai/src/preprocessing/stage1_dataset_audit.py` |
| Training script | `knee_oa_xai/src/training/stage3_train_baseline.py` |

> **Note:** All scripts hardcode the dataset path as `/Users/aadithyaaravishankar/Desktop/KNEE OA/dataset`.  
> If your username is different, update the `DATASET_DIR` variable at the top of each script, or edit `configs/config.yaml`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: dataset/train/0` | Dataset is not in the right folder — re-read Step 5 |
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` inside the `.venv` |
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` |
| `9786 images not found` | The zip may not have extracted into the right sub-folder; check the folder names |
| Kaggle download says "403 Forbidden" | You need to accept the dataset rules on the Kaggle website first |

---

## Contact

If you run into any issues, contact **Aadithyaa** or open an issue on the repo:  
https://github.com/Aadithyaa-Ravishankar/Knee-OA
