# Model Retraining Guide

## What was fixed first

ISOT is encoded as `0 = fake` and `1 = true`. The API previously displayed
those classes in reverse. `app/model_labels.py` now owns that contract, and v2
checkpoints save explicit `id2label` metadata.

This fixes reversed verdicts, but it does not turn an article-style classifier
into a general fact checker. Always measure both the held-out dataset score and
a small set of real claims.

## 1. Create a training environment

From `backend`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-training.txt
```

For an NVIDIA GPU, install the appropriate CUDA build of PyTorch before the
training requirements. The script automatically enables mixed precision when
CUDA is available.

## 2. Record the baseline

The evaluator selects the exact 15% test partition excluded by the v2 training
script. It also saves machine-readable metrics.

```powershell
python app\evaluate_baseline.py `
  --fake-csv "C:\path\to\Fake.csv" `
  --true-csv "C:\path\to\True.csv" `
  --report app\baseline_metrics.json
```

Do not treat a random sample from the full dataset as a valid post-training
test: it overlaps v2's training partition.

## 3. Train v2

```powershell
python app\train_model_v2.py `
  --fake-csv "C:\path\to\Fake.csv" `
  --true-csv "C:\path\to\True.csv" `
  --output-dir app\misinformation_model_v2
```

The script uses title plus body, a deterministic 70/15/15 split, 256 tokens,
three epochs, early stopping, and saves label metadata with the checkpoint.
Training can take hours on CPU.

## 4. Evaluate the new checkpoint

```powershell
python app\evaluate_baseline.py `
  --model-path app\misinformation_model_v2 `
  --fake-csv "C:\path\to\Fake.csv" `
  --true-csv "C:\path\to\True.csv" `
  --report app\v2_metrics.json
```

Compare accuracy, precision, recall, and F1 in `baseline_metrics.json` and
`v2_metrics.json`. There is no defensible guaranteed score such as 72%; accept
the model only if measured results improve.

## 5. Test v2 without replacing the current model

The API accepts a checkpoint path through `MODEL_PATH`:

```powershell
$env:MODEL_PATH = (Resolve-Path app\misinformation_model_v2)
uvicorn app.main:app --host 127.0.0.1 --port 8081
```

Test at least these two opposing controls:

```powershell
$trueBody = @{text='The Earth orbits the Sun.'} | ConvertTo-Json
$falseBody = @{text='The Earth is flat.'} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8081/predict -Method Post -ContentType application/json -Body $trueBody
Invoke-RestMethod http://127.0.0.1:8081/predict -Method Post -ContentType application/json -Body $falseBody
```

Also maintain a versioned challenge set of at least 50 claims covering science,
health, politics, dates, numbers, and negation. Report its results separately
from ISOT results.

## 6. Deploy

After both evaluations pass, either replace `app/misinformation_model` with the
validated checkpoint or set `MODEL_PATH` in the deployment. Then rebuild:

```powershell
docker compose up -d --build api
Invoke-RestMethod http://127.0.0.1:8081/health
```

## Recommended next model iteration

ISOT mainly teaches writing and source patterns in news articles. Short selected
claims are out of distribution, so full-text ISOT retraining alone is unlikely
to solve the product's core task. Use claim/evidence data such as FEVER and an
architecture that retrieves evidence before deciding `supported`, `refuted`, or
`not enough information`. Map only `refuted` to misinformation; do not force
unknown claims into a binary verdict.

## Acceptance criteria

- The same held-out test partition is used for baseline and v2.
- V2 improves held-out F1 without materially hurting either class's recall.
- Both true and false challenge claims are classified above the agreed target.
- Unknown or unsupported claims are not presented as high-confidence facts.
- The API health check passes and the extension shows the API's label unchanged.
