# Cloud-Based Fact Checker

A browser-assisted misinformation classifier built with a Firefox extension,
FastAPI, DistilBERT, PostgreSQL, and Docker Compose. Select text on a webpage,
click **Analyze**, and the extension sends the selection to the local API for a
`Truth` or `Misinformation` prediction with a confidence score.

> [!IMPORTANT]
> This project is a machine-learning classifier, not an authoritative source of
> truth. It can be confidently wrong, especially on short claims outside its
> training distribution. Do not use its output as the sole basis for medical,
> legal, financial, or civic decisions.

## Current status

- Browser selection-to-result workflow is operational.
- FastAPI is exposed locally at `http://127.0.0.1:8081`.
- PostgreSQL records inference latency and aggregate request metrics.
- The API uses the v2 DistilBERT checkpoint through configurable `MODEL_PATH`.
- The v2 ISOT held-out evaluation reached 99.97% accuracy and F1 on 6,727
  articles. This score reflects performance on ISOT-style news articles, not
  universal fact-checking accuracy.
- A short-claim smoke test still misclassified “Water freezes at zero degrees
  Celsius,” demonstrating the remaining out-of-distribution limitation.

Measured reports are stored in
[`backend/app/baseline_metrics.json`](backend/app/baseline_metrics.json) and
[`backend/app/v2_metrics.json`](backend/app/v2_metrics.json).

## How it works

```text
Selected webpage text
        |
        v
Firefox content script -> background script -> POST /predict
                                               |
                                               v
                                      DistilBERT checkpoint
                                               |
                         +---------------------+--------------------+
                         v                                          v
                 prediction response                    PostgreSQL inference log
```

The background script performs the HTTP request because direct requests from a
content script on an HTTPS page can be blocked as mixed content.

## Repository layout

```text
.
|-- backend/
|   |-- app/
|   |   |-- main.py                  FastAPI routes and rate limiting
|   |   |-- final_model.py           Model loading and inference
|   |   |-- model_labels.py          Stable 0=fake, 1=true label contract
|   |   |-- db.py                    PostgreSQL logging and metrics
|   |   |-- schemas.py               Request and response validation
|   |   |-- train_model_v2.py        Deterministic v2 training pipeline
|   |   `-- evaluate_baseline.py     Held-out checkpoint evaluation
|   |-- tests/                        API and label-contract tests
|   |-- Dockerfile
|   |-- requirements.txt             Runtime dependencies
|   `-- requirements-training.txt    Training dependencies
|-- extension/
|   |-- manifest.json                Firefox Manifest V3 configuration
|   |-- background.js                API request bridge
|   `-- content.js                   Selection UI and results popup
|-- FACT_CHECKER_GPU_TRAINING.ipynb  Colab GPU training workflow
|-- RETRAINING_GUIDE.md              Detailed retraining guide
`-- docker-compose.yml               API and PostgreSQL services
```

## Prerequisites

- Docker Desktop with Docker Compose
- Firefox 109 or newer
- The trained model directory described below
- About 4 GB of free disk space for Docker images and model artifacts

Python 3.11 is only required when running tests or training outside Docker.

## Quick start

### 1. Clone and configure

```powershell
git clone https://github.com/BBS17/Cloud_Project.git
cd Cloud_Project
Copy-Item .env.example .env
```

The development defaults start PostgreSQL locally and publish the API on port
8081. Change the database password before using the project beyond local
development.

### 2. Add the model checkpoint

Model weights are excluded from Git because `model.safetensors` is about 256 MB.
Place the trained directory at:

```text
backend/app/misinformation_model_v2/
|-- config.json
|-- model.safetensors
|-- tokenizer.json
|-- tokenizer_config.json
|-- special_tokens_map.json
`-- vocab.txt
```

If your model uses another directory, ensure it is copied into the Docker image
under `backend/app/` and set its container path in `.env`, for example:

```dotenv
MODEL_PATH=/app/app/my_model
```

To generate the checkpoint yourself, follow [Model training](#model-training).

### 3. Start the services

Make sure Docker Desktop is running, then execute:

```powershell
docker compose up -d --build
```

Wait for the API and database:

```powershell
Invoke-RestMethod http://127.0.0.1:8081/health
```

Expected response:

```json
{"status":"ok"}
```

API documentation is available at <http://127.0.0.1:8081/docs>.

### 4. Load the Firefox extension

1. Open `about:debugging#/runtime/this-firefox`.
2. Select **Load Temporary Add-on**.
3. Choose `extension/manifest.json`.
4. Open or refresh a webpage.
5. Select at least three characters of text and click **Analyze**.

After editing extension files, return to `about:debugging` and click **Reload**,
then refresh the target webpage. Temporary extensions must be loaded again after
Firefox restarts.

The checked-in manifest targets Firefox. Chromium Manifest V3 expects
`background.service_worker` instead of Firefox's `background.scripts`; create a
Chromium-specific manifest before loading this directory in Chrome or Edge.

## Configuration

Copy `.env.example` to `.env`. Supported settings:

| Variable | Default | Purpose |
|---|---|---|
| `DB_HOST` | `localhost` in `.env`; `db` in Compose | PostgreSQL hostname |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `factchecker` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `changeme` | Database password |
| `ALLOWED_ORIGINS` | `http://localhost:8080` | Comma-separated CORS origins |
| `MODEL_PATH` | `/app/app/misinformation_model_v2` | Checkpoint path inside the container |

Never commit `.env` or credentials. The file is ignored by Git.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Basic liveness response |
| `GET` | `/health` | Verifies model and database availability |
| `POST` | `/predict` | Classifies text between 3 and 5,000 characters |
| `GET` | `/metrics` | Returns aggregate inference count and latency |
| `GET` | `/docs` | Interactive OpenAPI documentation |

`POST /predict` is limited to 30 requests per minute per client IP.

Example:

```powershell
$body = @{ text = "The Earth orbits the Sun." } | ConvertTo-Json
Invoke-RestMethod `
  -Uri http://127.0.0.1:8081/predict `
  -Method Post `
  -ContentType application/json `
  -Body $body
```

Example response:

```json
{
  "label": "Truth",
  "confidence": 84.37
}
```

## Development without Docker

Start a PostgreSQL instance and configure the `DB_*` environment variables,
then run:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MODEL_PATH = (Resolve-Path app\misinformation_model_v2)
uvicorn app.main:app --reload --port 8081
```

## Testing

The API tests mock model loading, so they do not require the 256 MB checkpoint:

```powershell
cd backend
pip install -r requirements.txt
pip install pytest httpx
pytest tests -v
```

Syntax-only validation inside the running container:

```powershell
docker compose exec -T api python -m py_compile `
  app/main.py app/final_model.py app/model_labels.py
```

GitHub Actions runs Python linting, tests with a PostgreSQL service, and a Docker
image build on pushes and pull requests to `main`.

## Model training

### Local training

ISOT expects separate `Fake.csv` and `True.csv` files. These datasets are also
excluded from Git.

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-training.txt

python app\evaluate_baseline.py `
  --fake-csv "C:\path\to\Fake.csv" `
  --true-csv "C:\path\to\True.csv"

python app\train_model_v2.py `
  --fake-csv "C:\path\to\Fake.csv" `
  --true-csv "C:\path\to\True.csv" `
  --output-dir app\misinformation_model_v2
```

The current configuration uses a deterministic 70/15/15 split, full article
title plus body, a 256-token limit, batch size 8, three epochs, and early
stopping. CPU training took approximately 7 hours 23 minutes on the development
machine.

### Google Colab GPU training

Upload [`FACT_CHECKER_GPU_TRAINING.ipynb`](FACT_CHECKER_GPU_TRAINING.ipynb) to
Google Colab, select a T4 GPU, and run the cells. The notebook supports:

- Google Drive checkpoint persistence
- resume from the latest checkpoint
- Drive paths or interactive dataset uploads
- baseline and v2 reports on the same held-out partition
- smoke tests and ZIP export

See [`RETRAINING_GUIDE.md`](RETRAINING_GUIDE.md) for the complete workflow and
acceptance criteria.

## Known limitations

- ISOT teaches news-writing and source patterns more than evidence-backed claim
  verification. Very high ISOT accuracy can coexist with errors on simple facts.
- The model returns a binary verdict even when evidence is insufficient.
- Confidence is a softmax score, not a calibrated probability that a claim is
  objectively true.
- Text is truncated to the first 256 tokens during inference.
- The extension sends selected text to the configured API. Review the deployment
  and privacy policy before pointing it at a remote service.
- The current architecture does not retrieve or cite supporting sources.

A stronger fact-checking architecture would retrieve evidence and predict
`supported`, `refuted`, or `not enough information`, using claim/evidence data
such as FEVER.

## Troubleshooting

### Docker API is unavailable

Start Docker Desktop and check:

```powershell
docker compose ps
docker compose logs api --tail 100
```

### `/health` returns model unavailable

Confirm `MODEL_PATH` points to a directory inside the container and that the
weights were included in the build:

```powershell
docker compose exec -T api sh -c 'echo $MODEL_PATH && ls -lh $MODEL_PATH'
```

### The extension cannot connect

- Confirm <http://127.0.0.1:8081/health> works.
- Use `127.0.0.1`, not `localhost`, to avoid IPv6 resolution differences.
- Reload the extension and refresh the webpage.
- If you change the API port, update both `extension/background.js` and
  `host_permissions` in `extension/manifest.json`.

### Port 8081 is already in use

Change the host side of the API mapping in `docker-compose.yml`, then update the
extension URL and permission to match.

## Security and production notes

- Replace all default database credentials.
- Restrict CORS and extension host permissions to required origins.
- Terminate TLS at a trusted reverse proxy for remote deployments.
- Avoid logging sensitive selected text, or add redaction and retention rules.
- Store large model artifacts in controlled object storage or a versioned model
  registry rather than normal Git history.

## License

No license file is currently included. Add an explicit license before external
distribution or reuse.
