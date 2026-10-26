# Fake News Detection System

A Python text-classification project with a Streamlit interface, Flask JSON API,
Scikit-learn model training/evaluation, and an optional AI chatbot. The model
returns a **FAKE** or **REAL** prediction based on labeled training examples.
It is not a fact-checking service and its output is not evidence that an article
is true or false.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.train_model --data data/demo_news.csv
streamlit run app.py
```

`data/demo_news.csv` is **synthetic, template-generated** data so the project runs immediately. A model trained on
it learns writing style only and its scores say nothing about real-world accuracy. Replace it with a real,
documented dataset (see below). On macOS/Linux, activate with `source .venv/bin/activate`.

## The interface

| Tab | What it does |
|---|---|
| Analyze | Paste an article. Shows the verdict, the FAKE/REAL score split, and the article re-displayed with the words that pushed the score marked (solid red = toward FAKE, dashed teal = toward REAL). Scores under 60% show as "Too close to call". |
| Model quality | Accuracy, macro precision/recall/F1, confusion matrix, per-class table, and the terms the model leans on most. Warns when scores look too good to trust. |
| Batch | Upload a CSV, score every row, download the results. If your file has a `label` column it reports agreement. |
| Train | Train or retrain from a path, an uploaded CSV, or the synthetic demo data. |
| Ask | Chat about the model and your result (local assistant, or your AI provider if configured). |

## Project structure

```text
.
├── app.py                         # Streamlit frontend
├── api.py                         # Flask prediction and chat API
├── fake_news_detection/
│   ├── chatbot.py                 # Optional AI integration and local fallback
│   ├── config.py                  # Environment-backed configuration
│   ├── dataset.py                 # CSV loading and validation
│   ├── model.py                   # TF-IDF pipeline, training, evaluation, inference
│   └── preprocessing.py           # Shared text normalization
├── scripts/train_model.py         # Training/evaluation CLI
├── scripts/make_demo_data.py      # Synthetic demo dataset generator
├── tests/test_core.py             # Focused tests
├── data/                          # Place your labeled CSV here
├── models/                        # Generated trained model (not committed)
├── docs/TECHNOLOGIES.md           # Technology explanation
├── docs/WORKFLOW.md               # Workflow diagram
├── .streamlit/config.toml         # Interface theme
├── .env.example                   # Configuration template
└── requirements.txt
```

## Dataset format

Supply a UTF-8 CSV with one article per row and columns named `text` and `label`.
Labels are case-insensitive and must be `FAKE` or `REAL`.

```csv
text,label
"Example article text for a documented public announcement.",REAL
"Example article text carrying a claim labeled for your dataset.",FAKE
```

These rows illustrate the format only; their labels are not a factual judgment. Use a responsibly sourced,
documented dataset and ensure the label meanings, licensing, coverage, and quality are appropriate. Training
rejects other labels and requires at least two examples of each class; larger, balanced data is needed for
meaningful assessment. Put your CSV at `data/news.csv` or use `--data` / `DATA_PATH`.

## Configuration

Optionally copy `.env.example` to `.env`. `DATA_PATH` and `MODEL_PATH` override the default CSV and model locations.

The chatbot works locally without any API credentials. To enable a remote OpenAI-compatible chat-completions
service, set `AI_API_URL`, `AI_API_KEY`, `AI_MODEL`, and optionally `AI_TIMEOUT_SECONDS`. Never commit `.env`.
When enabled, the question and any prediction context are sent to the provider; do not send sensitive content.
If configuration is incomplete or the service fails, the local assistant answers and says it is the fallback.

## Train and evaluate

```powershell
python -m scripts.train_model --data data/news.csv
```

Optional arguments: `--text-column text`, `--label-column label`, `--test-size 0.2`,
`--model-out models/fake_news_pipeline.joblib`.

The script uses a reproducible stratified split, fits a TF-IDF unigram/bigram + Logistic Regression pipeline,
prints accuracy, macro precision/recall/F1, per-class metrics and a confusion matrix, then saves the pipeline and
its measured evaluation record. These metrics describe the dataset and split you run; this repository claims no
pre-established performance. A tiny dataset cannot support a credible assessment, and repeated tuning against the
holdout leaks evaluation information.

## Run the apps

```powershell
streamlit run app.py     # interface
python api.py            # API at http://127.0.0.1:5000
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict `
  -ContentType "application/json" -Body '{"text":"Paste article text here"}'

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/chat `
  -ContentType "application/json" -Body '{"question":"How does TF-IDF work?"}'
```

`GET /health` checks whether a saved model can be loaded (200 or 503). `POST /predict` needs a non-empty JSON
`text` string and returns the prediction, per-class scores, influential terms and a caution. `POST /chat` takes
`question` and optional string `prediction_context`, and reports whether the answer came from the provider or the
local fallback. Invalid input returns a readable error; a missing model returns HTTP 503 with training
instructions. The Flask built-in server is for local development only; use a production WSGI server and network
controls before exposing it.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Responsible use

Predictions depend on the training data and can be wrong, biased, or stale. Scores are classifier outputs and may
not be calibrated. Neither the prediction nor the chatbot verifies an article. Confirm claims using original
records and multiple credible, independent sources.

See [docs/TECHNOLOGIES.md](docs/TECHNOLOGIES.md) and [docs/WORKFLOW.md](docs/WORKFLOW.md).


python -m pip install -r requirements.txt
python -m scripts.train_model --data data/demo_news.csv
streamlit run app.py