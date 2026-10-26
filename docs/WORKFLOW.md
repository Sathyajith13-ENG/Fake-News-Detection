# Workflow

```mermaid
flowchart LR
    A[Labeled CSV<br/>text,label] --> B[dataset.py<br/>validate labels]
    B --> C[Stratified train/test split]
    C --> D[model.py<br/>TF-IDF + Logistic Regression]
    D --> E[Evaluate on held-out test set]
    E --> F[(models/*.joblib<br/>pipeline + evaluation)]
    F --> G[app.py<br/>Streamlit interface]
    F --> H[api.py<br/>Flask API]
    G --> I[Verdict, marked-up article,<br/>quality, batch, assistant]
    H --> J[/predict /chat /health]
```

1. Train once with `python -m scripts.train_model` (or the **Train** tab in the interface).
2. The saved file holds both the fitted pipeline and the metrics measured at training time.
3. The interface and the API load that file; the API reloads it automatically after retraining.
