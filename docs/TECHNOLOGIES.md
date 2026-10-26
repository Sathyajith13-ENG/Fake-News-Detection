# Technologies

| Piece | Used for |
|---|---|
| Python 3.10+ | Language for every module |
| pandas | CSV loading, validation, batch results |
| scikit-learn | `TfidfVectorizer`, `LogisticRegression`, `Pipeline`, stratified split, metrics |
| joblib | Saving the trained pipeline together with its measured evaluation record |
| Streamlit | The interface (`app.py`): analysis, model quality, batch, training, assistant |
| Flask | The JSON API (`api.py`): `/health`, `/predict`, `/chat` |
| unittest | Tests, standard library only (`tests/test_core.py`) |

## How the model works

1. **Normalize** (`preprocessing.py`): lowercase, unescape HTML, drop tags and URLs, collapse whitespace.
2. **TF-IDF** unigrams and bigrams with English stop words removed and sublinear term frequency.
3. **Logistic Regression** (balanced class weights) learns one weight per term. The weighted sum becomes a
   FAKE/REAL score.
4. **Explanation**: for one article, each present term's contribution is its TF-IDF value times its learned
   weight. The largest contributions are the "words that moved the score" shown and highlighted in the interface.

## What it cannot do

It learns writing style and vocabulary from labeled examples. It does not check facts, and its scores are not
calibrated probabilities.
