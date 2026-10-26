"""Streamlit interface for the fake news classifier.

    streamlit run app.py
"""
from __future__ import annotations

import html
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from fake_news_detection.chatbot import answer as chat_answer
from fake_news_detection.config import get_settings
from fake_news_detection.dataset import VALID_LABELS, DatasetError, load_dataset, validate_frame
from fake_news_detection.model import (ModelBundle, ModelNotFoundError, load_model, predict, predict_many,
                                       save_model, top_features, train_and_evaluate)
from scripts.make_demo_data import make_rows

SETTINGS = get_settings()
# Streamlit renamed use_container_width -> width="stretch" in 1.50; support both.
_V = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
FULL = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}
st.set_page_config(page_title="Fake news check", page_icon="📰", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&display=swap');
:root{--ink:#14213D;--muted:#5B6578;--line:#DDE2EA;--fake:#B42318;--fake-t:#FDE3E0;--real:#0F766E;--real-t:#DDF1EE;--warn:#8A5A00;--warn-t:#FFF4DB}
.stApp,.stApp p,.stApp label,.stApp li,.stApp textarea,.stApp button{font-family:'Public Sans',system-ui,-apple-system,'Segoe UI',sans-serif}
.block-container{max-width:1180px;padding-top:2.2rem}
footer,.stAppDeployButton{display:none!important}
.masthead h1{font-family:'Source Serif 4',Georgia,serif;font-size:2.3rem;font-weight:700;letter-spacing:-.01em;margin:0;padding:0}
.masthead p{color:var(--muted);margin:.25rem 0 1.2rem;font-size:1.02rem}
.stTabs [data-baseweb="tab-list"]{gap:1.6rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-weight:600;padding-left:0;padding-right:0}
.verdict{border-left:6px solid var(--ink);padding:.2rem 0 .2rem 1rem;margin-bottom:.9rem}
.verdict.fk{border-color:var(--fake)}.verdict.rl{border-color:var(--real)}.verdict.un{border-color:var(--warn)}
.vword{font-family:'Source Serif 4',Georgia,serif;font-size:2.5rem;font-weight:700;line-height:1.1}
.fk .vword{color:var(--fake)}.rl .vword{color:var(--real)}.un .vword{color:var(--warn)}
.vsub{color:var(--muted);margin-top:.3rem;font-size:.97rem}
.bar{display:flex;height:14px;border-radius:7px;overflow:hidden;background:#E8ECF2}
.bar .f{background:var(--fake)}.bar .r{background:var(--real)}
.barlabels{display:flex;justify-content:space-between;font-size:.9rem;font-weight:600;margin:.35rem 0 1rem}
.barlabels .f{color:var(--fake)}.barlabels .r{color:var(--real)}
.caution{background:var(--warn-t);border:1px solid #F0D9A0;color:#5C3D00;border-radius:8px;padding:.65rem .9rem;font-size:.92rem}
.paper{font-family:'Source Serif 4',Georgia,serif;font-size:1.1rem;line-height:1.8;background:#fff;border:1px solid var(--line);border-radius:10px;padding:1.3rem 1.6rem;max-width:74ch}
mark.fk,mark.rl{padding:0 3px;border-radius:2px;color:inherit}
mark.fk{background:var(--fake-t);border-bottom:2px solid var(--fake)}
mark.rl{background:var(--real-t);border-bottom:2px dashed var(--real)}
.legend{color:var(--muted);font-size:.9rem;margin:.2rem 0 .7rem}
.chip{display:inline-block;margin:0 .4rem .4rem 0;padding:.15rem .65rem;border-radius:999px;font-size:.88rem;border:1px solid}
.chip.fk{background:var(--fake-t);border-color:var(--fake);color:#7A1810}.chip.rl{background:var(--real-t);border-color:var(--real);color:#0B4F49}
.tb{display:grid;grid-template-columns:8.5rem 1fr;gap:.6rem;align-items:center;margin:.28rem 0;font-size:.92rem}
.tb .t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tb .b{height:10px;border-radius:5px}
table.cm{border-collapse:collapse;width:100%;text-align:center;font-size:.95rem}
table.cm th{font-weight:600;color:var(--muted);padding:.4rem}.cm td{padding:.9rem .4rem;border:1px solid var(--line);font-weight:700;font-size:1.15rem}
.empty{border:1px dashed #B9C2D0;border-radius:10px;padding:1.4rem;color:var(--muted)}
</style>
"""

SAMPLES = {
    "Sourced report": (
        "The Reserve Bank held its benchmark interest rate steady on Wednesday, saying inflation remained within "
        "its target range. The governor told reporters the board would review conditions again next quarter, "
        "according to a statement published on the bank's website."),
    "Clickbait claim": (
        "SHOCKING: Scientists are hiding the truth about your tap water! The mainstream media won't tell you this "
        "one weird trick. Share before they delete it!!!"),
    "Sober-sounding rumor": (
        "The health ministry has quietly confirmed that all tap water in the capital will be replaced by bottled "
        "supply next month, according to a person familiar with the matter who asked not to be named."),
}
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- model access
@st.cache_resource(show_spinner=False)
def _load(path: str, mtime: float) -> ModelBundle:
    return load_model(path)


def get_bundle() -> tuple[ModelBundle | None, str | None]:
    try:
        mtime = SETTINGS.model_path.stat().st_mtime
    except OSError:
        return None, "No trained model found yet. Open the Train tab, or run python -m scripts.train_model."
    try:
        return _load(str(SETTINGS.model_path), mtime), None
    except ModelNotFoundError as exc:
        return None, str(exc)


# --------------------------------------------------------------------- helpers
def highlight(text: str, terms: list[dict]) -> str:
    """Escape the article and mark the words/phrases that moved the score."""
    lookup = {t["term"]: t for t in terms}
    if not lookup:
        return html.escape(text).replace("\n", "<br>")
    alts = sorted(lookup, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(r"\W+".join(map(re.escape, a.split())) for a in alts) + r")\b(?:['’]\w+)?", re.I)
    out, last = [], 0
    for m in pattern.finditer(text):
        key = re.sub(r"['’]\w+$", "", m.group(0).lower())
        t = lookup.get(re.sub(r"\W+", " ", key).strip())
        if not t:
            continue
        cls = "fk" if t["leans"] == "FAKE" else "rl"
        out.append(html.escape(text[last:m.start()]))
        out.append(f'<mark class="{cls}" title="Pushes toward {t["leans"]}">{html.escape(m.group(0))}</mark>')
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


def verdict_html(res: dict) -> str:
    label, conf = res["label"], res["confidence"]
    if conf < 0.6:
        cls, word, sub = "un", "Too close to call", f"Slightly nearer {label} ({conf:.0%}). Treat this as no signal."
    else:
        cls = "fk" if label == "FAKE" else "rl"
        word = f"Leans {label}"
        sub = f"{conf:.0%} classifier score. This describes writing patterns, not whether the claims are true."
    return f'<div class="verdict {cls}"><div class="vword">{word}</div><div class="vsub">{sub}</div></div>'


def score_bar(scores: dict) -> str:
    f, r = scores.get("FAKE", 0) * 100, scores.get("REAL", 0) * 100
    return (f'<div class="bar" role="img" aria-label="FAKE {f:.0f} percent, REAL {r:.0f} percent">'
            f'<span class="f" style="width:{f:.1f}%"></span><span class="r" style="width:{r:.1f}%"></span></div>'
            f'<div class="barlabels"><span class="f">FAKE {f:.0f}%</span><span class="r">REAL {r:.0f}%</span></div>')


def term_bars(items: list[tuple[str, float]], cls: str) -> str:
    top = max((w for _, w in items), default=1) or 1
    color = "var(--fake)" if cls == "FAKE" else "var(--real)"
    return "".join(f'<div class="tb"><span class="t" title="{html.escape(t)}">{html.escape(t)}</span>'
                   f'<span class="b" style="width:{max(w / top * 100, 3):.0f}%;background:{color}"></span></div>'
                   for t, w in items)


def confusion_html(ev: dict) -> str:
    labs, cm = ev["labels"], ev["confusion_matrix"]
    head = "<tr><th></th>" + "".join(f"<th>Predicted {l}</th>" for l in labs) + "</tr>"
    rows = ""
    for i, row in enumerate(cm):
        tot = sum(row) or 1
        cells = ""
        for j, v in enumerate(row):
            base = "15,118,110" if i == j else "180,35,24"
            cells += f'<td style="background:rgba({base},{0.08 + 0.55 * v / tot:.2f})">{v}</td>'
        rows += f"<tr><th>True {labs[i]}</th>{cells}</tr>"
    return f'<table class="cm">{head}{rows}</table>'


def render_evaluation(ev: dict, bundle: ModelBundle | None = None) -> None:
    c = st.columns(4)
    c[0].metric("Accuracy", f"{ev['accuracy']:.1%}")
    c[1].metric("Macro precision", f"{ev['macro_precision']:.1%}")
    c[2].metric("Macro recall", f"{ev['macro_recall']:.1%}")
    c[3].metric("Macro F1", f"{ev['macro_f1']:.1%}")
    if ev["accuracy"] >= 0.99 or ev["n_test"] < 50:
        st.warning("A near-perfect score, or a test set under 50 rows, usually means the data is easy, repetitive "
                   "or too small to judge. Check for near-duplicate rows before trusting these numbers.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix** (held-out test set)")
        st.markdown(confusion_html(ev), unsafe_allow_html=True)
    with right:
        st.markdown("**Per class**")
        per = pd.DataFrame(ev["per_class"]).T.rename_axis("class")
        per["support"] = per["support"].astype(int)
        st.dataframe(per.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}), **FULL)
        st.caption(f"{ev['n_train']} training rows, {ev['n_test']} test rows, stratified split "
                   f"(seed {ev['random_state']}). Trained {fmt_date(ev['trained_at'])}.")
    if bundle:
        st.markdown("**Terms the model leans on most**")
        tf = top_features(bundle, 10)
        a, b = st.columns(2)
        a.markdown("Point toward FAKE")
        a.markdown(term_bars(tf["FAKE"], "FAKE"), unsafe_allow_html=True)
        b.markdown("Point toward REAL")
        b.markdown(term_bars(tf["REAL"], "REAL"), unsafe_allow_html=True)


def fmt_date(iso: str | None) -> str:
    try:
        return datetime.fromisoformat(str(iso)).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return "unknown date"


def snippet(text: str, n: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ------------------------------------------------------------------------ page
bundle, model_error = get_bundle()
st.session_state.setdefault("article", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("chat", [])

st.markdown('<div class="masthead"><h1>Fake news check</h1>'
            "<p>Paste an article to see how the model reads it and which words moved the result.</p></div>",
            unsafe_allow_html=True)

t_analyze, t_model, t_batch, t_train, t_ask = st.tabs(["Analyze", "Model quality", "Batch", "Train", "Ask"])

# ---- Analyze
with t_analyze:
    def _load_sample() -> None:
        choice = st.session_state.get("sample")
        if choice in SAMPLES:
            st.session_state.article = SAMPLES[choice]

    def _clear() -> None:
        st.session_state.article = ""
        st.session_state.pop("last", None)

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.selectbox("Start from an example", ["Choose an example"] + list(SAMPLES), key="sample",
                     on_change=_load_sample, help="Examples show writing styles only; they are not verified stories.")
        st.text_area("Article text", key="article", height=260, placeholder="Paste the article text here…")
        words = len(st.session_state.article.split())
        b1, b2, b3 = st.columns([1.8, 1, 1.2])
        go = b1.button("Analyze article", type="primary", disabled=bundle is None, **FULL)
        b2.button("Clear", on_click=_clear, **FULL)
        b3.caption(f"{words:,} words")
        if bundle is None:
            st.error(model_error)

    if go:
        try:
            res = predict(bundle, st.session_state.article)
            st.session_state["last"] = {"text": st.session_state.article, "res": res}
            st.session_state.history.insert(0, {"Article": snippet(st.session_state.article, 70),
                                                "Result": res["label"], "Score": f"{res['confidence']:.0%}"})
            st.session_state.history = st.session_state.history[:20]
        except ValueError:
            st.warning("Paste some article text first.")

    last = st.session_state.get("last")
    with right:
        if last:
            res = last["res"]
            st.markdown(verdict_html(res) + score_bar(res["scores"]), unsafe_allow_html=True)
            st.markdown(f'<div class="caution">{html.escape(res["caution"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">Select <b>Analyze article</b> to see the verdict, how sure the model is, '
                        "and the exact words that pushed it each way.</div>", unsafe_allow_html=True)

    if last:
        st.markdown("#### The article, marked up")
        st.markdown('<div class="legend"><mark class="fk">Solid red</mark> pushes toward FAKE &nbsp; '
                    '<mark class="rl">Dashed teal</mark> pushes toward REAL. Unmarked words had little effect.</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="paper">{highlight(last["text"], last["res"]["terms"])}</div>',
                    unsafe_allow_html=True)
        chips = "".join(f'<span class="chip {"fk" if t["leans"] == "FAKE" else "rl"}">{html.escape(t["term"])}</span>'
                        for t in last["res"]["terms"])
        st.markdown('<div class="legend" style="margin-top:1rem">Strongest words, most influential first</div>' + chips,
                    unsafe_allow_html=True)
    if st.session_state.history:
        with st.expander(f"This session ({len(st.session_state.history)})"):
            st.dataframe(pd.DataFrame(st.session_state.history), **FULL, hide_index=True)

# ---- Model quality
with t_model:
    if bundle and bundle.evaluation:
        render_evaluation(bundle.evaluation, bundle)
        st.caption("These numbers come from your dataset and split only. They are not a general accuracy claim.")
    else:
        st.info("Train a model to see its measured quality here.")

# ---- Batch
with t_batch:
    st.write("Score many articles at once. Upload a CSV that has a column of article text.")
    up = st.file_uploader("CSV file", type="csv", key="batch_up")
    if up is not None and bundle is None:
        st.error(model_error)
    elif up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
            st.error(f"Could not read this CSV: {exc}")
        else:
            cols = list(df.columns)
            col = st.selectbox("Text column", cols, index=cols.index("text") if "text" in cols else 0)
            if len(df) > 5000:
                st.warning("Only the first 5,000 rows are scored.")
                df = df.head(5000)
            preds = predict_many(bundle, df[col].tolist())
            out = pd.DataFrame({"text": df[col], "prediction": [p["label"] for p in preds],
                                "score": [p["confidence"] for p in preds]})
            skipped = int(out["prediction"].isna().sum())
            m = st.columns(3)
            m[0].metric("Rows scored", f"{len(out) - skipped:,}")
            m[1].metric("Predicted FAKE", int((out["prediction"] == "FAKE").sum()))
            m[2].metric("Predicted REAL", int((out["prediction"] == "REAL").sum()))
            if skipped:
                st.caption(f"{skipped} empty row(s) were skipped.")
            if "label" in df.columns:
                truth = df["label"].astype(str).str.strip().str.upper()
                ok = out["prediction"].notna() & truth.isin(VALID_LABELS)
                if ok.any():
                    st.info(f"Agreement with your label column: {(truth[ok] == out['prediction'][ok]).mean():.1%} "
                            f"over {int(ok.sum())} rows. This is only an honest test if these rows were not used "
                            "for training.")
            view = out.assign(text=out["text"].map(snippet), score=out["score"].round(3))
            st.dataframe(view, **FULL, hide_index=True)
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

# ---- Train
with t_train:
    st.write("Train or retrain the model. The new model is saved and used for the next analysis.")
    source = st.radio("Dataset", ["CSV file on disk", "Upload a CSV", "Synthetic demo data"], horizontal=True)
    default_path = next((p for p in (SETTINGS.data_path, SETTINGS.data_path.parent / "demo_news.csv")
                         if p.is_file()), SETTINGS.data_path)
    path_text, train_up = "", None
    if source == "CSV file on disk":
        path_text = st.text_input("Path", value=str(default_path))
    elif source == "Upload a CSV":
        train_up = st.file_uploader("Labeled CSV with `text` and `label` columns", type="csv", key="train_up")
    else:
        st.warning("The demo data is template-generated. A model trained on it learns writing style only, and its "
                   "scores say nothing about real-world accuracy.")
    test_size = st.slider("Share held out for testing", 0.1, 0.4, 0.2, 0.05)
    if st.button("Train model", type="primary"):
        try:
            if source == "CSV file on disk":
                data = load_dataset(path_text)
            elif source == "Upload a CSV":
                if train_up is None:
                    raise DatasetError("Choose a CSV file to upload first.")
                data = validate_frame(pd.read_csv(train_up))
            else:
                data = pd.DataFrame(make_rows(150, 7), columns=["text", "label"])
            with st.spinner("Training…"):
                pipe, ev = train_and_evaluate(data, test_size)
                save_model(pipe, ev, SETTINGS.model_path)
            st.success(f"Trained on {ev['n_train']} rows and saved to {SETTINGS.model_path.name}.")
            render_evaluation(ev, ModelBundle(pipe, ev))
        except (DatasetError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            st.error(str(exc))

# ---- Ask
with t_ask:
    share = st.toggle("Share my latest result with the assistant", value=True)
    if SETTINGS.ai_configured:
        st.caption("An AI provider is configured: your question and shared result are sent to it. "
                   "Avoid sensitive content.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("note"):
                st.caption(msg["note"])
    q = st.chat_input("Ask about the model, the metrics, or your result")
    if q:
        ctx = None
        if share and st.session_state.get("last"):
            r = st.session_state["last"]["res"]
            ctx = (f"Classified {r['label']} with score {r['confidence']:.0%}; influential terms: "
                   + ", ".join(t["term"] for t in r["terms"][:6]))
        reply = chat_answer(q, ctx, SETTINGS)
        st.session_state.chat += [{"role": "user", "content": q},
                                  {"role": "assistant", "content": reply["answer"], "note": reply["note"]}]
        st.rerun()

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Model")
    if bundle:
        ev = bundle.evaluation
        st.success("Loaded and ready")
        if ev:
            st.caption(f"Trained {fmt_date(ev.get('trained_at'))} on {ev.get('n_train', '?')} rows.")
    else:
        st.error("No model available")
    st.markdown("### Assistant")
    st.caption("Remote AI provider" if SETTINGS.ai_configured else "Local assistant (no AI provider configured)")
    st.markdown("### Read results with care")
    st.caption("The classifier learns patterns from labeled examples. It can be wrong, biased, or out of date, and it "
               "does not verify facts. Check original records and several independent sources.")
