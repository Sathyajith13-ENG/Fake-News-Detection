"""Generate a SYNTHETIC demo dataset so the project can be run end to end.

The rows are template-made, not real journalism. FAKE rows mimic sensational
clickbait style, REAL rows mimic sourced wire-style reporting, and ~10% are
deliberately style-swapped. A model trained on this learns *writing style
only*; its metrics say nothing about real-world accuracy. Replace it with a
documented real dataset for meaningful work.

    python -m scripts.make_demo_data            # writes data/demo_news.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PLACES = ["Chennai", "Coimbatore", "Madurai", "Pune", "Leeds", "Toronto", "Osaka", "Nairobi", "Lisbon", "Perth"]
ORGS = ["city council", "state transport department", "national statistics office", "regional health board",
        "university research team", "central bank", "electricity regulator", "public library board"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "March", "May", "July", "September", "November"]
ACTIONS = ["extend bus services to the eastern suburbs", "repair two bridges over the river",
           "open a new vaccination clinic", "publish its annual budget online", "upgrade the water treatment plant",
           "introduce evening opening hours at libraries"]
FINDINGS = ["average commute times fell slightly after the new timetable", "air quality improved in three districts",
            "reading scores rose modestly among primary pupils", "hospital waiting times were unchanged"]
METRICS = ["retail sales", "household electricity use", "unemployment", "bus ridership", "housing permits"]
ACTORS = ["a famous politician", "top celebrity", "the mayor", "a billionaire", "a popular doctor", "the president"]
FAKE_ACTS = ["destroying evidence", "poisoning the water supply", "faking the entire event", "hiding alien technology"]
TOPICS = ["vaccines", "the election", "the moon landing", "your tap water", "the new currency", "5G towers"]
REMEDIES = ["lemon juice", "herbal tea", "this secret oil", "one weird fruit"]
SOBER_FAKE = ["the new banknotes will be withdrawn next week", "the reservoir has been secretly sold to a foreign firm",
              "all schools will close for a month due to an undisclosed threat"]

REAL = [
    "The {org} in {p} announced on {d} that it will {act}, according to an official statement. Officials said the "
    "plan follows a public consultation and will be reviewed in {m}.",
    "A study by a {org} found that {f}, the report published this week states. The authors noted the sample was "
    "limited and called for further research.",
    "According to data released by the {org}, {met} changed by {n} percent in the last quarter compared with the "
    "same period a year earlier. A spokesperson said the figures are subject to revision.",
    "{p} officials confirmed on {d} that the {org} will {act}. A spokesperson told reporters the budget was approved "
    "after a committee vote and the work is scheduled for {m}.",
]
FAKE = [
    "SHOCKING: {a} caught secretly {fa}! The mainstream media is hiding this from you. Share before they delete it!!!",
    "You won't believe what {a} is hiding about {t}. Doctors hate this one weird trick and the government wants it "
    "banned. Click now before it is gone!",
    "BREAKING!!! Insiders reveal {t} is a total hoax created by elites to control you. Wake up people, this is the "
    "truth they don't want you to see!",
    "Miracle {r} cures everything overnight and scientists are being silenced! Anonymous sources say {a} paid "
    "millions to cover it up. Spread the word!",
]
REAL_EMOTIONAL = ("I still cannot believe what happened in {p} this weekend, said a shaken resident, after the {org} "
                  "confirmed the road would reopen on {d}. Neighbours said they were heartbroken but relieved.")
FAKE_SOBER = ("The {org} in {p} quietly confirmed that {s}, according to a source familiar with the matter who "
              "asked not to be named. No documents were provided.")


def make_rows(n_per_class: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for _ in range(n_per_class):
        f = dict(p=rng.choice(PLACES), org=rng.choice(ORGS), d=rng.choice(DAYS), m=rng.choice(MONTHS),
                 act=rng.choice(ACTIONS), f=rng.choice(FINDINGS), met=rng.choice(METRICS), n=rng.randint(1, 9),
                 a=rng.choice(ACTORS), fa=rng.choice(FAKE_ACTS), t=rng.choice(TOPICS), r=rng.choice(REMEDIES),
                 s=rng.choice(SOBER_FAKE))
        rows.append(((REAL_EMOTIONAL if rng.random() < 0.1 else rng.choice(REAL)).format(**f), "REAL"))
        rows.append(((FAKE_SOBER if rng.random() < 0.1 else rng.choice(FAKE)).format(**f), "FAKE"))
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/demo_news.csv")
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label"])
        w.writerows(make_rows(a.per_class, a.seed))
    print(f"Wrote {2 * a.per_class} synthetic rows to {out}")


if __name__ == "__main__":
    main()
