"""Clean AI-Hub transcripts following the labeling guide and write segment texts for speakers whose
text segment count equals the audio segment count produced by the Preprocess notebook.

Cleaning rule:
  1. remove tag forms: letter+"/" (u/ b/ n/ o/ l/), "/"+letter (/L /b /n), and any leftover "/"
  2. remove every symbol except Hangul, digits, Latin letters, whitespace and  . , ? !
Splitting: by . ? ! (the punctuation stays on the segment) if present, otherwise by whitespace.
Output: notebook_run/Text_Cleaned/output_PN_{person_code}_{i}.txt  (existing outputs are untouched)
"""
import json, re
from pathlib import Path
import pandas as pd

RUN = Path.home() / "DB/aihub_dysarthria/notebook_run"
LABEL = RUN / "Training/label_data/TL01/Peripheral_Neuropathy"
V2 = RUN / "review_targets_VAD_Peripheral_Neuropathy_v2.csv"
OUT = RUN / "Text_Cleaned"


# Clear labeling error confirmed by hand: unfinished spelling inside a word list
ERROR_TOKENS = [r"\bsilen\b"]


def clean(t):
    for pat in ERROR_TOKENS:
        t = re.sub(pat, " ", t)
    t = re.sub(r"<U\+[0-9A-Fa-f]{4,6}>", " ", t)   # broken unicode escapes exported as text
    t = re.sub(r"(?<![A-Za-z])[A-Za-z]/", " ", t)
    t = re.sub(r"/[LlBbNn](?![A-Za-z])", " ", t)
    t = t.replace("/", " ")
    t = re.sub(r"[^0-9A-Za-z가-힣\s\.,\?!]", "", t)
    return t


def split(t):
    t = clean(str(t).strip())
    parts = re.findall(r"[^\.\?!]+[\.\?!]*", t) if re.search(r"[\.\?!]", t) else t.split()
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if re.sub(r"[,\s\.\?!]", "", p):
            out.append(p)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    rows, n_files = [], 0
    for r in pd.read_csv(V2).itertuples():
        segs = split(json.load(open(LABEL / r.file, encoding="utf-8"))["Transcript"])
        ok = len(segs) == r.audio_files
        if ok:
            for i, s in enumerate(segs):
                (OUT / f"output_PN_{r.person_code}_{i}.txt").write_text(s, encoding="utf-8")
            n_files += len(segs)
        rows.append(dict(person_code=r.person_code, audio=r.audio_files, text=len(segs), matched=ok))
    df = pd.DataFrame(rows)
    df.to_csv(RUN / "relabel_compare/text_cleaned_report.csv", index=False)
    print("matched speakers", int(df.matched.sum()), "/", len(df), "| txt files written", n_files)


if __name__ == "__main__":
    main()
