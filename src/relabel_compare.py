"""Re-split transcripts using the AI-Hub labeling guide (guidedescription.pdf p.60) and compare the
segment counts with the audio segment counts already produced by the Preprocess notebook (v2 CSV).

Guide tags: /L(laugh) /b(cough) /n(noise) +(repeat) *(speech masked).  In the data these appear as
`b/`, `n/`, `u/`, `o/`, `l/` (letter followed by "/"), so "/" is a tag marker, not a sentence delimiter.
Nothing under notebook_run/Preprocessed or Text_Preprocessed is modified.
"""
import json, re
from pathlib import Path
import pandas as pd

RUN = Path.home() / "DB/aihub_dysarthria/notebook_run"
LABEL = RUN / "Training/label_data/TL01/Peripheral_Neuropathy"
V2 = RUN / "review_targets_VAD_Peripheral_Neuropathy_v2.csv"
OUT = RUN / "relabel_compare"


def split_original(t):
    """Rule used by the Preprocess notebook."""
    if "/" in t:
        parts = t.split("/")
    elif re.search(r"[\.?!]", t):
        parts = re.split(r"[\.?!]", t)
    else:
        parts = t.split()
    return [p.strip() for p in parts if p.strip()]


def strip_tags(t):
    t = re.sub(r"<U\+[0-9A-Fa-f]{4,6}>", " ", t)   # broken unicode escapes exported as text
    t = re.sub(r"(?<![A-Za-z])[A-Za-z]/", " ", t)   # u/ b/ n/ o/ l/ tags
    t = re.sub(r"/[LlBbNn](?![A-Za-z])", " ", t)     # guide form: /L /b /n
    t = t.replace("/", " ")                           # any leftover "/" is not a delimiter
    t = re.sub(r"[+\*]", "", t)                       # + repeat, * masked
    return re.sub(r"[\(\)~\-']", "", t)


def strip_all_alpha(t):
    """strip_tags + remove every Latin letter (tags, and any real English words)."""
    return re.sub(r"[A-Za-z]+", "", strip_tags(t))


def _split(t, strip):
    t = strip(t)
    parts = re.split(r"[\.?!]", t) if re.search(r"[\.?!]", t) else t.split()
    return [re.sub(r"[,\s]+", " ", p).strip() for p in parts if re.sub(r"[,\s]+", " ", p).strip()]


def split_guide(t):
    return _split(t, strip_tags)


def split_noalpha(t):
    return _split(t, strip_all_alpha)


def main():
    OUT.mkdir(exist_ok=True)
    v2 = pd.read_csv(V2)
    rows = []
    for r in v2.itertuples():
        t = str(json.load(open(LABEL / r.file, encoding="utf-8"))["Transcript"]).strip()
        n_orig, n_guide, n_alpha = len(split_original(t)), len(split_guide(t)), len(split_noalpha(t))
        rows.append(dict(file=r.file, person_code=r.person_code, audio=r.audio_files,
                         text_original=n_orig, text_guide=n_guide, gap_original=r.audio_files - n_orig,
                         gap_guide=r.audio_files - n_guide, match_original=r.audio_files == n_orig,
                         match_guide=r.audio_files == n_guide, text_noalpha=n_alpha,
                         gap_noalpha=r.audio_files - n_alpha, match_noalpha=r.audio_files == n_alpha,
                         has_alpha=bool(re.search(r"[A-Za-z]", t)), has_slash="/" in t,
                         has_sentence_punct=bool(re.search(r"[\.?!]", t))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "relabel_compare.csv", index=False)
    print(len(df), "files | original match", int(df.match_original.sum()), "| guide match", int(df.match_guide.sum()),
          "| guide+no-alphabet match", int(df.match_noalpha.sum()))
    print(pd.crosstab(df.match_original, df.match_guide).to_string())
    for tol in (0, 1, 2, 3):
        print(f"|gap|<={tol}: original {(df.gap_original.abs()<=tol).sum()}  guide {(df.gap_guide.abs()<=tol).sum()}"
              f"  no-alphabet {(df.gap_noalpha.abs()<=tol).sum()}")


if __name__ == "__main__":
    main()
