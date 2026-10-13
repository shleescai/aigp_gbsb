"""Run realign.py (CPU only, unmodified) on every speaker whose audio/text counts still mismatch.

Targets: matched == False in notebook_run/relabel_compare/text_cleaned_report.csv
Outputs: notebook_run/realign/{stem}_realign.csv  and  notebook_run/realign/summary.csv
usage:   python realign_batch.py [workers]
"""
import contextlib, io, re, sys, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realign

RUN = Path.home() / "DB/aihub_dysarthria/notebook_run"
AUDIO = Path.home() / "DB/aihub_dysarthria/013.구음장애_음성인식_데이터/01.데이터/1.Training/원천데이터"
LABEL = RUN / "Training/label_data/TL01/Peripheral_Neuropathy"
MODEL = Path.home() / "models/sherpa-onnx-zipformer-korean-2024-06-24"
OUT = RUN / "realign"


def run_one(job):
    stem, wav, js = job
    out = OUT / f"{stem}_realign.csv"
    buf = io.StringIO()
    rec = dict(stem=stem, status="ok", clips=None, words=None, gap_median=None, gap_sd=None)
    try:
        with contextlib.redirect_stdout(buf):
            realign.main(str(wav), str(js), str(MODEL), str(out))
        m = re.search(r"clips=(\d+) words=(\d+) \| silence gap median=([\d.]+)s sd=([\d.]+)", buf.getvalue())
        if m:
            rec.update(clips=int(m[1]), words=int(m[2]), gap_median=float(m[3]), gap_sd=float(m[4]))
        df = pd.read_csv(out)
        for k, v in df.flag.fillna("NONE").value_counts().items():
            rec[k] = int(v)
    except Exception as e:
        rec["status"] = f"error: {type(e).__name__}: {e}"
    return rec


def main(workers=8):
    OUT.mkdir(exist_ok=True)
    rep = pd.read_csv(RUN / "relabel_compare/text_cleaned_report.csv")
    codes = rep[~rep.matched].person_code.tolist()
    wavs = {p.stem: p for p in AUDIO.rglob("*") if p.suffix.lower() == ".wav"}
    jobs = []
    for js in sorted(LABEL.glob("*.json")):
        code = js.stem.split("-N-")[1]
        if code in codes and js.stem in wavs:
            jobs.append((js.stem, wavs[js.stem], js))
    print(len(codes), "mismatched speakers |", len(jobs), "with audio+label", flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_one, jobs):
            rows.append(r)
            print(r["stem"], r["status"], r.get("clips"), r.get("words"), r.get("gap_median"), r.get("gap_sd"), flush=True)
            pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print("done", len(rows))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
