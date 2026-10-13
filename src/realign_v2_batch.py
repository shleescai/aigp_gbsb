"""Run realign_v2.py (CPU only, unmodified) on every speaker whose audio/text counts still mismatch.

Targets : matched == False in notebook_run/relabel_compare/text_cleaned_report.csv (93 speakers, all of them,
          including the 5 large files that realign.py v1 could not finish)
Outputs : notebook_run/realign_v2/{stem}_units.csv / _clips.csv / _summary.json, plus batch_summary.csv
usage   : python realign_v2_batch.py [workers] [timeout_hours]
"""
import json, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = Path.home() / "DB/aihub_dysarthria/notebook_run"
AUDIO = Path.home() / "DB/aihub_dysarthria/013.구음장애_음성인식_데이터/01.데이터/1.Training/원천데이터"
LABEL = RUN / "Training/label_data/TL01/Peripheral_Neuropathy"
MODEL = Path.home() / "models/sherpa-onnx-zipformer-korean-2024-06-24"
OUT = RUN / "realign_v2"


def run_one(job):
    stem, wav, js, tmo = job
    pre = OUT / stem
    t0 = time.time()
    rec = dict(stem=stem, status="ok", size_gb=round(wav.stat().st_size / 1e9, 2))
    try:
        p = subprocess.run([sys.executable, str(HERE / "realign_v2.py"), str(wav), str(js), str(MODEL), str(pre),
                            "--threads", "2"], capture_output=True, text=True, timeout=tmo)
        if p.returncode != 0:
            rec["status"] = "error: " + (p.stderr.strip().splitlines() or ["?"])[-1][:200]
        else:
            s = json.load(open(f"{pre}_summary.json", encoding="utf-8"))
            uf = s["unit_flags"]
            rec.update(unit_mode=s["unit_mode"], n_units=s["n_units"], n_clips=s["n_clips"],
                       gap_thr=s["clip_gap_threshold_s"], ok=uf.get("OK", 0), low_conf=uf.get("LOW_CONF", 0),
                       boundary_guess=uf.get("BOUNDARY_GUESS", 0), not_found=uf.get("NOT_FOUND", 0),
                       coverage_ok=s["unit_coverage_ok"], coverage_assigned=s["unit_coverage_assigned"],
                       clip_usage=s["clip_usage"], runtime_s=s["runtime_s"])
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def main(workers=8, hours=4.0):
    OUT.mkdir(exist_ok=True)
    rep = pd.read_csv(RUN / "relabel_compare/text_cleaned_report.csv")
    codes = set(rep[~rep.matched].person_code)
    wavs = {p.stem: p for p in AUDIO.rglob("*") if p.suffix.lower() == ".wav"}
    jobs = [(js.stem, wavs[js.stem], js, hours * 3600) for js in sorted(LABEL.glob("*.json"))
            if js.stem.split("-N-")[1] in codes and js.stem in wavs]
    jobs.sort(key=lambda j: -j[1].stat().st_size)          # biggest first
    print(len(codes), "mismatched speakers |", len(jobs), "jobs", flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            print(f"[{len(rows)}/{len(jobs)}]", r["stem"], r["status"], r.get("coverage_ok"), r.get("wall_s"), flush=True)
            pd.DataFrame(rows).to_csv(OUT / "batch_summary.csv", index=False)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8, float(sys.argv[2]) if len(sys.argv) > 2 else 4.0)
