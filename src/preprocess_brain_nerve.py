"""Preprocess the AI-Hub 뇌신경장애 (TS01/TL01) data following the paper's pipeline.

1) RMS-energy VAD (frame_ms=500, silence_duration_sec=5, alpha=0.3)
2) Split the transcript by "/" -> sentence punctuation -> whitespace
3) Audio/text segment counts must match; mismatches are re-run with (600, 3, 0.4)
4) Still-mismatched files go to needs_manual_review.csv (no output written)
5) Matched segments are saved as output_PN_{person_code}_{i}.wav (16 kHz mono PCM16)
   and cleaned text as output_PN_{person_code}_{i}.txt
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path.home() / "DB/aihub_dysarthria/013.구음장애_음성인식_데이터/01.데이터/1.Training"
AUDIO_ROOT = ROOT / "원천데이터"
LABEL_ROOT = ROOT / "라벨링데이터"
OUT = Path.home() / "DB/aihub_dysarthria/preprocessed_brain_nerve"
TARGET_SR = 16000
PASSES = [(500, 5, 0.3), (600, 3, 0.4)]
NAME_RE = re.compile(r"ID-\d{2}-\d{2}-N-(.+)\.(wav|json)$", re.I)


def vad_segments(y, sr, frame_ms, silence_duration_sec, alpha):
    frame_len = int(sr * frame_ms / 1000)
    total = len(y)
    starts = range(0, total, frame_len)
    energy = np.array([np.sqrt(np.mean(np.square(y[i:i + frame_len], dtype=np.float64))) for i in starts])
    threshold = energy.mean() * alpha
    silent = energy < threshold
    min_frames = int(silence_duration_sec / (frame_ms / 1000))

    segments, is_silent, start = [], False, 0
    for idx, s in enumerate(silent):
        if not is_silent and s:
            run = silent[idx:idx + min_frames]
            if len(run) == min_frames and run.all():
                end = idx * frame_len
                if end - start > 0:
                    segments.append((start, end))
                is_silent = True
        elif is_silent and not s:
            start = idx * frame_len
            is_silent = False
    if not is_silent and start < total:
        segments.append((start, total))
    return segments


def split_transcript(text):
    text = str(text).strip()
    if "/" in text:
        parts = text.split("/")
    elif re.search(r"[\.?!]", text):
        parts = re.split(r"[\.?!]", text)
    else:
        parts = text.split()
    return [p.strip() for p in parts if p.strip()]


def load_16k_mono(path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)
    return y


def process(args):
    wav_path, json_path = args
    code = NAME_RE.match(Path(wav_path).name).group(1)
    try:
        texts = split_transcript(json.load(open(json_path, encoding="utf-8"))["Transcript"])
        y = load_16k_mono(wav_path)
        for n_pass, (fm, sd, a) in enumerate(PASSES, 1):
            segs = vad_segments(y, TARGET_SR, fm, sd, a)
            if len(segs) == len(texts):
                break
        else:
            return dict(person_code=code, status="mismatch", text_segments=len(texts), audio_segments=len(segs), passes=n_pass)

        for i, ((s, e), t) in enumerate(zip(segs, texts)):
            sf.write(OUT / "audio" / f"output_PN_{code}_{i}.wav", y[s:e], TARGET_SR, subtype="PCM_16")
            clean = re.sub(r"[+\*\(\)\?!,\.~\-']", "", t)
            (OUT / "text" / f"output_PN_{code}_{i}.txt").write_text(clean, encoding="utf-8")
        return dict(person_code=code, status="ok", text_segments=len(texts), audio_segments=len(segs), passes=n_pass)
    except Exception as ex:  # keep going; report in the CSV
        return dict(person_code=code, status=f"error: {ex}", text_segments=-1, audio_segments=-1, passes=0)


def main(workers=8):
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "text").mkdir(parents=True, exist_ok=True)
    labels = {p.stem: p for p in LABEL_ROOT.rglob("*.json")}
    jobs, missing = [], []
    for w in sorted(p for p in AUDIO_ROOT.rglob("*") if p.suffix.lower() == ".wav"):
        (jobs if w.stem in labels else missing).append((str(w), str(labels[w.stem])) if w.stem in labels else w.name)
    codes = [NAME_RE.match(Path(j[0]).name).group(1) for j in jobs]
    dup = {c for c in codes if codes.count(c) > 1}
    print(f"{len(jobs)} audio/label pairs, {len(missing)} without label, duplicate codes: {sorted(dup)}", flush=True)

    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(process, jobs):
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "preprocess_report.csv", index=False)
    df[df.status != "ok"].to_csv(OUT / "needs_manual_review.csv", index=False)
    print(df.status.value_counts().to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
