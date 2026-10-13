"""
realign_v2.py  -  AIHub 구음장애 낭독 데이터 재정렬 (CPU 전용)

v1 대비 변경점
- 정렬 단위: 대본에 문장부호가 있으면 '문장', 없으면 '단어' (자동 판별, --unit 으로 강제 가능)
- 1단계(클립 배정): 삽입된 긴 무음으로 나뉜 클립에 단위 1~K개를 DP로 배정 (K 상한 없음, 길이로 제한)
  큰 파일은 대각선 밴드만 계산 + rapidfuzz(C 구현) 사용 -> v1 대비 수백 배 빠름
- 2단계(클립 내부 분할): 한 클립에 단위가 여러 개면, ASR 토큰 타임스탬프와 문자 정렬로
  단위 경계 시간을 구하고, 가장 가까운 쉼(pause)으로 경계를 스냅
- 대본 정규화: '*'(표기 표식) 제거, '+'(말더듬/수정 표식) 처리, 한글 아닌 토큰 보고
- 지표: 단어 커버리지, 클립 사용률 (단위가 섞이지 않게 분리)
- --export DIR : 단위별 wav(16kHz) + 텍스트 manifest(jsonl) 생성

사용법
  python realign_v2.py <wav> <label.json> <model_dir> <out_prefix> [--unit sentence|word] [--export DIR]
출력
  <out_prefix>_units.csv   단위(문장/단어)별 시작·끝·CER·플래그
  <out_prefix>_clips.csv   클립별 ASR 결과와 배정 단위
  <out_prefix>_summary.json
"""
import sys, re, json, csv, argparse, time
from math import gcd
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from rapidfuzz.distance import Levenshtein
from jamo import h2j, j2hcj

# ---------------------------------------------------------------- 신호 분할
def frames_db(x, sr):
    hop, win = int(.01 * sr), int(.025 * sr)
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    return 10 * np.log10((fr ** 2).mean(1) + 1e-12)

def active_runs(db, off_db=10, merge_gap=0.3, min_len=0.15):
    act = db > np.percentile(db, 10) + off_db
    edges = np.flatnonzero(np.diff(np.r_[0, act.astype(int), 0])).reshape(-1, 2)
    segs = []
    for a, b in edges:
        if segs and a - segs[-1][1] < merge_gap * 100: segs[-1][1] = b
        else: segs.append([a, b])
    return [(a / 100, b / 100) for a, b in segs if b - a >= min_len * 100]

def macro_clips(fine, gap):
    """fine 구간을 gap 초 미만 간격이면 하나의 클립으로 묶음"""
    out = []
    for a, b in fine:
        if out and a - out[-1][1] < gap: out[-1][1] = b
        else: out.append([a, b])
    return out

def pick_clip_gap(fine):
    """삽입 무음 길이를 추정해 클립 경계 기준을 정함.
    간격 분포에서 가장 큰 '도약'을 찾아 그 사이 값을 기준으로 사용"""
    g = np.sort([fine[i + 1][0] - fine[i][1] for i in range(len(fine) - 1)])
    if len(g) < 3: return 3.0
    lg = np.log(g + 1e-3); k = int(np.argmax(np.diff(lg)))
    lo, hi = g[k], g[k + 1]
    if hi / max(lo, 1e-3) < 3 or hi < 2.0:   # 뚜렷한 삽입 무음 없음 -> 연속 낭독
        return 2.0
    return float(np.sqrt(lo * hi))

# ---------------------------------------------------------------- 텍스트
HANGUL = re.compile('[가-힣]')
_D = '영일이삼사오육칠팔구'
def sino(n):
    """정수 -> 한자어 수 읽기 (예: 365 -> 삼백육십오)"""
    if n == 0: return '영'
    out = ''
    for big, bn in ((10**8, '억'), (10**4, '만'), (1, '')):
        q, n = divmod(n, big)
        if not q: continue
        part = ''
        for v, u in ((1000, '천'), (100, '백'), (10, '십'), (1, '')):
            d, q = divmod(q, v)
            if d: part += ('' if (d == 1 and u) else _D[d]) + u
        out += part + bn
    return out
def num2kor(s):
    return re.sub(r'\d+', lambda m: sino(int(m.group())) if len(m.group()) < 10 else m.group(), s)
def J(s):
    s = num2kor(s)
    return j2hcj(h2j(''.join(c for c in s if '가' <= c <= '힣')))

def normalize_transcript(t):
    notes = []
    odd = sorted(set(w for w in t.split() if re.search(r'[^가-힣\s.,?!…\'"‘’“”·~\-]', w)))
    if odd: notes.append({'odd_tokens': odd})
    t = t.replace('*', '')
    t = re.sub(r'o/', '', t)        # 관찰된 오타 패턴
    t = t.replace('+', '')          # '나+ 나고' -> '나 나고' (반복 발화는 그대로 남김)
    t = re.sub(r'\s+', ' ', t).strip()
    return t, notes

def split_units(t, mode):
    if mode == 'auto':
        mode = 'sentence' if len(re.findall(r'[.?!…]', t)) >= 2 else 'word'
    if mode == 'word':
        return [w for w in t.split() if HANGUL.search(w)], mode
    parts = re.split(r'(?<=[.?!…])\s*', t)
    units = [p.strip() for p in parts if p.strip() and HANGUL.search(p)]
    return units, mode

# ---------------------------------------------------------------- ASR
def run_asr(x16, spans, md, threads=2):
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{md}/encoder-epoch-99-avg-1.int8.onnx", decoder=f"{md}/decoder-epoch-99-avg-1.int8.onnx",
        joiner=f"{md}/joiner-epoch-99-avg-1.int8.onnx", tokens=f"{md}/tokens.txt",
        num_threads=threads, decoding_method="modified_beam_search")
    out = []
    for a, b in spans:
        a0 = max(0, a - .2)
        st = rec.create_stream(); st.accept_waveform(16000, x16[int(a0 * 16000):int((b + .2) * 16000)])
        rec.decode_stream(st); r = st.result
        out.append(dict(start=a, end=b, text=r.text.strip(), tokens=list(r.tokens), ts=[a0 + t for t in r.timestamps]))
    return out

def asr_chunks(fine, clip, maxlen=18.0, join_gap=1.0):
    """클립 안을 18초 이하 청크로 나눔(쉼 위치에서 자름)"""
    inner = [s for s in fine if s[0] >= clip[0] - 1e-6 and s[1] <= clip[1] + 1e-6] or [clip]
    out = []
    for a, b in inner:
        if out and a - out[-1][1] < join_gap and b - out[-1][0] <= maxlen: out[-1][1] = b
        else: out.append([a, b])
    res = []
    for a, b in out:
        while b - a > maxlen: res.append([a, a + maxlen]); a += maxlen
        res.append([a, b])
    return res

def hyp_jamo_with_time(chunks):
    """ASR 결과를 자모 문자열과 자모별 (시작시간) 배열로 변환"""
    js, tt = [], []
    for c in chunks:
        toks, ts = c['tokens'], c['ts']
        for k, (tok, t) in enumerate(zip(toks, ts)):
            jj = J(tok)
            if not jj: continue
            t_next = ts[k + 1] if k + 1 < len(ts) else min(t + .4, c['end'] + .2)
            for m, ch in enumerate(jj):
                js.append(ch); tt.append(t + (t_next - t) * m / len(jj))
    return ''.join(js), np.array(tt)

# ---------------------------------------------------------------- 1단계: 클립-단위 DP
def ncost(a, b):
    if not a and not b: return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)

def assign_units(clip_hyp, clip_dur, unit_j, skip_clip_short=0.3, skip_clip=0.95, miss=1.0, kpen=0.03):
    S, N = len(clip_hyp), len(unit_j)
    ulen = np.array([len(u) for u in unit_j]); cum = np.r_[0, np.cumsum(ulen)]
    band = None
    if S * N > 200_000:
        band = max(60, int(0.1 * max(S, N)))
    INF = 1e18
    D = np.full((S + 1, N + 1), INF); D[0, 0] = 0; bp = {}
    rate = max(1e-6, cum[-1] / max(1, sum(len(h) for h in clip_hyp)))   # 대본 자모/인식 자모 비
    for i in range(S + 1):
        if band is not None:
            c = int(round(i * N / max(S, 1))); jlo, jhi = max(0, c - band), min(N, c + band)
        else: jlo, jhi = 0, N
        for j in range(jlo, jhi + 1):
            d0 = D[i, j]
            if d0 >= INF: continue
            if i < S:
                c = d0 + (skip_clip_short if (clip_dur[i] < .5 or not clip_hyp[i]) else skip_clip)
                if c < D[i + 1, j]: D[i + 1, j] = c; bp[i + 1, j] = (i, j, 'skip_clip')
            if j < N and d0 + miss < D[i, j + 1]:
                D[i, j + 1] = d0 + miss; bp[i, j + 1] = (i, j, 'missing')
            if i < S:
                h = clip_hyp[i]; maxlen = max(2 * len(h) * rate + 12, 12)
                k = 1
                while j + k <= N:
                    L = cum[j + k] - cum[j]
                    if k > 1 and L > maxlen: break
                    ref = ''.join(unit_j[j:j + k])
                    c = d0 + ncost(h, ref) + kpen * (k - 1)
                    if c < D[i + 1, j + k]: D[i + 1, j + k] = c; bp[i + 1, j + k] = (i, j, 'match')
                    k += 1
    i, j, path = S, N, []
    while (i, j) != (0, 0):
        pi, pj, t = bp[i, j]; path.append((t, pi, pj, i, j)); i, j = pi, pj
    return list(reversed(path))

# ---------------------------------------------------------------- 2단계: 클립 내부 경계
def split_inside(clip, units_j, hyp_j, hyp_t, pauses):
    """units_j: 이 클립에 배정된 단위들의 자모 문자열 리스트. 반환: [(start,end,unit_cer,mapped_frac)]"""
    a, b = clip
    if len(units_j) == 1:
        return [(a, b, ncost(hyp_j, units_j[0]), 1.0)]
    ref = ''.join(units_j); owner = np.repeat(np.arange(len(units_j)), [len(u) for u in units_j])
    ref2hyp = np.full(len(ref), -1)
    for op in Levenshtein.opcodes(ref, hyp_j):
        if op.tag in ('equal', 'replace'):
            n = min(op.src_end - op.src_start, op.dest_end - op.dest_start)
            ref2hyp[op.src_start:op.src_start + n] = np.arange(op.dest_start, op.dest_start + n)
    spans = []
    for u in range(len(units_j)):
        idx = ref2hyp[owner == u]; idx = idx[idx >= 0]
        spans.append((hyp_t[idx.min()], hyp_t[idx.max()] + .25, len(idx) / max(1, (owner == u).sum()), idx) if len(idx) else None)
    # 경계 결정: 이웃 단위 사이 중점 -> 가장 가까운 쉼 중앙으로 스냅
    nU = len(units_j); starts = [None] * nU; ends = [None] * nU
    starts[0], ends[-1] = a, b
    known = [k for k in range(nU) if spans[k] is not None]
    # 매핑 실패 단위는 자모 길이 비례로 보간
    clen = np.r_[0, np.cumsum([len(u) for u in units_j])]
    def interp_time(pos):  # 대본 자모 위치 -> 시간 (매핑된 이웃으로 선형 보간)
        pts = [(clen[k], spans[k][0]) for k in known] + [(clen[k + 1], spans[k][1]) for k in known]
        pts = sorted(set([(0, a), (clen[-1], b)] + pts))
        xs, ys = zip(*pts); return float(np.interp(pos, xs, np.maximum.accumulate(ys)))
    for k in range(nU - 1):
        e = spans[k][1] if spans[k] else interp_time(clen[k + 1])
        s = spans[k + 1][0] if spans[k + 1] else interp_time(clen[k + 1])
        mid = (min(e, s) + max(e, s)) / 2
        cand = [(abs((p0 + p1) / 2 - mid), p0, p1) for p0, p1 in pauses if a < p0 and p1 < b and abs((p0 + p1) / 2 - mid) < 1.0]
        if cand:
            _, p0, p1 = min(cand); ends[k], starts[k + 1] = p0 + .05, p1 - .05
        else:
            ends[k] = starts[k + 1] = mid
    # 단조성 보정
    for k in range(1, nU):
        starts[k] = max(starts[k], ends[k - 1] - 1e-3 if ends[k - 1] else starts[k])
    out = []
    for k in range(nU):
        s, e = starts[k], max(ends[k], starts[k] + .05)
        m = (hyp_t >= s - .05) & (hyp_t < e + .05)
        out.append((s, e, ncost(''.join(np.array(list(hyp_j))[m]) if m.any() else '', units_j[k]),
                    spans[k][2] if spans[k] else 0.0))
    return out

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav'); ap.add_argument('label'); ap.add_argument('model_dir'); ap.add_argument('out_prefix')
    ap.add_argument('--unit', default='auto', choices=['auto', 'sentence', 'word'])
    ap.add_argument('--export', default=None); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--low_conf', type=float, default=0.7)
    A = ap.parse_args(); T0 = time.time()

    lab = json.load(open(A.label, encoding='utf-8'))
    x, sr = sf.read(A.wav); x = x.mean(1) if x.ndim > 1 else x
    dur = len(x) / sr
    meta_sr = lab.get('Meta_info', {}).get('SamplingRate')
    checks = {'wav_sr': sr, 'label_sr': meta_sr, 'wav_dur': round(dur, 2), 'label_playTime': lab.get('playTime')}
    if meta_sr and int(meta_sr) != sr: checks['sr_mismatch'] = True
    if lab.get('playTime') and abs(float(lab['playTime']) - dur) > 1: checks['duration_mismatch'] = True

    text, notes = normalize_transcript(lab['Transcript'])
    units, mode = split_units(text, A.unit)
    unit_j = [J(u) for u in units]

    db = frames_db(x, sr)
    fine = active_runs(db, merge_gap=0.15, min_len=0.1)
    gap = pick_clip_gap(fine)
    clips = macro_clips(fine, gap)
    pauses = [(fine[i][1], fine[i + 1][0]) for i in range(len(fine) - 1) if fine[i + 1][0] - fine[i][1] >= .15]

    g = gcd(sr, 16000); x16 = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    asr_spans, owner = [], []
    for ci, c in enumerate(clips):
        for s in asr_chunks(fine, c): asr_spans.append(s); owner.append(ci)
    asr = run_asr(x16, asr_spans, A.model_dir, A.threads)
    clip_asr = [[r for r, o in zip(asr, owner) if o == ci] for ci in range(len(clips))]
    clip_hyp = [hyp_jamo_with_time(ch) for ch in clip_asr]
    clip_text = [' '.join(r['text'] for r in ch) for ch in clip_asr]
    clip_dur = [b - a for a, b in clips]

    path = assign_units([h for h, _ in clip_hyp], clip_dur, unit_j)

    unit_rows = [None] * len(units); clip_rows = []
    for t, pi, pj, i, j in path:
        if t == 'missing':
            unit_rows[pj] = dict(unit=pj, text=units[pj], clip='', start='', end='', jamo_cer='', flag='NOT_FOUND')
            continue
        a, b = clips[pi]
        if t == 'skip_clip':
            clip_rows.append(dict(clip=pi, start=round(a, 2), end=round(b, 2), asr=clip_text[pi], units='',
                                  clip_cer='', flag='NON_SPEECH' if (b - a < .5 or not clip_hyp[pi][0]) else 'UNMATCHED_SPEECH'))
            continue
        hj, ht = clip_hyp[pi]
        res = split_inside((a, b), unit_j[pj:j], hj, ht, pauses)
        clip_cer = ncost(hj, ''.join(unit_j[pj:j]))
        clip_rows.append(dict(clip=pi, start=round(a, 2), end=round(b, 2), asr=clip_text[pi], units=f'{pj}-{j - 1}',
                              clip_cer=round(clip_cer, 2), flag='OK' if clip_cer <= A.low_conf else 'LOW_CONF'))
        for k, (s, e, cer, mf) in enumerate(res):
            u = pj + k
            flag = 'OK' if cer <= A.low_conf else 'LOW_CONF'
            if j - pj > 1 and mf < .2: flag = 'BOUNDARY_GUESS'   # 경계가 보간으로만 정해짐
            unit_rows[u] = dict(unit=u, text=units[u], clip=pi, start=round(s, 2), end=round(e, 2),
                                jamo_cer=round(cer, 2), flag=flag)

    pre = A.out_prefix
    with open(pre + '_units.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(unit_rows[0].keys())); w.writeheader(); w.writerows(unit_rows)
    with open(pre + '_clips.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(clip_rows[0].keys())); w.writeheader(); w.writerows(clip_rows)

    from collections import Counter
    uf = Counter(r['flag'] for r in unit_rows); cf = Counter(r['flag'] for r in clip_rows)
    speech_clips = sum(1 for r in clip_rows if r['flag'] != 'NON_SPEECH')
    summary = dict(file=lab.get('File_id'), unit_mode=mode, n_units=len(units), n_clips=len(clips),
                   clip_gap_threshold_s=round(gap, 2), unit_flags=dict(uf), clip_flags=dict(cf),
                   unit_coverage_ok=round(uf.get('OK', 0) / len(units), 3),
                   unit_coverage_assigned=round(1 - uf.get('NOT_FOUND', 0) / len(units), 3),
                   clip_usage=round(sum(1 for r in clip_rows if r['units']) / max(1, speech_clips), 3),
                   checks=checks, transcript_notes=notes, runtime_s=round(time.time() - T0, 1))
    json.dump(summary, open(pre + '_summary.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))

    if A.export:
        import os; os.makedirs(A.export, exist_ok=True)
        stem = (lab.get('File_id') or 'file').rsplit('.', 1)[0]
        with open(os.path.join(A.export, stem + '.jsonl'), 'w', encoding='utf-8') as mf:
            for r in unit_rows:
                if r['flag'] not in ('OK',): continue          # 기본: OK만 내보냄
                s, e = float(r['start']), float(r['end'])
                fn = f"{stem}_{r['unit']:04d}.wav"
                sf.write(os.path.join(A.export, fn), x16[int(max(0, s - .15) * 16000):int((e + .15) * 16000)], 16000)
                mf.write(json.dumps(dict(audio=fn, text=r['text'], start=s, end=e, cer=r['jamo_cer']), ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
