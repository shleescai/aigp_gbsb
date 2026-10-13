"""
AIHub 구음장애 단어/문장 낭독 파일 재정렬 도구
사용법: python realign.py <wav> <label.json> <model_dir> <out.csv>
1) 무음(삽입된 ~10초 무음) 기준으로 원래 클립 경계 복원
2) 각 클립을 한국어 ASR(sherpa-onnx zipformer)로 인식
3) 인식 결과와 대본 단어열을 자모 편집거리 DP로 정렬
   - 한 클립에 여러 단어(1~3개) 허용, 비발화 클립 skip, 미발화 단어 skip 허용
4) 클립별 배정 단어 + 신뢰도 + 플래그를 CSV로 출력
"""
import sys, json, csv, numpy as np, soundfile as sf, sherpa_onnx
from scipy.signal import resample_poly
from jamo import h2j, j2hcj

def segments(x, sr, off_db=10, merge_gap=0.3, min_len=0.15):
    hop, win = int(.01*sr), int(.025*sr)
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    db = 10*np.log10((fr**2).mean(1)+1e-12)
    act = db > np.percentile(db, 10) + off_db
    edges = np.flatnonzero(np.diff(np.r_[0, act.astype(int), 0]))
    segs = []
    for a, b in edges.reshape(-1, 2):
        if segs and a - segs[-1][1] < merge_gap*100: segs[-1][1] = b
        else: segs.append([a, b])
    return [(a/100, b/100) for a, b in segs if b - a >= min_len*100]

def J(s): return j2hcj(h2j(''.join(c for c in s if '가' <= c <= '힣')))
def ed(a, b):
    d = np.arange(len(b)+1)
    for i, ca in enumerate(a, 1):
        p = d.copy(); d[0] = i
        for k, cb in enumerate(b, 1): d[k] = min(p[k]+1, d[k-1]+1, p[k-1]+(ca != cb))
    return int(d[-1])
def cost(h, ws):
    a, b = J(h), J(''.join(ws)); return ed(a, b)/max(len(a), len(b), 1)

def main(wav, js, md, out):
    x, sr = sf.read(wav)
    if x.ndim > 1: x = x.mean(1)
    lab = json.load(open(js, encoding='utf-8'))
    raw = lab['Transcript'].split()
    words = [''.join(c for c in w if '가' <= c <= '힣') for w in raw]
    segs = segments(x, sr)
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{md}/encoder-epoch-99-avg-1.int8.onnx", decoder=f"{md}/decoder-epoch-99-avg-1.int8.onnx",
        joiner=f"{md}/joiner-epoch-99-avg-1.int8.onnx", tokens=f"{md}/tokens.txt",
        num_threads=2, decoding_method="modified_beam_search")
    from math import gcd
    g = gcd(sr, 16000); x16 = resample_poly(x, 16000//g, sr//g).astype(np.float32)
    hyps = []
    for a, b in segs:
        st = rec.create_stream(); st.accept_waveform(16000, x16[int(max(0, a-.4)*16000):int((b+.4)*16000)])
        rec.decode_stream(st); hyps.append(st.result.text.strip())
    S, N, INF = len(segs), len(words), 1e9
    D = np.full((S+1, N+1), INF); D[0, 0] = 0; bp = {}
    for i in range(S+1):
        for j in range(N+1):
            if D[i, j] >= INF: continue
            if i < S:  # 비발화 클립: 짧거나 인식 결과가 비면 싸게
                c = D[i, j] + (0.3 if (segs[i][1]-segs[i][0] < .5 or not J(hyps[i])) else 0.95)
                if c < D[i+1, j]: D[i+1, j] = c; bp[i+1, j] = (i, j, 'noise')
            if j < N and D[i, j]+1.0 < D[i, j+1]: D[i, j+1] = D[i, j]+1.0; bp[i, j+1] = (i, j, 'missing')
            for k in (1, 2, 3):
                if i < S and j+k <= N:
                    c = D[i, j] + cost(hyps[i], words[j:j+k]) + .15*(k-1)
                    if c < D[i+1, j+k]: D[i+1, j+k] = c; bp[i+1, j+k] = (i, j, 'match')
    i, j, path = S, N, []
    while (i, j) != (0, 0):
        pi, pj, t = bp[i, j]; path.append((t, pi, pj, j)); i, j = pi, pj
    rows = []
    for t, pi, pj, j in reversed(path):
        ws = words[pj:j] if t != 'noise' else []
        c = cost(hyps[pi], ws) if t == 'match' else None
        flag = {'noise': 'NON_SPEECH', 'missing': 'WORD_NOT_FOUND'}.get(t)
        if t == 'match':
            flag = 'LOW_CONF' if c > .7 else ('MULTI_WORD' if len(ws) > 1 else 'OK')
        a, b = (segs[pi] if t != 'missing' else (None, None))
        rows.append(dict(clip=pi if t != 'missing' else '', start=a, end=b,
                         asr=hyps[pi] if t != 'missing' else '', words=' '.join(ws),
                         word_idx=f"{pj}-{j-1}" if ws else '', jamo_cer=None if c is None else round(c, 2), flag=flag))
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    gaps = np.diff([s[0] for s in segs]) if S > 1 else []
    sil = [segs[k+1][0]-segs[k][1] for k in range(S-1)]
    odd = [w for w in raw if not all('가' <= c <= '힣' for c in w)]
    print(f"clips={S} words={N} | silence gap median={np.median(sil):.2f}s sd={np.std(sil):.2f}")
    print("odd tokens in transcript:", odd)
    from collections import Counter; print(Counter(r['flag'] for r in rows))

if __name__ == '__main__': main(*sys.argv[1:5])
