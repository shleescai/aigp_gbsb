# 구음장애 음성 데이터 전처리 / 재정렬 스크립트

AI-Hub "구음장애 음성인식 데이터"(한국어, 뇌신경장애 하위 데이터)를 전처리하고, 길게 녹음된 낭독 음성을
대본과 다시 정렬하는 스크립트 모음입니다.
[논문 원본 저장소](https://github.com/yoona-J/Diagnosis-Aware_Multitask_Fine-Tuning_of_Whisper_for_DSR)를
보완하기 위해 작성했습니다.

**음성, 라벨, 데이터에서 파생된 파일은 포함하지 않았습니다.** 데이터셋은 AI-Hub에서 직접 내려받고,
각 스크립트 상단의 경로 상수를 환경에 맞게 수정해서 사용하세요(기본값은 `~/DB/aihub_dysarthria/...`).

| 스크립트 | 설명 |
|---|---|
| `src/preprocess_brain_nerve.py` | RMS 에너지 기반 VAD, 전사문 분할, 개수 검증, 16 kHz 출력 (논문 노트북을 간소화한 변형) |
| `src/run_preprocess_notebook.py` | 논문의 `Preprocessing_Peripheral_Neuropathy.ipynb`를 수정 없이 실행 (`BASE_PATH`만 교체) |
| `src/relabel_compare.py` | AI-Hub 라벨링 가이드의 태그(`/b /n /L + *`)를 반영해 전사문을 다시 분할하고 세그먼트 개수를 비교 |
| `src/clean_text_by_guide.py` | 전사문 정제(태그, 기호, 깨진 `<U+xxxx>` 제거) 후 개수가 일치한 화자의 세그먼트별 텍스트 저장 |
| `src/realign.py` | 무음으로 나뉜 클립과 sherpa-onnx ASR, 자모 편집거리 DP로 단어 목록 낭독을 대본에 정렬 (CPU) |
| `src/realign_batch.py` | `realign.py` 일괄 실행 |
| `src/realign_v2.py` | 문장/단어 단위 자동 선택, 적응형 클립 간격, 밴드 DP, 클립 내부 경계 분할, `--export` 옵션 (CPU) |
| `src/realign_v2_batch.py` | `realign_v2.py` 일괄 실행 |

## 요구 사항
Python 3.10 이상, `numpy`, `scipy`, `soundfile`, `pandas`, `jamo`, `rapidfuzz`, `sherpa-onnx`.
`run_preprocess_notebook.py`에는 `nbformat`, `nbclient`, `imageio-ffmpeg`, `pydub`도 필요합니다.
ASR 모델은 k2-fsa/sherpa-onnx 릴리스의 한국어 zipformer(`sherpa-onnx-zipformer-korean-2024-06-24`)를 사용합니다.

## 참고
- 재정렬 스크립트는 낭독 녹음을 가정합니다. 과제 유형에 따라 품질이 다르므로 단위별 CER과 플래그를 꼭 확인하세요.
- `realign_v2.py`의 `OK`는 자모 CER 0.7 이하라는 뜻으로 기준이 느슨합니다. 구간을 내보내기 전에 더 엄격한 기준을 사용하세요.
