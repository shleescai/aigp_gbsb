"""Run the original Preprocess/[Distribution]Preprocessing_Peripheral_Neuropathy.ipynb unmodified
(except BASE_PATH and `!pip` cells) on the AI-Hub 26.청각+뇌신경장애 data.
The source notebook is not edited; a patched copy is executed inside RUN."""
import json, os, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import imageio_ffmpeg

SRC = Path(__file__).resolve().parent.parent / "Preprocess/[Distribution]Preprocessing_Peripheral_Neuropathy.ipynb"
T = Path.home() / "DB/aihub_dysarthria/013.구음장애_음성인식_데이터/01.데이터/1.Training"
RUN = Path.home() / "DB/aihub_dysarthria/notebook_run"

def link_all(src_dirs, dst, pattern):
    dst.mkdir(parents=True, exist_ok=True)
    for d in src_dirs:
        for f in sorted(d.iterdir()):
            if f.suffix.lower() == pattern and not (dst / f.name).exists():
                (dst / f.name).symlink_to(f)

link_all([T / "원천데이터/26.청각+뇌신경장애", T / "원천데이터/26.청각+뇌신경장애(2)"],
         RUN / "Training/audio_data/TS01/Peripheral_Neuropathy", ".wav")
link_all([p for p in (T / "라벨링데이터").glob("26.*")],
         RUN / "Training/label_data/TL01/Peripheral_Neuropathy", ".json")
for p in ["Validation/audio_data/VS01/Cerebral_Palsy_disease", "Validation/label_data/VL01/Cerebral_Palsy_disease",
          "Validation/audio_data/VS01/Stroke", "Validation/label_data/VL01/Stroke"]:
    (RUN / p).mkdir(parents=True, exist_ok=True)

bin_dir = RUN / "bin"; bin_dir.mkdir(exist_ok=True)
if not (bin_dir / "ffmpeg").exists():
    (bin_dir / "ffmpeg").symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"

nb = nbformat.read(SRC, as_version=4)
nb.cells = [c for c in nb.cells if not ''.join(c.source).lstrip().startswith("!pip")]
for c in nb.cells:
    c.source = c.source.replace('Path("YOUR_ROOT")', f'Path(r"{RUN}")')
os.chdir(RUN)
client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(RUN)}},
                        allow_errors=True)
try:
    client.execute()
finally:
    nbformat.write(nb, RUN / "executed_Preprocessing_Peripheral_Neuropathy.ipynb")
