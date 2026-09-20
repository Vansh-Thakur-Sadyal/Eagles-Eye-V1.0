# Installation

## 1. Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
cp .env.example .env
```

That is enough to run the platform. It will start, serve the dashboard, drive
cameras, run every agent, raise incidents and answer forensic queries — using
the built-in fallbacks described below. Add the model stack when you want real
detection accuracy.

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 --reload
```

On first run the generated administrator password is printed to the log and
written to `storage/FIRST_RUN_CREDENTIALS.txt`. **Change it, then delete that
file.**

> If port 8000 is refused on Windows with "an attempt was made to access a
> socket in a way forbidden by its access permissions", the port sits inside a
> reserved Hyper-V range. Check with
> `netsh interface ipv4 show excludedportrange protocol=tcp` and pick another
> port.

## 2. Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies the API
npm run build    # writes dist/, which the backend then serves at /
```

Build the frontend and the backend serves the whole app on one port — that is
the simplest deployment.

## 3. GPU (RTX 50-series)

The RTX 5070 Ti is Blackwell (compute capability 12.0) and **needs a CUDA 12.8
build of torch**. The default PyPI wheel will install and then fail at runtime
with `no kernel image is available for execution on the device`.

```bash
cd backend
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see a `+cu128` version and `True`. If `torch.cuda.is_available()` is
`False`, the dashboard says so plainly rather than pretending — the GPU toggle
returns a warning and inference stays on CPU.

## 4. Detection models

```bash
.venv/Scripts/python.exe -m pip install -r requirements-gpu.txt
```

> **Do not `pip install ultralytics` on its own after installing CUDA torch.**
> Ultralytics depends on `torch` without pinning an index, so pip resolves it
> from plain PyPI, installs the CPU-only wheel and silently replaces your
> working CUDA build. `torch.cuda.is_available()` flips to `False` and
> inference quietly drops to CPU with no error.
>
> `requirements-gpu.txt` pins torch to the cu128 index and orders the installs
> correctly. If it happens anyway:
>
> ```bash
> .venv/Scripts/python.exe -m pip install --force-reinstall torch==2.11.0 torchvision >     --index-url https://download.pytorch.org/whl/cu128
> ```
>
> Always confirm afterwards with `python tools/verify_stack.py`.

Ultralytics downloads `yolo11m.pt` on first use. To pin a specific file, put it
in `storage/models/` and set `SENTINEL_YOLO_WEIGHTS` to its filename.

### Optional: open-vocabulary grounding

```bash
.venv/Scripts/python.exe -m pip install transformers accelerate
```

Then set `SENTINEL_LOCATEANYTHING_ENABLED=true`.

`nvidia/LocateAnything-3B` accepts free-text prompts ("an unattended suitcase",
"a person carrying a black backpack"), which covers the long tail YOLO's fixed
class list misses. It is roughly a second per call, so Sentinel runs it on
keyframes and on operator forensic queries — never in the per-frame loop. Set
`SENTINEL_DETECTOR=ensemble` to use YOLO every frame with periodic grounding
refinement.

At 3B parameters in fp16 it needs about 7 GB of VRAM, which fits alongside YOLO
on a 16 GB card.

## 5. Optional components

| Component | Install | Without it |
|---|---|---|
| Person Re-ID | `pip install torchreid gdown tensorboard` | Falls back to a striped HSV-histogram descriptor. Works for short-horizon association; weaker across long gaps. **See the note below about weights.** |
| Face matching | see below | Falls back to OpenCV Haar detection. Detects fewer faces; watchlist matching relies more on body appearance, which is capped lower on purpose. |
| Vector search | `pip install chromadb sentence-transformers` | Falls back to a built-in TF-IDF index. Genuinely good for short incident text; no semantic paraphrase matching. |
| Behaviour training | `pip install scikit-learn` | `training/train_behavior.py` will not run. |

### Re-ID weights

`pip install torchreid` gives you the OSNet architecture, but
`pretrained=True` loads **ImageNet** weights — features trained to classify
objects, never trained to tell two people apart. They beat the handcrafted
fallback, but cross-camera association will be weaker than the green tick
suggests.

For real re-identification, download a Market-1501 (or DukeMTMC) OSNet
checkpoint from the torchreid model zoo and drop it in:

```
storage/models/reid/osnet_x1_0_market1501.pth
```

Sentinel loads the first checkpoint it finds there and reports the provenance
as `weights: reid:<name>` instead of `weights: imagenet`, both in
`verify_stack.py` and in **Settings → Compute & models**. Until then it logs a
warning at startup rather than letting you assume otherwise.

### Face matching, and the onnxruntime trap

```bash
.venv/Scripts/python.exe -m pip install insightface
.venv/Scripts/python.exe -m pip uninstall -y onnxruntime
.venv/Scripts/python.exe -m pip install --force-reinstall --no-deps onnxruntime-gpu
```

insightface depends on plain `onnxruntime`. When both that and `onnxruntime-gpu`
are present, the CPU build shadows the GPU one: `CUDAExecutionProvider`
disappears and face matching silently runs on CPU. Removing the CPU build after
installing insightface is what fixes it.

Confirm:

```bash
.venv/Scripts/python.exe -c "import torch, onnxruntime; print(onnxruntime.get_available_providers())"
```

`CUDAExecutionProvider` must be in the list. Importing torch first matters —
onnxruntime reuses the CUDA and cuDNN DLLs torch has already loaded, so one
CUDA install serves both.

Sentinel reports the provider it actually got in **Settings → Compute & models**
(`on_gpu: true/false`), so a silent CPU fallback is visible rather than assumed.

Every fallback reports itself honestly in **Settings → Compute & models**, so
the UI never implies a trained model is running when one is not.

## 6. Language model (optional)

Incident narration, reports and the assistant all work with no LLM — they use
deterministic templates over the same retrieved evidence, which means they
cannot hallucinate. Configure a provider for more natural phrasing:

```bash
# Local, via Ollama - pairs well with your own GPU
SENTINEL_LLM_PROVIDER=ollama
SENTINEL_LLM_MODEL=llama3.1:8b

# Any OpenAI-compatible endpoint
SENTINEL_LLM_PROVIDER=openai_compatible
SENTINEL_LLM_BASE_URL=https://api.example.com/v1
SENTINEL_LLM_API_KEY=...
SENTINEL_LLM_MODEL=...
```

Check it from the UI (Settings → Compute & models → Re-check) or
`GET /api/system/llm/health`.

## 7. Docker

```bash
cd docker
cp .env.example .env      # set SENTINEL_JWT_SECRET and POSTGRES_PASSWORD
docker compose up -d --build
```

Dashboard on `:8080`, API on `:8000`, PostgreSQL behind them. Add
`--profile automation` to bring up n8n.

For GPU inside the container, build with the CUDA base image and uncomment the
`deploy.resources.devices` block:

```bash
docker build --build-arg BASE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 \
  -f docker/Dockerfile.backend -t sentinel-backend:gpu .
```

## 8. Verify the install

```bash
# What is actually installed, and the fix for anything that is not
backend/.venv/Scripts/python.exe tools/verify_stack.py
backend/.venv/Scripts/python.exe tools/verify_stack.py --bench   # GPU vs CPU timing

cd backend
.venv/Scripts/python.exe -m pytest tests/ -q        # 137 tests
cd ..
backend/.venv/Scripts/python.exe tools/make_datasets.py all --per-category 24
backend/.venv/Scripts/python.exe tools/evaluate_agents.py
```

The last command scores every behaviour agent against generated ground truth.
See `docs/TRAINING.md` for what those numbers do and do not mean.
