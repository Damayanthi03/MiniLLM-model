# MiniLM — A Lightweight Transformer Language Model Trained From Scratch

MiniLM is a small **decoder-only Transformer** (GPT-style) written in PyTorch and trained from random weights
on a small text file. A dark-navy Streamlit website lets you train it, watch real loss curves, generate text,
evaluate it on held-out text and explore its attention.

* No pretrained weights, no Hugging Face downloads, no OpenAI / Claude / Gemini / external AI API calls.
* No hard-coded answers and no simulated progress: every number on screen comes from the running code.
* Runs on CPU; uses CUDA automatically only if it exists.

> **Honest expectation:** this is a ~0.4 M-parameter character-level model trained on ~8 K characters.
> It learns spelling, word shapes and sentence rhythm of its training text. It does **not** understand meaning
> and is nowhere near ChatGPT-level quality.

---

## 1. Project layout

```
minilm/
  app.py                 Streamlit entry point (navigation, theme)
  config.py              Paths, ModelConfig, TrainConfig, presets, device detection
  requirements.txt
  .replit                Replit Run-button configuration
  .streamlit/config.toml Theme + server settings (Replit friendly)
  model/
    tokenizer.py         Character-level tokenizer (encode / decode)
    transformer.py       Causal multi-head attention, feed-forward, blocks, MiniLM
    generate.py          Autoregressive generation, temperature + top-k sampling
    checkpoint.py        Save / load checkpoints, clear errors for missing/corrupt files
  training/
    dataset.py           Text loading, train/validation split, mini-batches
    train.py             The real training loop (backprop + AdamW) and train_step()
    evaluate.py          Validation loss, perplexity, sample generations
    manager.py           Background-thread runner; refuses duplicate training jobs
  ui/theme.py            CSS theme + HTML components
  views/                 One module per page (dashboard, generation, training, info, evaluation)
  data/sample_text.txt   Original sample text written for this project
  checkpoints/           Saved model (minilm.pt) appears here after training
  outputs/               training_log.json and evaluation_report.json
  tests/                 pytest suite
```

## 2. Run it

### On Replit
1. Create a Python Repl and upload the **contents** of the `minilm/` folder to the Repl root
   (so `app.py` sits next to `.replit`).
2. Press **Run**. The first run installs `requirements.txt` (PyTorch takes a few minutes), then starts Streamlit
   on port 8501 and opens the web view.

### Locally (VS Code / terminal)
```bash
cd minilm
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501.

### Run the tests
```bash
pytest
```

## 3. How to train and generate
1. Open **Model Training**, pick a preset (*Quick demo* / *Standard* / *Extended*) and press **Start training**.
   Live step count, batch loss, train/validation loss, speed and a loss chart update every second.
   A second job cannot be started while one runs; **Stop training** ends it early (marked as partial).
2. When it finishes, a checkpoint is written to `checkpoints/minilm.pt` and the status becomes **Trained**.
3. Open **Text Generation**, type a prompt (or click an example), adjust length / temperature / top-k and press **Generate**.
4. Open **Evaluation** and press **Run evaluation** for validation loss, perplexity and sample generations.

You can also train your own text: choose *My uploaded text* on the training page and upload a UTF-8 `.txt` file
(at least ~1,500 characters for the default context length).

## 4. Suggested demo for your professor (≈ 8 minutes)
1. **Dashboard** – state the claim: Transformer written from scratch, random init, no pretrained models, no APIs.
   Point at the architecture chips and the parameter count.
2. Show the code briefly: `model/transformer.py` (attention + causal mask), `training/train.py` (`train_step`:
   forward → loss → `backward()` → AdamW step).
3. **Model Training** – start the *Quick demo* (or *Standard*) run. Explain that step 0 loss is about ln(vocabulary)
   (random guessing) and that the curve falls as the weights are updated.
4. **Text Generation** – generate from an *untrained* state first if you like (nothing is available, the app says so),
   then from the trained model. Vary temperature and top-k and explain the effect.
5. **Model Information** – walk through the plain-language explanation, the parameter table and the
   **attention explorer** (the empty upper triangle is the causal mask).
6. **Evaluation** – run it, discuss validation loss vs. training loss (overfitting on tiny data) and perplexity vs. the
   random baseline.
7. Close with the **limitations** honestly: tiny model, tiny data, character-level, no understanding.
8. Run `pytest` live to show the tests, including "a training step updates the weights".

## 5. Download from Replit and run in VS Code
1. In Replit, open the three-dot menu in the Files panel → **Download as zip** (or Tools → Download).
2. Unzip, open the folder in VS Code, create a virtual environment and `pip install -r requirements.txt` (see above).
3. `streamlit run app.py`. If you copied `checkpoints/minilm.pt` too, the trained model loads immediately;
   otherwise train again locally.

## 6. How the pieces work
* **Tokenizer** – each distinct character is one id.
* **Model** – token + position embeddings → N × [LayerNorm → causal multi-head attention → residual → LayerNorm →
  feed-forward → residual] → LayerNorm → linear layer to vocabulary scores (weights tied to the token embeddings).
* **Learning** – a window of text is the input; the same window shifted by one character is the target.
  Cross-entropy loss, backpropagation and AdamW (with warm-up + cosine learning-rate decay and gradient clipping).
* **Validation** – the last 10 % of the text is never trained on. Perplexity = exp(validation loss), a per-character value.
* **Generation** – predict one character, sample it (temperature / top-k), append, repeat.

## 7. Limitations and Replit notes
* Training runs inside the Streamlit server process (a background thread). If Replit restarts or sleeps the Repl,
  the running job is lost; start it again. Shorter presets exist for this reason.
* Training speed depends on the CPU Replit gives you; the app shows the measured steps/second and a time estimate,
  so use *Quick demo* if it feels slow.
* PyTorch is large. `requirements.txt` points at the CPU-only wheel index to keep the install small.
* With only ~8 K characters the model overfits if trained long: training loss keeps falling while validation loss
  flattens or rises. The Evaluation page flags this.
* Troubleshooting: if `import torch` fails on Replit with a `libstdc++` error, add `pkgs.stdenv.cc.cc.lib` to
  `replit.nix` and expose it through `LD_LIBRARY_PATH`. (Standard Replit workaround; not needed on a normal machine.)

## 8. Tests (`pytest`)
`tests/` covers: tokenizer encode/decode; forward-pass shapes and loss; causal masking (zero attention to the future,
and changing future tokens leaves earlier outputs unchanged); a training step changes the weights; a full tiny run
lowers loss and saves a checkpoint; stop-early checkpoint; checkpoint round-trip; generation from a loaded
checkpoint (seeded, greedy, top-k, empty prompt, longer than the context); missing and corrupted checkpoints;
duplicate-job prevention and error reporting in the training manager; and every web page rendering with no checkpoint,
a trained checkpoint and a corrupted checkpoint (Streamlit `AppTest`).
