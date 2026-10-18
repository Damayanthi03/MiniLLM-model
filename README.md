# MiniLLM — A Lightweight Transformer Language Model Trained From Scratch

A small decoder-only Transformer language model built using PyTorch. MiniLLM learns to predict the next character from a training text dataset and generates text using its learned parameters.

## About the Project

MiniLLM is an educational language-model project developed to understand how Transformer models work internally. It is trained from randomly initialized weights using a custom training loop, without pretrained model weights or external AI APIs.

The project includes a Streamlit dashboard for training the model, generating text, exploring model architecture, and evaluating performance.

## Key Features

- Transformer language model implemented in PyTorch
- Training from random initialization
- Character-level tokenization
- Next-character prediction and text generation
- Causal self-attention and multiple attention heads
- Live training progress and loss-curve visualization
- Model information and parameter statistics
- Text-generation controls such as temperature and top-k sampling
- Evaluation using validation loss and perplexity
- CPU support, with CUDA used when available and configured

## Model Architecture

| Component | Configuration |
|---|---|
| Model type | Decoder-only Transformer |
| Trainable parameters | 419,072 |
| Vocabulary size | 46 characters |
| Transformer blocks | 2 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Context length | 128 characters |
| Feed-forward width | 512 |
| Dropout | 0.1 |
| Training steps | 3,000 |

*These values represent the displayed configuration of the current trained model. They may change if the model is reconfigured or retrained.*

## Technologies Used

- **Python** — Core programming language
- **PyTorch** — Transformer implementation and training
- **Streamlit** — Interactive web interface
- **NumPy** — Numerical operations
- **Matplotlib** — Training and loss visualizations

## How It Works

1. **Text preparation:** The training text is converted into character IDs.
2. **Input and target creation:** Text sequences are shifted by one character to create next-character prediction targets.
3. **Embeddings:** Character IDs are converted into vectors, with positional information added.
4. **Transformer processing:** Causal self-attention and feed-forward layers process the sequence.
5. **Loss calculation:** Cross-entropy loss measures the difference between predictions and actual next characters.
6. **Training:** Backpropagation and the AdamW optimizer update the model weights.
7. **Text generation:** The trained model predicts one character at a time until the requested output length is reached.

## Project Structure

```text
minilm/
├── README.md
├── app.py
├── model.py
├── train.py
├── requirements.txt
├── data/
├── checkpoints/
└── .gitignore
```

*Adjust this structure to match the actual files in your repository.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Damayanthi03/MiniLLM-model.git
cd MiniLLM-model
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

If the Streamlit interface is in `app.py`:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

**Note:** Ensure the filenames and commands above match your actual project files.

## Model Training and Evaluation

MiniLLM trains on a small text dataset and reports training loss, validation loss, and perplexity. A saved checkpoint can be loaded for text generation and evaluation.

Since this is a small character-level model, generated text may contain spelling mistakes, repeated phrases, or incomplete sentences. It is an educational prototype, not a general-purpose LLM.

## Limitations

- Uses a small training dataset and vocabulary.
- Predicts characters rather than complete word-level tokens.
- Output quality depends on the training data and model configuration.
- Requires additional data, model capacity, and evaluation to improve language-generation quality.
- Does not have the general knowledge or reasoning abilities of large pretrained language models.

## Future Enhancements

- Train on a larger and more diverse text dataset.
- Experiment with deeper Transformer architectures.
- Add word-level or subword tokenization.
- Improve text-generation quality and evaluation.
- Add support for saving and loading different model configurations.

## Project Objective

The main objective of MiniLLM is to implement and train a small Transformer language model from scratch to understand tokenization, self-attention, training, next-character prediction, and text generation without relying on pretrained model weights or external AI APIs.
