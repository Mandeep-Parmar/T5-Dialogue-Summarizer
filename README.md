# T5 Dialogue Summarizer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end dialogue and conversation summarization system powered by a fine-tuned **T5-small Transformer**, **PyTorch**, **Hugging Face Transformers**, and a **FastAPI** serving backend with an editorial Vanilla JavaScript web interface.

---

## Overview & Problem Statement

Conversational text—such as chat messenger logs, customer support conversations, and meeting transcripts—is inherently unstructured, noisy, informal, and distributed across multiple speaker turns. Standard document summarization models often perform poorly on conversational data because they are primarily trained on formal, third-person prose like news articles.

**T5 Dialogue Summarizer** addresses this challenge by fine-tuning Google's **T5-small (Text-To-Text Transfer Transformer)** on the **SAMSum** dataset. The model learns to compress multi-turn dialogues into concise, accurate, third-person summaries while capturing key agreements, decisions, and speaker intentions.

---

## Key Features

- **Fine-Tuned Seq2Seq Model**: Fine-tuned `t5-small` architecture specialized for conversational turn-taking and dialogue comprehension.
- **FastAPI REST Service**: RESTful web API built on FastAPI and Uvicorn with Pydantic request validation and interactive Swagger documentation.
- **Editorial SaaS Web Interface**: Minimalist writing-tool aesthetic inspired by Linear, Notion, and Readwise—built strictly with semantic HTML5, modern CSS, and Vanilla JavaScript (zero external UI dependencies).
- **Interactive Productivity Controls**:
  - Live character and word counters updating on every keystroke.
  - One-click copy to clipboard with 2-second confirmation feedback.
  - Keyboard submission shortcut (`Ctrl + Enter` / `Cmd + Enter`).
  - Animated loading spinner and graceful error handling.
- **Clean Architecture**: Decoupled repository layout separating web services (`app/`), datasets (`data/`), and experimentation (`notebooks/`).
- **Interactive API Documentation**: Automated Swagger UI explorer available out of the box at `/docs`.

---

## Technology Stack

- **Model & Deep Learning**: PyTorch, Hugging Face Transformers (`T5ForConditionalGeneration`, `T5Tokenizer`), SentencePiece
- **Backend Framework**: FastAPI, Uvicorn (ASGI), Pydantic, Jinja2
- **Frontend Architecture**: Semantic HTML5, CSS3 (Custom Design System with CSS Variables), Vanilla JavaScript (ES6+)
- **Data Engineering**: Pandas, Regex Preprocessing

---

## Project Directory Structure

```text
T5-Dialogue-Summarizer/
├── app/
│   ├── app.py                      # FastAPI application routes & inference logic
│   ├── saved_summary_model/        # Local model weights directory (excluded from git)
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors       # 242 MB trained model weights
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css           # Minimal editorial CSS design system
│   │   └── js/
│   │       └── script.js           # Live counter, form submission, clipboard API
│   └── templates/
│       └── index.html              # Clean semantic HTML5 web template
│
├── data/
│   ├── samsum-train.csv            # SAMSum training split (14,732 dialogues)
│   ├── samsum-test.csv             # SAMSum test split (819 dialogues)
│   └── samsum-validation.csv       # SAMSum validation split (818 dialogues)
│
├── notebooks/
│   └── text_summarizer.ipynb       # Jupyter notebook for preprocessing & training
│
├── results/                        # Local training checkpoints (excluded from git)
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/
│   ├── checkpoint-2000/
│   ├── checkpoint-2500/
│   └── checkpoint-3000/
│
├── .gitignore                      # Excludes large binaries (>100MB) & checkpoints
├── LICENSE                         # MIT License
├── requirements.txt                # Pinned production & training dependencies
└── README.md                       # Repository documentation
```

---

## Model Weights & Setup Notice

> [!IMPORTANT]
> **Trained Model Availability:**
> The fine-tuned model files (`model.safetensors` ~242 MB / 230.83 MiB, `tokenizer.json` 2.42 MB, and config files) and intermediate checkpoints in `results/` (~4.1 GB) are currently stored **locally** on the author's machine. They are intentionally excluded from Git history via `.gitignore` to comply with GitHub's 100 MB per-file limit.
>
> **Hugging Face Model Status:**
> The model has **not yet been published** to Hugging Face Model Hub.
> - **Placeholder Link**: `https://huggingface.co/Mandeep-Parmar/t5-dialogue-summarizer` *(Reserved for a future release — currently inactive)*
> - Users cannot pull weights from Hugging Face yet; you must train the model locally using the included notebook or supply your own model files.

### How to Run on Your Own Machine (Train Locally)

Because the complete SAMSum dataset is pre-packaged inside the `data/` directory, you can train and export the fine-tuned model locally:

1. **Open the Training Notebook**: Launch Jupyter Lab, Jupyter Notebook, or VS Code:
   ```bash
   jupyter notebook notebooks/text_summarizer.ipynb
   ```
2. **Execute the Cells**: Run through the preprocessing and training cells. The notebook trains the `t5-small` model on the dialogues in `data/samsum-train.csv` and evaluates on `data/samsum-validation.csv`.
   * **Realistic Hardware Expectations**: On an NVIDIA GPU with CUDA, training 4,000 samples for 6 epochs took **~30.4 minutes** (1,825.5s). On CPU-only environments, training will take considerably longer (several hours).
   * **Quick CPU Testing Tip**: For a rapid local dry run on a CPU, you can temporarily reduce the sample size to `n=1000` in Cell 6 and `num_train_epochs=1` in Cell 22.
3. **Automatic Model Export (Cell 25)**: At the end of training, Cell 25 automatically exports the model weights and tokenizer directly into:
   ```text
   app/saved_summary_model/
   ├── config.json
   ├── generation_config.json
   ├── model.safetensors       # 242 MB fine-tuned weights
   ├── tokenizer.json          # 2.4 MB tokenizer vocabulary
   └── tokenizer_config.json
   ```
4. **Launch the Web App**: Once Cell 25 completes, proceed to run `uvicorn app.app:app --reload` to start the web application!

---

## Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Mandeep-Parmar/T5-Dialogue-Summarizer.git
cd T5-Dialogue-Summarizer
```

### Step 2: Create and Activate a Virtual Environment

**Using Conda (Recommended):**
```bash
conda create -n t5-summarizer python=3.10 -y
conda activate t5-summarizer
```

**Using standard `venv`:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> [!TIP]
> **For NVIDIA GPU (CUDA Acceleration):**
> If you have an NVIDIA GPU, install the CUDA-enabled PyTorch build:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 4: Generate Model Weights Locally
If you are running the repository on a fresh machine without pre-existing model weights, open and run the fine-tuning notebook:
```bash
jupyter notebook notebooks/text_summarizer.ipynb
```
*Cell 25 automatically exports the trained weights and tokenizer directly into `app/saved_summary_model/`.*

### Step 5: Run the Web Application

Start the FastAPI application with Uvicorn from the project root:

```bash
uvicorn app.app:app --reload
```

Once started, open your browser:
* **Web Application Interface**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Reference

### Summarize Dialogue

Submits a conversation transcript and returns an abstractive summary.

* **URL**: `/summarize/`
* **Method**: `POST`
* **Content-Type**: `application/json`

#### Request Payload
*(Using the exact test dialogue evaluated in Cell 29 of the notebook)*

```json
{
  "dialogue": "Reporter: In today's technology news, artificial intelligence continues to expand rapidly across industries, from healthcare to finance and education. Recent reports suggest that AI adoption has significantly increased over the past few years.\nReporter: Companies are investing heavily in machine learning systems to automate tasks, improve decision-making, and enhance customer experiences. However, this growth has also raised questions about job displacement and ethical concerns.\nExpert: AI systems are becoming more capable due to advances in deep learning and access to large datasets. These models can now perform complex tasks such as language understanding, image recognition, and even code generation.\nExpert: At the same time, there are valid concerns about bias in AI models, as they often reflect the data they are trained on. Ensuring fairness and transparency is becoming a key area of research.\nReporter: Governments and organizations are beginning to introduce regulations to guide the development and deployment of AI technologies. The goal is to balance innovation with accountability.\nExpert: Another challenge is explainability. Many modern AI systems, especially deep neural networks, operate as \"black boxes,\" making it difficult to understand how decisions are made.\nReporter: Experts also highlight the importance of responsible AI development, including data privacy, security, and long-term societal impact.\nExpert: Looking ahead, collaboration between researchers, policymakers, and industry leaders will be crucial to ensure that AI systems are developed and used in a safe and beneficial way."
}
```

#### Response (200 OK)
*(Actual summary generated by the fine-tuned T5 model in Cell 29)*

```json
{
  "summary": "ai technology continues to expand rapidly across industries, from healthcare to finance and education. ai adoption has significantly increased over the past few years. experts highlight the importance of responsible ai development, including data privacy, security, and long-term societal impact."
}
```

---

## Dataset Information & Attribution

The model was fine-tuned on the **SAMSum Corpus**, created by researchers at **Samsung R&D Institute Poland**:

| Split | Number of Dialogues | Source File |
|---|---|---|
| **Train** | 14,732 | `data/samsum-train.csv` |
| **Validation** | 818 | `data/samsum-validation.csv` |
| **Test** | 819 | `data/samsum-test.csv` |

Each sample contains:
- `dialogue`: Realistic multi-speaker chat transcripts mimicking mobile and desktop messaging apps.
- `summary`: A concise, human-written summary capturing decisions, agreements, and speaker context.

### Sampling Details
For the training run in `notebooks/text_summarizer.ipynb`, **simple random sampling** was applied (using `pandas.DataFrame.sample(..., random_state=42)`) to extract **4,000 training pairs** and **500 validation pairs** to balance training time with generalization.

### Dataset Citation & License
* **Paper**: Bogdan Gliwa, Iwona Mochol, Maciej Biesek, Przemyslaw Szczepaniak. *"SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization"*, Proceedings of the 2nd Workshop on New Frontiers in Summarization (EMNLP-IJCNLP 2019), pp. 70–79.
* **Dataset License**: [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).

---

## Model Training & Evaluation

The complete preprocessing and training pipeline is documented in [`notebooks/text_summarizer.ipynb`](notebooks/text_summarizer.ipynb).

### Training Hyperparameters
- **Base Architecture**: `google/t5-small` (~60 Million parameters)
- **Input / Output Token Limits**: 512 input tokens / 150 summary target tokens
- **Optimizer**: AdamW with weight decay `0.01`
- **Training Schedule**: 6 epochs with 500 warmup steps
- **Batch Size**: 8 per device (train and validation)
- **Device Resolution**: Code automatically checks and resolves hardware in this priority:
  1. `torch.backends.mps.is_available()` (Apple Silicon MPS)
  2. `torch.cuda.is_available()` (NVIDIA CUDA)
  3. `torch.device("cpu")` (CPU fallback)
  *(This training run was executed on **NVIDIA CUDA**)*

### Actual Training Progress & Losses
The training ran for **3,000 total global steps** across 6 epochs (total runtime: **1,825.5 seconds / ~30.4 minutes**):

| Epoch | Evaluation Training Loss | Validation Loss |
|:---:|:---:|:---:|
| 1 | 3.6278 | 0.3819 |
| 2 | 0.3959 | 0.3612 |
| 3 | 0.3740 | 0.3541 |
| 4 | 0.3628 | 0.3508 |
| 5 | 0.3555 | 0.3496 |
| **6** | **0.3516** | **0.3490** |

* **Overall Aggregate Training Loss**: `0.9113` (reported as `TrainOutput.training_loss` across the full 3,000 steps).
* **Final Validation Loss**: `0.3490` (steady convergence from 0.3819).

### Inference Decoding Strategy
During inference in both the notebook and `app/app.py`, summaries are generated using **Beam Search**:
- `num_beams`: 4 (generates 4 hypothesis beams and decodes the highest scoring sequence)
- `max_length`: 150 tokens
- `early_stopping`: True
- `skip_special_tokens`: True (strips EOS and padding tokens)
- *Note on backend execution*: In `app/app.py`, model generation runs synchronously on the loaded PyTorch model within FastAPI route handlers.

---

## License

This project is open-source and licensed under the [MIT License](LICENSE).

---

## Author & Contact

Developed by **Mandeep Parmar**

* **GitHub**: [@Mandeep-Parmar](https://github.com/Mandeep-Parmar)
* **Project Repository**: [T5-Dialogue-Summarizer](https://github.com/Mandeep-Parmar/T5-Dialogue-Summarizer)
