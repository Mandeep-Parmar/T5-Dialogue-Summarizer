# T5 Dialogue Summarizer

> An end-to-end NLP dialogue and conversation summarization system powered by a fine-tuned **T5-small** transformer, served with a **FastAPI** backend and an editorial, minimal SaaS web interface.

---

## Overview

Conversational text (chats, meeting transcripts, support logs) is inherently noisy, informal, and disjointed. This project implements a fine-tuned **Google T5 (Text-To-Text Transfer Transformer)** on the **SAMSum** dialogue dataset to extract concise, abstractive summaries from multi-speaker conversations.

The web application features an editorial productivity-tool design inspired by Linear and Notion, complete with dynamic character/word counters, keyboard shortcuts (`Ctrl+Enter`), and one-click clipboard copying.

---

## Features

- **Fine-Tuned Seq2Seq Transformer**: Built on Hugging Face `t5-small`, fine-tuned for conversational summarization using beam search (4 beams).
- **FastAPI Backend**: Asynchronous REST API with Pydantic schema validation and Swagger UI documentation at `/docs`.
- **Editorial SaaS Frontend**: Clean HTML5, CSS3, and Vanilla JavaScript interface without bloated frontend dependencies.
- **Production-Ready Structure**: Clean separation of web services (`app/`), datasets (`data/`), and experimentation (`notebooks/`).
- **Interactive UI**: Real-time character & word counters, copy-to-clipboard with instant feedback, and accessible loading/error states.

---

## Project Structure

```text
T5-Dialogue-Summarizer/
├── app/
│   ├── app.py                      # FastAPI server & inference endpoint
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css           # Editorial design system & responsive layout
│   │   └── js/
│   │       └── script.js           # Live counter, submission handler, clipboard API
│   └── templates/
│       └── index.html              # Semantic HTML5 template
├── data/
│   ├── samsum-train.csv            # SAMSum training split
│   ├── samsum-test.csv             # SAMSum test split
│   └── samsum-validation.csv       # SAMSum validation split
├── notebooks/
│   └── text_summarizer.ipynb       # End-to-end fine-tuning pipeline & evaluation
├── .gitignore                      # Excludes large safetensors & checkpoints
├── requirements.txt                # Pinned dependencies
└── README.md                       # Documentation
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mandeep-Parmar/T5-Dialogue-Summarizer.git
cd T5-Dialogue-Summarizer
```

### 2. Set Up a Virtual Environment

```bash
# Using conda
conda create -n t5-summarizer python=3.10
conda activate t5-summarizer

# Or using venv
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web Application

Launch the FastAPI application with Uvicorn:

```bash
uvicorn app.app:app --reload
```

Open your browser at:
* **Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Reference

### Summarize Text

* **URL**: `/summarize/`
* **Method**: `POST`
* **Content-Type**: `application/json`

#### Request Body
```json
{
  "dialogue": "Hannah: Hey, are you free this afternoon?\nMark: Yes, what's up?\nHannah: Want to grab coffee and discuss the project slides?\nMark: Sounds great, let's meet at 3pm at Blue Bottle."
}
```

#### Response (200 OK)
```json
{
  "summary": "Hannah and Mark are meeting at 3pm at Blue Bottle for coffee to discuss the project slides."
}
```

---

## Model Training

The training notebook is available at [`notebooks/text_summarizer.ipynb`](notebooks/text_summarizer.ipynb).

### Training Configuration
* **Base Architecture**: `t5-small` (~60M parameters)
* **Dataset**: SAMSum dialogue dataset (14,732 dialogue-summary pairs)
* **Max Input Length**: 512 tokens
* **Max Target Length**: 150 tokens
* **Epochs**: 6
* **Batch Size**: 8 per device
* **Optimization**: AdamW with weight decay `0.01` and `500` warmup steps
* **Generation**: Beam search with 4 beams and early stopping

---

## Tech Stack

* **Machine Learning**: PyTorch, Hugging Face Transformers, SentencePiece
* **Backend**: FastAPI, Uvicorn, Pydantic, Jinja2
* **Frontend**: Semantic HTML5, Modern CSS (Custom Design System), Vanilla JavaScript (ES6+)

---

## License

This project is licensed under the MIT License.
