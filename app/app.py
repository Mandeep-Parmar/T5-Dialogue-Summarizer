from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates  # UI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_summary_model")

# Initialize out fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Load Model and Tokenizer 
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# Device 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Templating & Paths
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)  # lines
    text = re.sub(r"\s+", " ", text)   # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags
    text = text.strip().lower()

    return text

def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # Clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate summary => token_ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4, # produce 4 summary and return best 1
        early_stopping=True
    )

    # token ids convert to summary => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # skip -> EOS, SEP

    return summary

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary} # return JSON res

# Custom 404 Exception Handler
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        accept = request.headers.get("accept", "")
        # Render custom 404 HTML template for browser requests
        if "text/html" in accept:
            return templates.TemplateResponse(
                request=request,
                name="404.html",
                status_code=404
            )
        # Return standard JSON response for API requests
        return JSONResponse(status_code=404, content={"detail": exc.detail or "Not Found"})

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
