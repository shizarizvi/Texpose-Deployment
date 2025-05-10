from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.classifier_model import load_models, classify_text
import os
from fastapi.staticfiles import StaticFiles


print("svc_texpose - starting")

# Load models
tokenizer, model_ai_hum, model_llm = load_models()

# Initialize FastAPI app
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets /app/app/
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "input_text": ""})

@app.post("/classify", response_class=HTMLResponse)
async def classify(request: Request, input_text: str = Form(default="")):
    if input_text.strip():  # Ensure input is not empty
        result = classify_text(input_text, model_ai_hum, model_llm, tokenizer)
        prediction = result["type"]

        # Ensure "llm" exists and is not empty
        if result.get("llm"):
            prediction += f" Using {result['llm']}"

        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "input_text": input_text})

    return templates.TemplateResponse("index.html", {"request": request, "prediction": "No text provided.", "input_text": ""})



