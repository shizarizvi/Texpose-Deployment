from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from classifier_model import load_models, classify_text

print("svc_texpose - starting")

# Load models
tokenizer, model_ai_hum, model_llm = load_models()

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "input_text": ""})

@app.post("/classify", response_class=HTMLResponse)
async def classify(request: Request, input_text: str = Form(...)):
    if input_text.strip():  # Ensure input is not empty
        result = classify_text(input_text, model_ai_hum, model_llm, tokenizer)
        prediction = result["type"]

        # Ensure "llm" exists and is not empty
        if result.get("llm"):
            prediction += f" Using {result['llm']}"

        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "input_text": input_text})

    return templates.TemplateResponse("index.html", {"request": request, "prediction": "No text provided.", "input_text": ""})