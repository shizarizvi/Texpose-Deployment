from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.classifier_model import load_models, classify_text
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse


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




@app.post("/api/classify")
async def classify_api(request: Request):
    data = await request.json()
    input_text = data.get("input_text", "")

    if input_text.strip():
        result = classify_text(input_text, model_ai_hum, model_llm, tokenizer)
        prediction = result["type"]

        if result.get("llm"):
            prediction += f" Using {result['llm']}"

        return JSONResponse(content={"prediction": prediction})
    
    return JSONResponse(content={"prediction": "No text provided."})
