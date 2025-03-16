print("svc_texpose - starting")

from classifier_model import load_models, classify_text
from flask import Flask, render_template, request

tokenizer, model_ai_hum, model_llm = load_models()

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "GET":
        return {"svc_Texpose is alive :) "}
    
    if request.method == "POST":
        input_text = request.form.get("text", "")
        if input_text.strip():  #  input if not empty then proceed
            result = classify_text(input_text, model_ai_hum, model_llm, tokenizer)
            prediction = result["type"]
            if len(result["llm"]) > 1:
                prediction += " Using " + result["llm"]
        
            print(prediction)
    return render_template("index.html", prediction=prediction if prediction is not None else "")


if __name__ == "__main__":
    app.run(debug=True)

