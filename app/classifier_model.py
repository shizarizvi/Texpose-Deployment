import transformers
from transformers import BertForSequenceClassification,BertTokenizer
import numpy as np
import torch
import re
import nltk
from nltk.corpus import stopwords
import gdown
import os

file_path = "bert_model.pth"  # ML model's path

if not os.path.exists(file_path):
    print('Model not found locally, downloading from Drive...')

    def download_model(file_id, output_name):
        current_dir = os.getcwd()
        output_path = os.path.join(current_dir, output_name)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)


    # Model file IDs from Google Drive
    bert_model_id = "14OBJIgUtGLujlCzEaBb2Mxc5eUsAMZk5"
    cascade_bert_model_id = "14Shk7Yt6ilSrzFvppjSsqqZBv2RM1qwt"

    # Download models
    download_model(bert_model_id, "bert_model.pth")
    download_model(cascade_bert_model_id, "cascade_bert_model.pth")

    print('Models downloaded locally!')

else:
    print('Using pre-downloaded local models...')

nltk.download('stopwords')
sw = stopwords.words('english') 

def load_models(): 

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model_ai_hum = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = True, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
    )

    model_llm = model_ai_hum

    model_ai_hum = torch.load('bert_model.pth', map_location="cpu", weights_only=False)
    model_llm.load_state_dict(torch.load('cascade_bert_model.pth', map_location="cpu"))

    return tokenizer, model_ai_hum, model_llm


def clean_text(text):
    try:
        text = text.lower()
    except Exception as e:
        print(text)
        print(e)
        return
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') #Removing punctuations
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = " ".join(text) #removing stopwords
    return text

def classify_text(text, model_ai_hum, model_llm, tokenizer):

    assert tokenizer is not None, "Error: tokenizer is None!"
    encoded_input = tokenizer.encode_plus(
        clean_text(text),
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"].to("cpu")
    attention_mask = encoded_input["attention_mask"].to("cpu")

    with torch.no_grad():
        outputs = model_ai_hum(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_pred = np.argmax(logits.cpu().numpy(), axis=1).item()

    predicted_label      = ''
    predicted_label_llm  = '' 

    if predicted_pred == 1:
        predicted_label = "Human"
    else:
        predicted_label = "AI"

        # with torch.no_grad():
        #     outputs_llm = model_llm(input_ids, attention_mask=attention_mask)
        #     logits_llm = outputs_llm.logits
        #     predicted_llm_pred = np.argmax(logits_llm.cpu().numpy(), axis=1).item()


        predicted_llm_pred = 0
        if predicted_llm_pred == 0:
            predicted_label_llm = "LLAMA"
        else:
            predicted_label_llm = "Gemini"

    return {"type": predicted_label, "llm": predicted_label_llm}
