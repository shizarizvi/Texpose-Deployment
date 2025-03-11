import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
import numpy as np
from transformers import AutoTokenizer
import torch
import re
import nltk
from nltk.corpus import stopwords

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

    model_llm = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = True, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
    )

    model_ai_hum = torch.load('model/bert_model/data.pkl', map_location="cpu", weights_only=False)
    model_llm.load_state_dict(torch.load('model/bert_model_llm_only__2.pth', map_location="cpu"))

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

        with torch.no_grad():
            outputs_llm = model_llm(input_ids, attention_mask=attention_mask)
            logits_llm = outputs_llm.logits
            predicted_llm_pred = np.argmax(logits_llm.cpu().numpy(), axis=1).item()

        if predicted_llm_pred == 0:
            predicted_label_llm = "LLAMA"
        else:
            predicted_label_llm = "Gemini"

    return {"type": predicted_label, "llm": predicted_label_llm}

    
# Example usage:
# abstract = df["Abstract"].values[39000]
# result = classify_abstract(abstract, model_ai_hum, model_llm, tokenizer)
# print(result)
