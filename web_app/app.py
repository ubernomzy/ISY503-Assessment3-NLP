from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import threading
import os

app = Flask(__name__)

# -- Model state --
model_ready = False
model_error = None
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bert_finetuned')

def load_model():
    global model, tokenizer, model_ready, model_error
    try:
        print(f"[INFO] Loading tokenizer from {MODEL_PATH} ...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

        print(f"[INFO] Loading model from {MODEL_PATH} ...")
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        model_ready = True
        print(f"[INFO] Model loaded successfully on {device}.")
    except Exception as e:
        model_error = str(e)
        print(f"[ERROR] Failed to load model: {e}")

# Start loading immediately
threading.Thread(target=load_model, daemon=True).start()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "ready": model_ready,
        "error": model_error,
        "device": str(device)
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model_ready:
        return jsonify({"error": "Model is still loading. Please wait."}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400
        
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
   
    encoding = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    label = "Positive" if pred == 1 else "Negative"

    return jsonify({
        "label": label,
        "confidence": round(conf * 100, 2)
    })

if __name__ == "__main__":
    # Note: debug=False prevents the background thread from running twice
    app.run(host='127.0.0.1', port=5000, debug=False)
