from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import threading
import os

app = Flask(__name__)

# ── Model state ───────────────────────────────────────────────────────
model_ready = False
model_error = None
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Get the directory that the current script is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# saved by model_bert.ipynb
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bert_finetuned')
 
 
# ── Load model in background thread so Flask starts immediately ───────
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
 
 
# Start loading immediately when the app boots
threading.Thread(target=load_model, daemon=True).start()
 
 
# ── Routes ────────────────────────────────────────────────────────────

    @app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """
    Polled by the frontend to know when the model is ready.
    Returns:
        ready (bool)  — True once weights are loaded and model.eval() has run
        error (str)   — set if loading failed, otherwise null
        device (str)  — 'cuda' or 'cpu'
    """
    return jsonify({
        "ready": model_ready,
        "error": model_error,
        "device": str(device)
    })
 

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON { "text": "..." }
    Returns  JSON { "label": "Positive"|"Negative", "confidence": 94.3 }
    """
       if not model_ready:
        return jsonify({"error": "Model is still loading. Please wait."}), 503
    
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
   
    # Tokenise — max_length=256 matches fine-tuning settings in model_bert.ipynb.
    # If the notebook used 128, change this to match.
    encoding = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
 
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
 
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1)
        pred    = torch.argmax(probs, dim=1).item()
        conf    = probs[0][pred].item()
 
    # Label mapping: 0 = Negative, 1 = Positive
    # Must match the encoding used during training in model_bert.ipynb
    label = "Positive" if pred == 1 else "Negative"
 
    return jsonify({
        "label":      label,
        "confidence": round(conf * 100, 2)
    })
 
 if __name__ == "__main__":
    app.run(debug=True)
