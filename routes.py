from flask import Blueprint, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from config import Config
import threading
import re

routes = Blueprint('routes', __name__)

# Hugging Face token from config
HF_TOKEN = Config.HF_TOKEN
model_name = Config.MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    token=HF_TOKEN
)

lock = threading.Lock()

def sanitize_input(input_text):
    return re.sub(r'[^\w\s]', '', input_text).strip()

def get_response(prompt, max_tokens, temperature):
    with lock:
        sequences = text_generator(
            prompt, 
            max_new_tokens=max_tokens, 
            temperature=temperature
        )
    gen_text = sequences[0]["generated_text"]
    return gen_text

@routes.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    
    prompt = sanitize_input(data.get("prompt", ""))
    max_tokens = data.get("max_tokens", Config.MAX_TOKENS)
    temperature = data.get("temperature", Config.TEMPERATURE)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    response = get_response(prompt, max_tokens, temperature)
    return jsonify({"response": response})

@routes.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running"})
