from flask import Flask, render_template, request 
import transformers 
from transformers import pipeline, AutoTokenizer
import torch

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def llama():
    if request.method == 'POST':
        prompt = request.form["prompt"] 
        
    model = "meta-llama/Llama-2-7b-chat-hf"
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
    
    llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    )
    
    def get_llama_response(prompt):
        sequences = llama_pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        return sequences[0]['generated_text']
    
    results = get_llama_response(prompt)
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)