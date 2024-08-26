from flask import Flask, render_template, request
import torch
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from peft import PeftModel
import os

app = Flask(__name__)

# Load the model and tokenizer once at startup
model_checkpoint = '../gemma-2-9b-it-4bit'
inference_checkpoint = '../code/model_output/checkpoint-20/'

tokenizer = GemmaTokenizerFast.from_pretrained(model_checkpoint)
model = Gemma2ForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3,
    device_map="auto",
    use_cache=False
)
model = PeftModel.from_pretrained(model, inference_checkpoint)
model.eval()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        response_a = request.form['response_a']
        response_b = request.form['response_b']

        # Prepare input for the model
        input_text = f"{prompt} <response_a>: {response_a} <response_b>: {response_b}"
        inputs = tokenizer(input_text, max_length=2048, truncation=True, padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1).cpu().numpy()

        # Determine the preferred model based on probabilities
        winner = probs.argmax(-1)[0]
        if winner == 0:
            result = "Model A is preferred."
        elif winner == 1:
            result = "Model B is preferred."
        else:
            result = "There is a tie."

        return render_template('index.html', result=result, prompt=prompt, response_a=response_a, response_b=response_b)

    return render_template('index.html', result=None)


if __name__ == '__main__':
    app.run(debug=True)
