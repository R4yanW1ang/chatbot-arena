# chatbot-arena
Deployed LLM that accepts prompt (a question) + responses from two Large language models. 
The app returns human preferance on these two responses (a tie is also possible).

<b>
base_model: unsloth/gemma-2-9b-it-bnb-4bit
library_name: transformers, peft
</b>

## How to Get Started with the Model

Use the code below to get started with the model.

model.py

## Training Details

### Training Data

100K+ Datapoints, including columns: prompt + response_a + response_b

### Training Procedure

Data Pre-processing -> Tokenization -> Fine-Tuning (1 epoch) -> Add the QLora weight to the freezed Gemma weight for final inference.


## Evaluation

Log-loss + Accuracy (after softmax)

### Testing Data, Factors & Metrics

#### Validation Data

Train-validation split (80%, 20%)

### Results

Log-loss = 0.912
