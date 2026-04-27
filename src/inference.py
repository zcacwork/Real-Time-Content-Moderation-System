import onnxruntime as ort
import numpy as np
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("../models/")
session = ort.InferenceSession("../models/model.onnx")

def predict(text):
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=128)

    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    prediction = np.argmax(probs)

    return "Toxic" if prediction == 1 else "Clean"

# Test
while True:
    text = input("Enter comment: ")
    print("Prediction:", predict(text))
