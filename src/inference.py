
import onnxruntime as ort
import numpy as np
from transformers import DistilBertTokenizer

import onnxruntime as ort
import numpy as np
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("../models/")
session = ort.InferenceSession("../models/model.onnx")

def predict(text):
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=128)
