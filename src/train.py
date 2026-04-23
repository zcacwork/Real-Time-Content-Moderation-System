#
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from preprocess import load_data

# Load dataset
df = load_data("../data/train.csv")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['comment_text'], df['toxic_label'], test_size=0.1
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class ToxicDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True)
        self.labels = list(labels)
