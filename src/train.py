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

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_texts, train_labels)
val_dataset = ToxicDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


