#
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from preprocess import load_data

df = load_data("../data/train.csv")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['comment_text'], df['toxic_label'], test_size=0.1
)
