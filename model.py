import re
import emoji
from bs4 import BeautifulSoup
import torch.nn as nn
from transformers import BertModel

def clean_text(text):
    text = str(text)
    text = emoji.demojize(text)
    text = re.sub(r'\:(.*?)\:', '', text)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text

class BertClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5, num_classes=6):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        return self.fc2(pooled_output)
