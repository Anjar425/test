import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
import torch
from transformers import BertTokenizer
from model import BertClassifier, clean_text

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertClassifier(num_classes=6)
model.load_state_dict(torch.load('bert_model.pth', map_location=torch.device('cpu'), weights_only=True), strict=False)
model.eval()

emotion_labels = ['caring', 'love', 'gratitude', 'sadness', 'fear', 'anger']

def predict_emotions(text, threshold=0.6):
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.sigmoid(outputs)
        predicted_classes = (probabilities > threshold).int()
        predicted_emotions = [emotion_labels[i] for i, val in enumerate(predicted_classes[0]) if val == 1]
        return {
            "predicted_emotions": predicted_emotions,
            "probabilities": probabilities.numpy().tolist()
        }

if __name__ == '__main__':
    input_text = sys.argv[1]
    prediction = predict_emotions(input_text)
    print(json.dumps(prediction))
