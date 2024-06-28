import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import logging

logger = logging.getLogger(__name__)

# Load the trained model and tokenizer
MODEL_DIRECTORY = 'app/parsers/text_analysis/trained_models'
LABEL_MAPPING_PATH = 'app/parsers/text_analysis/label_mapping.csv'
latest_model_dir = max([os.path.join(MODEL_DIRECTORY, d) for d in os.listdir(MODEL_DIRECTORY) if os.path.isdir(os.path.join(MODEL_DIRECTORY, d))], key=os.path.getmtime)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(latest_model_dir)
model = BertForSequenceClassification.from_pretrained(latest_model_dir)
model.to(device)


label_mapping_df = pd.read_csv(LABEL_MAPPING_PATH)
id_to_label = {row['id']: row['label'] for _, row in label_mapping_df.iterrows()}


def predict_class(text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_class = id_to_label[predicted_class_id]
    confidence = probabilities[0, predicted_class_id].item() * 100  # Convert to percentage

    logger.info(f"Text: '{text}' | Predicted class: '{predicted_class}' | Confidence: {confidence:.2f}%")
    return predicted_class, confidence


def analyze_texts(all_texts):
    for text_info in all_texts.get('texts', []):
        text = text_info['text']
        predicted_class = predict_class(text)
        text_info['predicted_class'] = predicted_class

    for text_info in all_texts.get('mtexts', []):
        text = text_info['text']
        predicted_class = predict_class(text)
        text_info['predicted_class'] = predicted_class

    for text_info in all_texts.get('attdefs', []):
        text = text_info['text']
        predicted_class = predict_class(text)
        text_info['predicted_class'] = predicted_class

    return all_texts


if __name__ == "__main__":
    example_texts = {
        "texts": [],
        "mtexts": [
            {"text": "Teostas:", "center": [884, 136], "text_direction": [1, 0, 0], "attachment_point": 1, "height": 8.0, "style": "Note Text (ISO)", "font": "tahoma.ttf", "color": "#000000"},
            {"text": "Kontrollis:", "center": [1124, 136], "text_direction": [1, 0, 0], "attachment_point": 1, "height": 8.0, "style": "Note Text (ISO)", "font": "tahoma.ttf", "color": "#000000"},
            {"text": "Kinnitas:", "center": [1364, 136], "text_direction": [1, 0, 0], "attachment_point": 1, "height": 8.0, "style": "Note Text (ISO)", "font": "tahoma.ttf", "color": "#000000"},
        ]
    }
    analyzed_results = analyze_texts(example_texts)
    for result in analyzed_results:
        print(result)