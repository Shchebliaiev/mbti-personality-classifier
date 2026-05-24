import os
import re
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict_social_type(text, model, tokenizer, device, reverse_label_mapping):
    # Preprocess custom input
    text = text.lower()
    text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', text)
    text = re.sub(r'[^0-9a-z]', ' ', text)
    
    # Tokenize input text
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
    
    # Run prediction
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get highest probability index and convert to class label
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    predicted_label = reverse_label_mapping[prediction]
    
    return predicted_label

def main():
    # Define model path (can point to local folder or Google Drive mount)
    model_path = "./saved_model"
    
    if not os.path.exists(model_path):
        print(f"Error: Model folder not found at '{model_path}'.")
        print("Please ensure your fine-tuned BERT model, tokenizer, and 'label_mapping.json' are saved in this directory.")
        return
        
    # Detect processing unit (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    
    # Load Label Mapping
    mapping_path = os.path.join(model_path, "label_mapping.json")
    with open(mapping_path, "r") as f:
        label_mapping = json.load(f)
    reverse_label_mapping = {int(val): key for key, val in label_mapping.items()}
    
    print("\nModel successfully loaded! Ready for personality type prediction.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Enter text to analyze: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break
        if not user_input.strip():
            continue
            
        predicted_type = predict_social_type(user_input, model, tokenizer, device, reverse_label_mapping)
        print(f"Predicted MBTI Type: {predicted_type}\n")

if __name__ == "__main__":
    main()
