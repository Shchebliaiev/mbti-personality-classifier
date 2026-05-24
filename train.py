import os
import re
import json
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from evaluate import load

# Initialize and download NLTK components
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

def preprocess_text(df):
    """
    Cleans text by removing URLs, non-alphabetic characters, repeat patterns,
    words that are too short/long, and explicit MBTI class names to avoid data leakage.
    """
    # Remove URLs
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'https?:\/\/.*?[ \s+]', '', x.replace("|||", " EOSTokenPost ").replace("|", " ") + " "))
    
    # Remove non-words (keep letters only)
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x + " "))
    
    # Convert to lowercase
    df["posts"] = df["posts"].apply(lambda x: x.lower())
    
    # Remove letter repetition (3 or more consecutive identical letters)
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*', '', x + " "))
    
    # Remove short (0-3 chars) or overly long (30+ chars) words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{0,3})?\b', '', x))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{30,1000})?\b', '', x))
    
    # Remove MBTI personality names to prevent direct data leakage
    pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 
                  'ENTJ', 'ISTJ', 'ENFJ', 'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
    pers_types_lower = [p.lower() for p in pers_types]
    pattern = re.compile(r"\b(" + "|".join(pers_types_lower) + r")\b")
    df["posts"] = df["posts"].apply(lambda x: re.sub(pattern, '', x))
    
    return df

def extract_and_combine_posts(row, combined_posts, min_length=3, max_length=256):
    """
    Processes user posts, splits them, lemmatizes words, and combines them into chunks 
    up to a specified maximum length to create high-quality, balanced input samples.
    """
    personality_type = row[0]
    raw_posts = row[1]
    current_post = ""

    for post in raw_posts.split("eostokenpost"):
        post = post.strip()
        # Lemmatize words longer than 3 characters
        post = " ".join([lemmatizer.lemmatize(word) for word in post.split() if len(word) > 3])
        post_len = len(post)

        if post_len < min_length:
            continue

        if len(current_post) + post_len >= max_length:
            combined_posts.append((personality_type, current_post.strip()))
            current_post = post
        else:
            current_post += " " + post if current_post else post

    if current_post.strip():
        combined_posts.append((personality_type, current_post.strip()))

def main():
    dataset_path = 'mbti_1.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at '{dataset_path}'.")
        print("Please place the Kaggle 'mbti_1.csv' dataset in this directory.")
        return

    print("Loading raw dataset...")
    df = pd.read_csv(dataset_path)
    
    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    # 1. Text Preprocessing
    print("Preprocessing text...")
    df = preprocess_text(df)

    # 2. Extracting and chunking posts
    print("Extracting and combining posts...")
    combined_posts = []
    # Using row tuples directly to avoid Pandas Series indexing warnings
    for row in df.itertuples(index=False):
        extract_and_combine_posts(row, combined_posts)
        
    data = pd.DataFrame(combined_posts, columns=['type', 'posts'])
    
    # Map text labels to integers
    label_mapping = {label: idx for idx, label in enumerate(data['type'].unique())}
    data['type_id'] = data['type'].map(label_mapping)
    
    # 3. Vectorization & Class Balancing (SMOTE)
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    posts_tfidf = vectorizer.fit_transform(data['posts'])
    type_ids = data['type_id']
    
    print("Class distribution before balancing:", Counter(type_ids))
    print("Applying SMOTE balancing (data augmentation)...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    posts_resampled, type_ids_resampled = smote.fit_resample(posts_tfidf, type_ids)
    print("Class distribution after balancing:", Counter(type_ids_resampled))
    
    # Convert resampled sparse matrix back to string text for BERT
    print("Reconstructing text dataset from balanced TF-IDF features...")
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    reconstructed_posts = vectorizer.inverse_transform(posts_resampled)
    reconstructed_posts = [" ".join(words) for words in reconstructed_posts]
    
    data_balanced = pd.DataFrame({
        'posts': reconstructed_posts,
        'type_id': type_ids_resampled,
        'type': [inverse_label_mapping[tid] for tid in type_ids_resampled]
    })

    # 4. Train-Test Split
    train_df, val_df = train_test_split(
        data_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=data_balanced['type']
    )

    # Initialize Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        tokenized = tokenizer(examples['posts'], padding="max_length", truncation=True, max_length=256)
        tokenized["labels"] = examples["type_id"]
        return tokenized

    train_dataset = Dataset.from_pandas(train_df[['posts', 'type_id']])
    val_dataset = Dataset.from_pandas(val_df[['posts', 'type_id']])

    print("Tokenizing train and validation splits...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
    val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4)

    # Load Model
    print("Initializing BERT Sequence Classification Model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
    model.to(device)

    # Training Metrics
    accuracy_metric = load("accuracy")
    def compute_metrics(pred):
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # Training Hyperparameters
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting BERT fine-tuning (This can take several hours depending on your GPU)...")
    trainer.train()

    print("Evaluating model...")
    results = trainer.evaluate()
    print(f"Final Evaluation Results: {results}")

    # 5. Safe Model Saving
    save_path = "./saved_model"
    print(f"Saving fine-tuned model and tokenizer to '{save_path}'...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    with open(os.path.join(save_path, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)
        
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
