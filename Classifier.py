#!/usr/bin/env python3

import os
import re
import argparse
from email import policy
from email.parser import BytesParser

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Lowercase, remove non-word chars, strip extra spaces."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)      # URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)               # punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_emails(maildir: str):
    texts, labels = [], []
    for user in os.listdir(maildir):
        user_dir = os.path.join(maildir, user)
        if not os.path.isdir(user_dir):
            continue
        for folder in os.listdir(user_dir):
            folder_dir = os.path.join(user_dir, folder)
            if not os.path.isdir(folder_dir):
                continue
            for fname in os.listdir(folder_dir):
                path = os.path.join(folder_dir, fname)
                try:
                    with open(path, 'rb') as f:
                        msg = BytesParser(policy=policy.default).parse(f) # build message in binaryy 
                    body = msg.get_body(preferencelist=('plain',)) # turn into tsring
                    if body is None: 
                        continue
                    raw = body.get_content()
                    text = clean_text(raw)
                    if not text:
                        continue
                    # data cleaning
                    tokens = [w for w in text.split() if w not in STOPWORDS] #cleanning. Added lemmentatizn or stemming?
                    texts.append(' '.join(tokens))
                    labels.append(user)
                except Exception:
                    continue
    return texts, labels

def main(data_dir: str, test_size: float, random_state: int):
    print("Loading and cleaning emails…")
    texts, labels = load_emails(data_dir)
    print(f"Loaded {len(texts)} messages from {len(set(labels))} authors.")
 
    print("Vectorizing with TF–IDF…")
    vect = TfidfVectorizer(max_features=20000) #term frequency vectorizory
    X = vect.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    ) #seperate data into training and test sets

    models = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(max_iter=200), # classifincation moddels
        "Decision Tree": DecisionTreeClassifier(max_depth=20)
    }

    for name, model in models.items():
        print(f"\nTraining {name}…")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"--- {name} Evaluation ---")
        print(classification_report(y_test, preds))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="maildir",
        help="Path to the unpacked Enron maildir folder"
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing"
    )
    p.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed"
    )
    args = p.parse_args()
    main(args.data_dir, args.test_size, args.random_state)
