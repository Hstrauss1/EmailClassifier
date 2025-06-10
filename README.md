# Email Author Classification using TF-IDF and Neural Network with PCA

This project is focused on classifying the authors of email messages based on their content. It uses data from the Enron email dataset and implements a machine learning pipeline involving text preprocessing, TF-IDF vectorization, dimensionality reduction via PCA, and classification using a neural network.

## 📁 Project Structure

- **Text Preprocessing**: Emails are cleaned by removing URLs, punctuation, and stopwords, and then stemmed.
- **Vectorization**: Processed text is transformed into TF-IDF vectors with unigrams and bigrams.
- **Filtering**: Authors with fewer than 2 emails are removed to ensure meaningful learning.
- **Dimensionality Reduction**: PCA is applied to reduce the feature space and improve model performance.
- **Modeling**: A neural network is trained to classify authors based on their email content.

## 🚀 Features

- Efficient text cleaning using NLTK.
- Scalable vectorization using `TfidfVectorizer`.
- Dimensionality reduction using `PCA` from `sklearn`.
- Neural network built using `MLPClassifier` from `sklearn`.
- Evaluation via classification report and confusion matrix.

## 🧪 Requirements

- Python 3.x
- Scikit-learn
- NLTK
- NumPy
- Pandas
- Matplotlib (for visualization)

Install dependencies with:

```bash
pip install scikit-learn nltk numpy pandas matplotlib
```

## 🗂️ Data

The project assumes access to a subset of the Enron email dataset, specifically:
```
/WAVE/projects/CSEN-140-Sp25/HHJ140Proj/Sent_Items_only
```
Ensure this directory structure exists and contains subfolders for each user, each with a `sent_items` folder of email files.

## 🛠️ Usage

You can run the notebook step-by-step to:

1. Clean and preprocess the email data.
2. Vectorize the text with TF-IDF.
3. Reduce dimensions with PCA.
4. Train and evaluate a neural network.

## 📈 Results

Performance is evaluated using metrics such as precision, recall, and F1-score, along with confusion matrices for insight into misclassifications.

## 📌 Notes

- The project is sensitive to the dataset structure and email formatting.
- Make sure NLTK stopwords are downloaded before use (`nltk.download('stopwords')` is included).
