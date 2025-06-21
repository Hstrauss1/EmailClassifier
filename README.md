### CHECK The  "FinalFolder" for the associated paper with results and final code

#  Email Author Classification with TF-IDF & Classical ML

Classifies the **sender** of an email purely from its text.  
Built on the Enron corpus and a scikit-learn pipeline that cleans, vectorises, and feeds messages into several traditional classifiers (best so far: a tuned Linear SVC).

---

##  Project Map

| Stage | What Happens |
|-------|--------------|
| **1. Ingest** | Recursively loads every file in `Sent_Items_only/**/sent_items/`, storing the raw body and owner. |
| **2. Clean** | Lower-cases, strips URLs / punctuation, removes NLTK stop-words, then stems **and** lemmatises each token. |
| **3. Vectorise** | `TfidfVectorizer` with unigrams + bigrams, `min_df=5`. |
| **4. Filter** | Drops authors with \< 2 messages to avoid single-sample classes. |
| **5. Split** | 70 % training / 30 % test (`random_state=36`). |
| **6. Model** | Out-of-the-box models ⬇️ — followed by C-sweep tuning for Linear SVC. |
| **7. Evaluate** | Accuracy, precision/recall/F1, and confusion matrices. Optional plots show “accuracy vs C”. |

### Models Tried

* Multinomial NB  
* Logistic Regression  
* Ridge Classifier  
* SGD (Log-loss)  
* Passive-Aggressive  
* Decision Tree  
* Random Forest  
* k-NN (cosine)  
* **Linear SVC** ← best so far (~0.9 accuracy with C ≈ 0.4)
* Further we had a comparison testbench with a RNN more information in the branch and final code file
