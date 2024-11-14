import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.tsv", sep="\t")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
ps = PorterStemmer()
corpus = []

for x in range(0, len(df)):
    review = re.sub(
        "[^a-zA-Z]", " ", df["Review"][x]
    ).lower()  # bersihin char selain alphabet terus ganti sama spasi
    review = review.split()
    review = [ps.stem(x) for x in review if x not in stopwords.words("english")]
    review = " ".join(review)
    corpus.append(review)

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = svm.SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Confusion Matrix: {conf_mat}")
