import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
stopwords = stopwords.words('english')

def remove_stop_words(text):
    words = [w for w in text if w not in stopwords]
    return words

lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

stemmer = PorterStemmer()

def word_stemmer(text):
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text

effects = ["No Side Effects", 
"Moderate Side Effects",
 "Mild Side Effects",
  "Severe Side Effects", 
  "Extremely Severe Side Effects"]

def convert_to_index(text):
    if text == "No Side Effects":
        return 0
    elif text == "Moderate Side Effects":
        return 1
    elif text == "Mild Side Effects":
        return 2
    elif text == "Severe Side Effects":
        return 3
    else:
        return 4

test = pd.read_csv('drugLibTest_raw.tsv',delimiter='\t')    # Read the files with the pandas dataFrame
train = pd.read_csv('drugLibTrain_raw.tsv', delimiter='\t')
test.shape
train.shape
train = train.drop("Unnamed: 0", 1)
test = test.drop("Unnamed: 0", 1)

#Tokenization
tokenizer = RegexpTokenizer(r'\w+')
train["benefitsReview"] = train["benefitsReview"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))
train["sideEffectsReview"] = train["sideEffectsReview"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))
train["commentsReview"] = train["commentsReview"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))
test["benefitsReview"] = test["benefitsReview"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))
test["sideEffectsReview"] = test["sideEffectsReview"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))
test["commentsReview"] = test["commentsReview"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))

#Remove stopwords
train["benefitsReview"] = train["benefitsReview"].apply(lambda x: remove_stop_words(x))
train["sideEffectsReview"] = train["sideEffectsReview"].apply(lambda x: remove_stop_words(x))
train["commentsReview"] = train["commentsReview"].apply(lambda x: remove_stop_words(x))
test["benefitsReview"] = test["benefitsReview"].apply(lambda x: remove_stop_words(x))
test["sideEffectsReview"] = test["sideEffectsReview"].apply(lambda x: remove_stop_words(x))
test["commentsReview"] = test["commentsReview"].apply(lambda x: remove_stop_words(x))

#Lemmatization
train["benefitsReview"] = train["benefitsReview"].apply(lambda x: word_lemmatizer(x))
train["sideEffectsReview"] = train["sideEffectsReview"].apply(lambda x: word_lemmatizer(x))
train["commentsReview"] = train["commentsReview"].apply(lambda x: word_lemmatizer(x))
test["benefitsReview"] = test["benefitsReview"].apply(lambda x: word_lemmatizer(x))
test["sideEffectsReview"] = test["sideEffectsReview"].apply(lambda x: word_lemmatizer(x))
test["commentsReview"] = test["commentsReview"].apply(lambda x: word_lemmatizer(x))

#Stemmerization
train["benefitsReview"] = train["benefitsReview"].apply(lambda x: word_stemmer(x))
train["sideEffectsReview"] = train["sideEffectsReview"].apply(lambda x: word_stemmer(x))
train["commentsReview"] = train["commentsReview"].apply(lambda x: word_stemmer(x))
test["benefitsReview"] = test["benefitsReview"].apply(lambda x: word_stemmer(x))
test["sideEffectsReview"] = test["sideEffectsReview"].apply(lambda x: word_stemmer(x))
test["commentsReview"] = test["commentsReview"].apply(lambda x: word_stemmer(x))

train["sideEffects"] = train["sideEffects"].apply(lambda x: convert_to_index(x))
test["sideEffects"] = test["sideEffects"].apply(lambda x: convert_to_index(x))

train["condition"] = train['condition'].apply(lambda x: str(x)).tolist()
test["condition"] = test['condition'].apply(lambda x: str(x)).tolist()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

label_encode = ["urlDrugName", "condition", "benefitsReview", "sideEffectsReview", "commentsReview", "rating", "effectiveness"]

for feature in label_encode:
    train[feature] = le.fit_transform(train[feature].values)
    test[feature] = le.fit_transform(test[feature].values)

x_train = train.drop("sideEffects", 1)
y_train = train["sideEffects"]
x_test = test.drop("sideEffects", 1)
y_test = test["sideEffects"]

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("MLP")
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(max_iter=1000)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

print("Decision Tree")
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=10)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))