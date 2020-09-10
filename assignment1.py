import pandas as pd
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
    lem_text = ([lemmatizer.lemmatize(i) for i in text])
    return lem_text

stemmer = PorterStemmer()

def word_stemmer(text):
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text

def convert_side_effect_index(text):
    if text == "No Side Effects":
        return 0
    elif text == "Mild Side Effects":
        return 1
    elif text == "Moderate Side Effects":
        return 2
    elif text == "Severe Side Effects":
        return 3
    else:
        return 4

def convert_effectiveness_index(text):
    if text == "Ineffective":
        return 0
    elif text == "Marginally Effective":
        return 1
    elif text == "Moderately Effective":
        return 2
    elif text == "Considerably Effective":
        return 3
    else:
        return 4
test = pd.read_csv('drugLibTest_raw.tsv',delimiter='\t')    # Read the files with the pandas dataFrame
train = pd.read_csv('drugLibTrain_raw.tsv', delimiter='\t')

train.insert(1, column="review", value=None)
train["review"] = train["benefitsReview"] + train["sideEffectsReview"]  + train["commentsReview"]
train = train.drop(["Unnamed: 0", "benefitsReview", "sideEffectsReview", "commentsReview"], axis=1)
test["review"] = test["benefitsReview"] + test["sideEffectsReview"]  + test["commentsReview"]
test = test.drop(["Unnamed: 0", "benefitsReview", "sideEffectsReview", "commentsReview"], axis=1)

#Tokenization
#Converts the text into a list of tokens and removes any punctuation
tokenizer = RegexpTokenizer(r'\w+')
train["review"] = train["review"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))
test["review"] = test["review"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))

# Remove stopwords 
train["review"] = train["review"].apply(lambda x: remove_stop_words(x))
test["review"] = test["review"].apply(lambda x: remove_stop_words(x))

# Lemmatization
train["review"] = train["review"].apply(lambda x: word_lemmatizer(x))
test["review"] = test["review"].apply(lambda x: word_lemmatizer(x))

#Stemmerization
train["review"] = train["review"].apply(lambda x: word_stemmer(x))
test["review"] = test["review"].apply(lambda x: word_stemmer(x))

train["sideEffects"] = train["sideEffects"].apply(lambda x: convert_side_effect_index(x))
test["sideEffects"] = test["sideEffects"].apply(lambda x: convert_side_effect_index(x))

train["effectiveness"] = train["effectiveness"].apply(lambda x: convert_effectiveness_index(x))
test["effectiveness"] = test["effectiveness"].apply(lambda x: convert_effectiveness_index(x))

from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer(max_features=150)
x = vectorizer.fit_transform(train["review"]).toarray()
x_train = pd.DataFrame(x, columns =vectorizer.get_feature_names())
x_train["rating"] = train["rating"]
x_train["effectiveness"] = train["effectiveness"]
    
x = vectorizer.fit_transform(test["review"]).toarray()
x_test = pd.DataFrame(x, columns =vectorizer.get_feature_names())
x_test["rating"] = test["rating"]
x_test["effectiveness"] = test["effectiveness"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_encode = ["urlDrugName", "condition"]
for feature in label_encode:
    train[feature] = le.fit_transform(train[feature].astype(str))
    test[feature] = le.fit_transform(test[feature].astype(str))

y_train = train["sideEffects"]
y_test = test["sideEffects"]

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("SVM")
from sklearn.svm import SVC
classifier = SVC(gamma='auto', C=1, class_weight='balanced')
classifier.fit(x_train, y_train)

target_names = ['No Side Effects', 'Mild Side Effects', 'Moderate Side Effects', 'Severe Side Effects', 'Extremely Severe Side Effects']
y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred, target_names=target_names))
print(accuracy_score(y_test, y_pred))

print("Decision Tree")
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=6, random_state=1)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred, target_names=target_names))
print(accuracy_score(y_test, y_pred))
from sklearn.metrics import precision_score
print()
# Grid search to find the optimal parameters
# from sklearn.model_selection  import GridSearchCV
# param_grid = {'C':[1,10,100,1000], 'gamma':[1,0.1,0.001,0.0001,'auto'], 'kernel':['linear','rbf'],'class_weight' : ['balanced', None]}
# grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
# grid.fit(x_train,y_train)
# print(grid.best_params_)

# param_grid = {'max_depth': np.arange(3, 10), "random_state" :[0, 1, None]}
# grid = GridSearchCV(DecisionTreeClassifier(), param_grid)
# grid.fit(x_train,y_train)
# print(grid.best_params_)