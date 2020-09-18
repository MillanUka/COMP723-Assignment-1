import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
stopwords = stopwords.words('english')
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
#Remove the stopwords that have low semantic value
def remove_stop_words(text):
    words = [w for w in text if w not in stopwords]
    return words

#Turns all the tokens into lemmas
#Returns an array of text
def word_lemmatizer(text):
    lem_text = []
    for i in text:
        pos = i[1]
        #Convert the tag so wordnet lemmatizer can use it
        pos = convert_tag(pos)
        x = None
        #If no tag is specified just lemmatize without a tage
        if pos == '':
            x = lemmatizer.lemmatize(i[0])
        else:
            x = lemmatizer.lemmatize(i[0], pos=pos)
        lem_text.append(x)
    return lem_text

#Converts the nltk POS tags into POS tags that the wordnet lemmatizer can use
def convert_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

# Stem the text using the porter stemmer
def word_stemmer(text):
    stem_text = []
    for i in text:
        x = stemmer.stem(i)
        stem_text.append(x)
    return stem_text 

#Convert the sides from categorical text to an int
#The models will out a number ranging from 0-4 corresponding to the respectie labels
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

#Converts the effectiveness from categorical text to an int
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

from nltk.tag import pos_tag
#Use the nltk POS tagger and gives each text a POS tag 
#returns a two element array. First element is the text. 
#Second is the POS tag
def tag(text):
    return pos_tag(text)

#Convert the array of tokens into one string per data point
def tokens_to_string(text):
    string = ""
    for i in text:
        string += " " + i
    return string

#Reads the training and testing sets into a pandas dataframe
test = pd.read_csv('drugLibTest_raw.tsv',delimiter='\t')    
train = pd.read_csv('drugLibTrain_raw.tsv', delimiter='\t')

#Adds the review column with no values
#concatenates all the review columsn into the review column
#Removes redundant columsn such as the benefitsReview, etc. columns
train.insert(1, column="review", value=None)
train["review"] = train["benefitsReview"] + train["sideEffectsReview"]  + train["commentsReview"]
train = train.drop(["Unnamed: 0", "benefitsReview", "sideEffectsReview", "commentsReview", "urlDrugName", "condition"], axis=1)
test["review"] = test["benefitsReview"] + test["sideEffectsReview"]  + test["commentsReview"]
test = test.drop(["Unnamed: 0", "benefitsReview", "sideEffectsReview", "commentsReview", "urlDrugName", "condition"], axis=1)

#Tokenization
#Converts the text into a list of tokens and removes any punctuation
tokenizer = RegexpTokenizer(r'\w+')
train["review"] = train["review"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))
test["review"] = test["review"].apply(lambda x: tokenizer.tokenize((str)(x).lower()))

# Remove stopwords 
train["review"] = train["review"].apply(lambda x: remove_stop_words(x))
test["review"] = test["review"].apply(lambda x: remove_stop_words(x))

#POS tagging
train["review"] = train["review"].apply(lambda x: tag(x))
test["review"] = test["review"].apply(lambda x: tag(x))

# Lemmatization
train["review"] = train["review"].apply(lambda x: word_lemmatizer(x))
test["review"] = test["review"].apply(lambda x: word_lemmatizer(x))

#uncomment to stem the data
# Stemmerization
# train["review"] = train["review"].apply(lambda x: word_stemmer(x))
# test["review"] = test["review"].apply(lambda x: word_stemmer(x))

#convert the array of tokens to string 
train["review"] = train["review"].apply(lambda x: tokens_to_string(x))
test["review"] = test["review"].apply(lambda x: tokens_to_string(x))

#convert the side effects into an int
train["sideEffects"] = train["sideEffects"].apply(lambda x: convert_side_effect_index(x))
test["sideEffects"] = test["sideEffects"].apply(lambda x: convert_side_effect_index(x))

#convert the effectiveness into an int
train["effectiveness"] = train["effectiveness"].apply(lambda x: convert_effectiveness_index(x))
test["effectiveness"] = test["effectiveness"].apply(lambda x: convert_effectiveness_index(x))

# Vectorization
# The vocab is restricted to 150 words
from sklearn.feature_extraction.text import  TfidfVectorizer
vectorizer=TfidfVectorizer(use_idf=True, max_features=150, ngram_range=(3, 3))

#convert the review column from the trainign data into a array of floats
x = vectorizer.fit_transform(train["review"]).toarray()
x_train = pd.DataFrame(x,columns=vectorizer.get_feature_names())
#Add the rating and effectiveness columns
x_train["rating"] = train["rating"]
x_train["effectiveness"] = train["effectiveness"]
    
x = vectorizer.fit_transform(test["review"]).toarray()
x_test = pd.DataFrame(x,columns=vectorizer.get_feature_names())
x_test["rating"] = test["rating"]
x_test["effectiveness"] = test["effectiveness"]

# removes invalid data that is such as NaN, Infinity, etc.
x_train = x_train.dropna()
x_test = x_test.dropna()

# Add the target columns to test and train
y_train = train["sideEffects"]
y_test = test["sideEffects"]

# Classifcation
from sklearn.metrics import classification_report
print("SVM")
from sklearn.svm import SVC
classifier = SVC(gamma='auto', C=1, class_weight='balanced')
classifier.fit(x_train, y_train)

#Prints out the classifcation report with the precision, recall, f-values and accuracy score
target_names = ['No Side Effects', 'Mild Side Effects', 'Moderate Side Effects', 'Severe Side Effects', 'Extremely Severe Side Effects']
y_pred = classifier.predict(x_test)
print(classification_report(y_test,y_pred, target_names=target_names))

print("Decision Tree")
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=6, random_state=1)
classifier.fit(x_train, y_train)

#Prints out the classifcation report with the precision, recall, f-values and accuracy score
y_pred = classifier.predict(x_test)
print(classification_report(y_test,y_pred, target_names=target_names))


#Uncomment if you wish to run the grid search
#Grid search to find the optimal parameters
# from sklearn.model_selection  import GridSearchCV
# param_grid = {'C':[1,10,100,1000], 'gamma':[1,0.1,0.001,0.0001,'auto'], 'kernel':['linear','rbf'],'class_weight' : ['balanced', None]}
# grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
# grid.fit(x_train,y_train)
# print(grid.best_params_)

# param_grid = {'max_depth': np.arange(3, 10), "random_state" :[0, 1, None]}
# grid = GridSearchCV(DecisionTreeClassifier(), param_grid)
# grid.fit(x_train,y_train)
# print(grid.best_params_)