import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
test = pd.read_csv('drugLibTest_raw.tsv',delimiter='\t')     # Read the files with the pandas dataFrame
train = pd.read_csv('drugLibTrain_raw.tsv', delimiter='\t') 
data = pd.concat([test,train])
test.shape
train.shape
data.shape
data.columns = ['Id','urlDrugName ','rating','effectiveness','sideEffects','condition','benefitsReview' , 'sideEffectsReview', 'commentsReview']
print(len(data.values))

y = data.drop('Id', axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=0, shuffle=False)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))