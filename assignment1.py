import csv
from sklearn.model_selection import train_test_split

labels = ["urlDrugName", "rating", "effectiveness", "sideEffects", "condition",	"benefitsReview", "sideEffectsReview", "commentsReview"]
train = []
with open('drugLibTrain_raw.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        value = [row[labels[0]], row[labels[1]], row[labels[2]], row[labels[3]], row[labels[4]], row[labels[5]], row[labels[6]], row[labels[7]]]
        train.append(value)
test = []
with open('drugLibTest_raw.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        value = [row[labels[0]], row[labels[1]], row[labels[2]], row[labels[3]], row[labels[4]], row[labels[5]], row[labels[6]], row[labels[7]]]
        test.append(value)

print(len(train))
print(train[len(train)-1])

y = range(0, len(train))
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

'''Now predict on the testing data'''
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))