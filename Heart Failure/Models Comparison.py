import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib


def train_test_clf(clf, X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(accuracy_score(real, predicts))
    print(confusion_matrix(real, predicts))
    print(classification_report(real, predicts))


df = pd.read_csv('patientdataV6.csv')


hf_encoder = LabelEncoder()
sex_encoder = LabelEncoder()
family_encoder = LabelEncoder()
smoker_encoder = LabelEncoder()


df.HEARTFAILURE = hf_encoder.fit_transform(df.HEARTFAILURE)
df.SEX = sex_encoder.fit_transform(df.SEX)
df.FAMILYHISTORY = family_encoder.fit_transform(df.FAMILYHISTORY)
df.SMOKERLAST5YRS = smoker_encoder.fit_transform(df.SMOKERLAST5YRS)


joblib.dump(hf_encoder, 'hf_encoder.pkl')
joblib.dump(sex_encoder, 'sex_encoder.pkl')
joblib.dump(family_encoder, 'family_encoder.pkl')
joblib.dump(smoker_encoder, 'smoker_encoder.pkl')


X = df.drop(['HEARTFAILURE'], axis=1).values
y = df.HEARTFAILURE.values


print(train_test_clf(DecisionTreeClassifier(), X, y))


print(train_test_clf(RandomForestClassifier(), X, y))


print(train_test_clf(KNeighborsClassifier(), X, y))


print(train_test_clf(SVC(), X, y))


print(train_test_clf(GaussianNB(), X, y))


print(train_test_clf(MLPClassifier(), X, y))


print(train_test_clf(LogisticRegression(), X, y))


clf = RandomForestClassifier()
clf.fit(X, y)
joblib.dump(clf, 'rf.pkl')