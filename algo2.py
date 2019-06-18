import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv("TrainedCleaned.csv")

target = train["Survived"].values
feature_names = ["Pclass", "Age", "Sex", "Parch", "Fare", "SibSp"]
features = train[feature_names].values

classifier = LogisticRegression()
classifier_ = classifier.fit(features, target)

print(classifier_.score(features,target))

poly = PolynomialFeatures()
poly_features = poly.fit_transform(features, target)

classifier_ = classifier.fit(poly_features, target)

print(classifier_.score(poly_features,target))