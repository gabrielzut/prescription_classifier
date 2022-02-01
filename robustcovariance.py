from sklearn.covariance import EllipticEnvelope
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt

raw_df = read_csv('./dataSet.csv')

df = raw_df[['frequency',
             'dose']]
labels = raw_df[['target']]

np_dataset = df.values

X_train, X_test, y_train, y_test = train_test_split(
    np_dataset, labels, test_size=0.1, random_state=10)

# ro = RandomOverSampler()
# X_train, y_train = ro.fit_resample(X_train, y_train)
# ad = ADASYN()
# X_train, y_train = ad.fit_resample(X_train, y_train)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)

ee = EllipticEnvelope()
model = ee.fit(X_train, y_train)
predictions = model.predict(X_test)
y_score = model.decision_function(X_test)

predictions[predictions == 1] = 0
predictions[predictions == -1] = 1

print("ACCURACY: ", balanced_accuracy_score(y_test.to_numpy(), predictions))
print("PRECISION: ", precision_score(
    y_test.to_numpy(), predictions, average='weighted'))
print("RECALL: ", recall_score(y_test.to_numpy(),
      predictions, average='weighted', zero_division=1))
print("F1 SCORE: ", f1_score(y_test.to_numpy(), predictions, average='weighted'))

bc = BinaryClassification(y_test, y_score, labels=["Class 1", "Class 2"])

plt.figure(figsize=(5, 5))
bc.plot_roc_curve()
plt.show()