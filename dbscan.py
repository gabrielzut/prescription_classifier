from sklearn.cluster import DBSCAN
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

raw_df = read_csv('./dataSet.csv').head(20000)

df = raw_df[['frequency',
             'dose']]
labels = raw_df[['target']]

np_dataset = df.values

X_train, X_test, y_train, y_test = train_test_split(np_dataset, labels, test_size=0.1, random_state=10)

# ro = RandomOverSampler()
# X_train, y_train = ro.fit_resample(X_train, y_train)
# ad = ADASYN()
# X_train, y_train = ad.fit_resample(X_train, y_train)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)

db = DBSCAN(min_samples=50, eps=100)
db.fit(X_train)
predictions = db.fit_predict(X_test)

predictions[predictions != -1] = 0
predictions[predictions == -1] = 1

print("ACCURACY: ", balanced_accuracy_score(y_test.to_numpy(), predictions))
print("PRECISION: ", precision_score(
    y_test.to_numpy(), predictions, average='weighted'))
print("RECALL: ", recall_score(y_test.to_numpy(), predictions, average='weighted', zero_division=1))
print("F1 SCORE: ", f1_score(y_test.to_numpy(), predictions, average='weighted'))
