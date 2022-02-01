from sklearn.svm import OneClassSVM
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

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

db = OneClassSVM(gamma='auto')
model = db.fit(X_train, y_train)
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

# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# fpr[1], tpr[1], _ = roc_curve(y_test.to_numpy()[:, 0], y_score)
# roc_auc[1] = auc(fpr[1], tpr[1])

# fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score)
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plt.figure()
# lw = 2
# plt.plot(fpr[1], tpr[1], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Taxa de falsos positivos')
# plt.ylabel('Taxa de verdadeiros positivos')
# plt.title('Curva ROC - OneClassSVM - SMOTE')
# plt.legend(loc="lower right")
# plt.show()

RocCurveDisplay.from_predictions(y_test, predictions)
