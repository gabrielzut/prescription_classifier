from numpy.core.fromnumeric import mean
from sklearn_som.som import SOM
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

raw_df = read_csv('./dataSet.csv')

df = raw_df[['frequency',
             'dose']]
labels = raw_df[['target']]

np_dataset = df.values

X_train, X_test, y_train, y_test = train_test_split(
    np_dataset, labels, test_size=0.1)

dataset_som = SOM(m=2, n=1, dim=2)

dataset_som.fit(X_train)

predictions = dataset_som.predict(X_test)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 7))

x = X_test[:, 0]
y = X_test[:, 1]
colors = ['red', 'green', 'blue']

ax[0].scatter(x, y, c=predictions, cmap=ListedColormap(colors))
ax[0].title.set_text('SOM Predictions')
ax[1].scatter(x, y, c=y_test.to_numpy(), cmap=ListedColormap(colors))
ax[1].title.set_text('Actual Classes')
plt.savefig('som.png')

print("ACCURACY: ", balanced_accuracy_score(y_test.to_numpy(), predictions))
print("PRECISION: ", precision_score(
    y_test.to_numpy(), predictions, average='micro'))
print("RECALL: ", recall_score(y_test.to_numpy(), predictions, average='micro'))
print("F1 SCORE: ", f1_score(y_test.to_numpy(), predictions, average='micro'))
