from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import statistics

def process_data(file_name):
    with open(str(file_name)) as f:
        file = f.readlines()
        label = file[0].strip().split(",")
        label[0] = label[0][3:]
        file = file[1:]
        row = len(file)
        col = len(file[0].strip().split(","))-1 #59 attributes and 1 label
        mat = np.zeros((row, col))
        for i in range(len(file)):
            row = file[i].strip()
            cols = row.split(",")[1:]
            for j in range(len(cols)):
                mat[i][j] = int(cols[j])
    np.random.seed(10)
    np.random.shuffle(mat)
    X = mat[:, 0:-1]
    y = mat[:, -1]

    return X, y, label

X, y, label = process_data("sports_articles.csv")
X = X[0:10, :]
y = y[0:10]
dtc = DecisionTreeClassifier()
score = cross_val_score(dtc, X, y, cv=5)
avg_score = statistics.mean(score)
print(1-avg_score)