import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def transform(x):
    if x == 'M':
        return 0
    elif x == 'B':
        return 1

iris = datasets.load_iris()

original = pd.read_csv("wdbc.csv")
original["Diagnosis"] = original["Diagnosis"].apply(lambda x: transform(x))
X = original.drop("Diagnosis", 1)
y = original["Diagnosis"] 

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


lda = LinearDiscriminantAnalysis(n_components=3)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1], ["M", "B"]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Seeds dataset')

print(len(X_r2))
print(len(X_r2[0]))

plt.figure()
for color, i, target_name in zip(colors, [0, 1], ["M", "B"]):
    print(X_r2[y == i, 0])
    print(X_r2[y == i, 1])
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Seeds dataset')

plt.show()









