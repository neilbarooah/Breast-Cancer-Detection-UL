import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

def transform(x):
	if x == 'M':
		return 0
	elif x == 'B':
		return 1


original = pd.read_csv("wdbc.csv")
original["Diagnosis"] = original["Diagnosis"].apply(lambda x: transform(x))
X = original.drop("Diagnosis", 1)
y = original["Diagnosis"] 
#fig = plt.figure(1, figsize=(4,3))
#plt.clf()
#ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

#plt.cla()



## USEFUL
# pca= PCA(n_components = 2)
# pca.fit(X)
# X = pca.transform(X)

# print('explained variance ratio: %s' % str(pca.explained_variance_ratio_))

# plt.figure()
# colors = ['navy', 'turquoise']
# lw = 2
# for color, i , target_name in zip(colors, [0, 1], ['M', 'B']):
# 	plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)

# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title("PCA of WDBC (n = 2")
# plt.show()


## USEFUL
# for name, label in [('M', 0), ('B', 1)]:
# 	ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 0]).astype(np.float)
# ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spectral,
#            edgecolor='k')

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

# plt.show()
# pca = PCA(n_components=3)
# pca.fit(df)

# U, S, VT = np.linalg.svd(df - df.mean(0))
# #assert_array_almost_equal(VT[:6], pca.components_)

# X_train_pca = pca.transform(df)
# X_train_pca2 = (df - pca.mean_).dot(pca.components_.T)
# #assert_array_almost_equal(X_train_pca, X_train_pca2)

# X_projected = pca.inverse_transform(X_train_pca)
# X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_
# #assert_array_almost_equal(X_projected, X_projected2)

# loss = ((df - X_projected) ** 2).mean()
# print(loss)
# sse_loss = np.sum((df-X_projected)**2)
# print(sse_loss)
# print(pca.components_)
# print(pca.explained_variance_ratio_)
# # loadings
# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# print(loadings)
# print(X_projected)
# print(len(X_projected))
# print(len(X_projected[0]))

# # We center the data and compute the sample covariance matrix.
# X_centered = df - np.mean(df, axis=0)
# cov_matrix = np.dot(X_centered.T, X_centered) / 569
# eigenvalues = pca.explained_variance_
# for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
#     print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
#     print(eigenvalue)

#np.savetxt("wdbc_ica.csv", X_projected, delimiter=",")


# print(pca)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# print(len(pca.transform(df)))
# print(len(pca.transform(df)[0]))


# import numpy as np
# from sklearn.decomposition import PCA
# import pandas as pd
# from numpy.testing import assert_array_almost_equal

# original = pd.read_csv("wdbc.csv")
# df = original.drop("Diagnosis", 1)
# pca = PCA(n_components=6)
# pca.fit(df)

# U, S, VT = np.linalg.svd(df - df.mean(0))
# #assert_array_almost_equal(VT[:6], pca.components_)

# X_train_pca = pca.transform(df)
# X_train_pca2 = (df - pca.mean_).dot(pca.components_.T)
# #assert_array_almost_equal(X_train_pca, X_train_pca2)

# X_projected = pca.inverse_transform(X_train_pca)
# X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_
# #assert_array_almost_equal(X_projected, X_projected2)

# loss = ((df - X_projected) ** 2).mean()
# print(loss)
# sse_loss = np.sum((df-X_projected)**2)
# print(sse_loss)
# print(pca.components_)
# print(pca.explained_variance_ratio_)
# # loadings
# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# print(loadings)
# print(X_projected)
# print(len(X_projected))
# print(len(X_projected[0]))

# # We center the data and compute the sample covariance matrix.
# X_centered = df - np.mean(df, axis=0)
# cov_matrix = np.dot(X_centered.T, X_centered) / 569
# eigenvalues = pca.explained_variance_
# for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
#     print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
#     print(eigenvalue)

#np.savetxt("wdbc_ica.csv", X_projected, delimiter=",")


# print(pca)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# print(len(pca.transform(df)))
# print(len(pca.transform(df)[0]))