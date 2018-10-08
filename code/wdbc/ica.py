import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
from numpy.testing import assert_array_almost_equal

df = pd.read_csv("wdbc.csv")
df = df.drop("Diagnosis", 1)
ica = FastICA(n_components=2)
ica.fit(df)

U, S, VT = np.linalg.svd(df - df.mean(0))
#assert_array_almost_equal(VT[:6], pca.components_)

X_train_ica = ica.transform(df)
#X_train_ica2 = (df - pca.mean_).dot(pca.components_.T)
#assert_array_almost_equal(X_train_pca, X_train_pca2)

X_projected = ica.inverse_transform(X_train_ica)
#X_projected2 = X_train_ica.dot(ica.components_) + pca.mean_
#assert_array_almost_equal(X_projected, X_projected2)

mse_loss = ((df - X_projected) ** 2).mean()
print(mse_loss)
sse_loss = np.sum((df-X_projected)**2)
print(sse_loss)
print(ica.components_)
# loadings
print(X_projected)
print(len(X_projected))
print(len(X_projected[0]))
print(type(X_projected))

#np.savetxt("wdbc_ica.csv", X_projected, delimiter=",")

# print(pca)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# print(len(pca.transform(df)))
# print(len(pca.transform(df)[0]))