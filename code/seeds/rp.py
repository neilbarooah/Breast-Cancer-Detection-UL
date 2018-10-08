import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
import pandas as pd
from numpy.testing import assert_array_almost_equal
from sklearn import utils

original = pd.read_csv("seeds.csv")
df = original.drop("Class", 1)

mse_df = pd.DataFrame(columns=["Area", "Perimeter", "Compactness", "Length", "Width", "Asymmetry Coefficient", "Length of Kernel Groove"])
transformed_df = pd.DataFrame(columns=["Area", "Perimeter", "Compactness", "Length", "Width", "Asymmetry Coefficient", "Length of Kernel Groove"])

for i in range(500):
	rp = GaussianRandomProjection(n_components=7)
	rp.fit(df)
	transformed_instances = rp.transform(df)
	inverse_components = np.linalg.pinv(rp.components_)
	reconstructed_instances = utils.extmath.safe_sparse_dot(transformed_instances, inverse_components.T)
	mse = ((df - reconstructed_instances) ** 2).mean()
	mse_df.loc[i] = mse
	new_df = pd.DataFrame(reconstructed_instances, columns=["Area", "Perimeter", "Compactness", "Length", "Width", "Asymmetry Coefficient", "Length of Kernel Groove"])
	transformed_df = transformed_df.append(new_df)


print(mse_df)
print(mse_df.mean())
print(transformed_df.var())