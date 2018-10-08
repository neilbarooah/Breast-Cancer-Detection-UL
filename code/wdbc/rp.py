import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
import pandas as pd
from numpy.testing import assert_array_almost_equal
from sklearn import utils

original = pd.read_csv("wdbc.csv")
df = original.drop("Diagnosis", 1)

mse_df = pd.DataFrame(columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"])
transformed_df = pd.DataFrame(columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"])

for i in range(500):
	rp = GaussianRandomProjection(n_components=28)
	rp.fit(df)
	transformed_instances = rp.transform(df)
	inverse_components = np.linalg.pinv(rp.components_)
	reconstructed_instances = utils.extmath.safe_sparse_dot(transformed_instances, inverse_components.T)
	print(reconstructed_instances)
	break
	#mse = ((df - reconstructed_instances) ** 2).mean()
	#mse_df.loc[i] = mse
	new_df = pd.DataFrame(reconstructed_instances, columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"])
	transformed_df = transformed_df.append(new_df)


#print(mse_df)
#print(mse_df.mean())
print(transformed_df.var())