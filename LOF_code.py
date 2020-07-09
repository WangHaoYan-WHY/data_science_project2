import numpy as np
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(10)

# get train data
inliers = 0.3 * np.random.randn(80, 2)
inliers = np.r_[inliers + 2, inliers - 2]
# get outliers
outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
store = np.r_[inliers, outliers]
outliers_num = len(outliers)
ground_truth = np.ones(len(store), dtype=int)
ground_truth[-outliers_num:] = -1
# get model for outlier detection
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# compute the predicted labels of the training samples
pred = clf.fit_predict(store)
scores = clf.negative_outlier_factor_
print(scores)
