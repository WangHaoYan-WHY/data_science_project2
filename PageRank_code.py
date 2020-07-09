import numpy as np

data_matrix = np.array([[0, 1 / 3, 0, 0, 0],
                   [0.5, 0, 0, 0, 0],
                   [0, 1 / 3, 0, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0.5, 1 / 3, 1, 0, 0]])  # data matrix represents edge weights
probability = 0.86
threshold = 0.000000001
data_size = 5
Page_jump = np.array([1 / data_size] * (data_size ** 2)).reshape(data_size, data_size)
Page_rank = np.array([1 / data_size] * data_size).reshape(data_size, 1)  # page rank
data_matrix = probability * data_matrix + (1 - probability) * Page_jump  # data matrix
while True:
    rank_new = np.dot(data_matrix, Page_rank)
    dist = np.linalg.norm(Page_rank - rank_new)  # calculate the distance
    Page_rank = rank_new
    if dist < threshold:  # break when smaller than threshold
        break
print(Page_rank)
