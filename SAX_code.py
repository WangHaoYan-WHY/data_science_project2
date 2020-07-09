import numpy as np
import math

time_list = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.11, 1.22, 1.33, 1.44, 1.55, 1.66,
       1.77, 1.88, 1.99, 2.11, 2.22, 2.33, 2.44, 2.55, 2.66, 2.77, 2.88, 2.99, 3.00, 4.00, 5.00]  # list of time series
words_num = 6  # words's number
alpha = 3  # alpha number
char = ord('a')
points = {'0': [-0.40, 0.40],
               '1': [-0.60, 0, 0.60],
               '2': [-0.80, -0.20, 0.20, 0.80],
               '3': [-0.90, -0.40, 0, 0.40, 0.90],
               '4': [-1.09, -0.60, -0.20, 0.20, 0.60, 1.09],
               '5': [-1.20, -0.70, -0.30, 0, 0.30, 0.70, 1.20],
               }  # points
beta = points[str(alpha)]  # beta
# do normalization
s = np.asanyarray(time_list)
normalize = (s - np.nanmean(s)) / np.nanstd(s)
# do transfer
paa_list = []
n = len(normalize)
ceil_num = math.ceil(n / words_num)
for i in range(0, n, ceil_num):
    temp_ts = normalize[i:i + ceil_num]
    paa_list.append(np.mean(temp_ts))
    i = i + ceil_num
# do SAX transfer
transfer = paa_list
len_transfer = len(transfer)
len_beta = len(beta)
str = ''
for i in range(len_transfer):
    letter = False
    for j in range(len_beta):
        if np.isnan(transfer[i]):
            str += '-'
            letter = True
            break
        if transfer[i] < beta[j]:
            str += chr(char + j)
            letter = True
            break
    if not letter:
        str += chr(char + len_beta)
print(str)
