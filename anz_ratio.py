import json
import numpy as np
from matplotlib import pyplot as plt

with open('ratio.json','r') as f:
    D = json.load(f)

size_to_ratio = {}
size_to_fix = {}
for each in D:
    size_to_ratio.setdefault(D[each]['size'],[]).extend(D[each]['ratio'])
    size_to_fix.setdefault(D[each]['size'],[]).extend(D[each]['lr_fix'])

size_to_mean_ratio = {}
for each in size_to_ratio:
    size_to_mean_ratio[int(each)] = np.mean(size_to_ratio[each])

size_to_mean_fix = {}
size_to_std_fix = {}
for each in size_to_ratio:
    size_to_mean_fix[int(each)] = np.mean(size_to_fix[each])
    size_to_std_fix[int(each)] = np.std(size_to_fix[each])



param_to_fix = {}
max_std = 0
min_std = 100000
max_std_idx = ''
min_std_idx = ''
for each in D:
    param_to_fix[each] = D[each]['lr_fix']
    currrent_max = np.max(param_to_fix[each])
    currrent_min = np.min(param_to_fix[each])

    if currrent_max > max_std:
        max_std = currrent_max
        max_std_idx = each

    if currrent_min < min_std:
        min_std = currrent_min
        min_std_idx = each

plt.scatter(size_to_mean_ratio.keys(),size_to_mean_ratio.values())
plt.title("ratio vs size")

plt.figure()
plt.scatter(size_to_mean_fix.keys(),size_to_mean_fix.values())
plt.title("mean fix vs size")

plt.figure()
plt.scatter(size_to_std_fix.keys(),size_to_std_fix.values())
plt.title("std fix vs size")

plt.figure()
plt.plot(D['56']['lr_fix'])
plt.title(" param 56 lr_fix")

plt.figure()
plt.plot(D['140']['lr_fix'])
plt.title(" param 140 lr_fix")


plt.figure()
plt.plot(D[max_std_idx]['lr_fix'])
plt.title(" max param {} lr_fix".format(max_std_idx))

plt.figure()
plt.plot(D[min_std_idx]['lr_fix'])
plt.title(" min param {} lr_fix".format(min_std_idx))

plt.show()
