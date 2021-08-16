import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


store = pd.HDFStore('../datasets/shapenet_cropped.h5', 'r')
lens = [store[k].shape[0] for k in store.keys()]
store.close()

density = plt.hist(lens, bins='auto', density=True)[0]

search_space = np.floor(np.linspace(np.min(lens), np.max(lens), len(density)))
len_values = np.array([ln * np.sum(np.array(lens) >= ln) for ln in search_space])
len_values /= np.max(len_values)
len_values *= np.max(density)

print(int(search_space[np.argmax(len_values)]))

plt.plot(search_space, len_values)
plt.show()
