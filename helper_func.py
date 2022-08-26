import numpy as np

def load_matlab_struct(struct):
    data = struct.popitem()[-1]
    labels = data.dtype.descr
    n_labels = len(labels)
    return {labels[i][0] : data[0][0][i] for i in range(n_labels)}

def filter_by_index(data, keys, operators, values):
    filters = np.ones_like(data[keys[0]], dtype=bool)
    for i, k in enumerate(keys):
        filters *= operators[i](data[k], values[i])
    return filters