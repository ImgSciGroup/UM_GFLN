import numpy as np
def preprocess_img(data, d_type, norm_type):
    g_data = np.array(data).astype(np.float32)
    if d_type == 'opt':
        if norm_type == 'stad':
            g_data = stad_data(g_data)
        else:
            g_data = norm_data(g_data)
    elif d_type == 'sar':
        g_data[np.abs(g_data) <= 0] = np.min(g_data[np.abs(g_data) > 0])
        g_data = np.log(g_data + 1.0)
        if norm_type == 'stad':
            g_data = stad_data(g_data)
        else:
            g_data = norm_data(g_data)
    return g_data
def norm_data(data):
        data_height, data_width, channel = data.shape
        data = np.reshape(data, (data_height * data_width, channel))  # (channel, height * width)
        max = np.max(data, axis=0, keepdims=True)  # (channel, 1)
        min = np.min(data, axis=0, keepdims=True)  # (channel, 1)
        diff_value = max - min
        nm_data = (data - min) / diff_value
        nm_data = np.reshape(nm_data, (data_height, data_width, channel))
        return nm_data
def stad_data(data):
        data_height, data_width, channel = data.shape
        data = np.reshape(data, (data_height * data_width, channel))  # (height * width, channel)
        mean = np.mean(data, axis=0, keepdims=True)  # (1, channel)
        center = data - mean  # (height * width, channel)
        var = np.sum(np.power(center, 2), axis=0, keepdims=True) / (data_height * data_width)  # (1, channel)
        std = np.sqrt(var)  # (channel, 1)
        nm_data = center / std  # (channel, height * width)
        nm_data = np.reshape(nm_data, (data_height, data_width, channel))
        return nm_data
