# -*- coding: utf-8 -*-
import numpy as np
import random
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model

def preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')  
    return x

def deprocess_image(x):
    x = x.reshape((32, 32, 3))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def constraint_light(grads):
    return np.clip(grads, -0.1, 0.1)

def constraint_occl(grads, start_point=(0, 0), occlusion_size=(10, 10)):
    mask = np.zeros_like(grads)
    mask[:, start_point[0]:start_point[0]+occlusion_size[0], start_point[1]:start_point[1]+occlusion_size[1], :] = 1
    return grads * mask

def constraint_black(grads):
    return np.zeros_like(grads)

def init_coverage_table(model):
    model_layer_dict = {}
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False
    return model_layer_dict

def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        return random.choice(not_covered)
    return random.choice(list(model_layer_dict.keys()))

def neuron_covered(model_layer_dict):
    covered = sum(model_layer_dict.values())
    total = len(model_layer_dict)
    return covered, total, float(covered) / total

def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if 'flatten' not in layer.name and 'input' not in layer.name]
    intermediate_model = Model(inputs=model.input, outputs=[model.get_layer(name).output for name in layer_names])
    intermediate_outputs = intermediate_model.predict(input_data)

    for i, output in enumerate(intermediate_outputs):
        scaled = (output[0] - np.min(output[0])) / (np.max(output[0]) - np.min(output[0]) + 1e-5)
        for neuron_idx in range(scaled.shape[-1]):
            if np.mean(scaled[..., neuron_idx]) > threshold:
                model_layer_dict[(layer_names[i], neuron_idx)] = True
