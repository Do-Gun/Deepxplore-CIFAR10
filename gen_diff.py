# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse, os, numpy as np, random, sys
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
from scipy.misc import imsave
from utils import *
from configs import bcolors

parser = argparse.ArgumentParser(description='Difference-inducing input generation with 4 models')
parser.add_argument('transformation', choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', type=float)
parser.add_argument('weight_nc', type=float)
parser.add_argument('step', type=float)
parser.add_argument('seeds', type=int)
parser.add_argument('grad_iterations', type=int)
parser.add_argument('threshold', type=float)
parser.add_argument('-t', '--target_model', choices=[0, 1, 2, 3], default=0, type=int)
parser.add_argument('-sp', '--start_point', default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', default=(10, 10), type=tuple)
args = parser.parse_args()

np.random.seed(1)
random.seed(1)
input_shape = (32, 32, 3)

models = [load_model('./trained_models/resnet50model%d.h5' % i) for i in range(2, 7)]
model_layer_dicts = [init_coverage_table(model) for model in models]

img_folder = './test/center/'
img_paths = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png') or f.endswith('.jpg')])
if not img_paths:
    raise Exception("테스트할 이미지가 없습니다.")
print("총 테스트 이미지 수: {}".format(len(img_paths)))

if not os.path.exists('./generated_inputs'):
    os.makedirs('./generated_inputs')

for i in range(args.seeds):
    img_path = random.choice(img_paths)
    print("[%d/%d] 현재 이미지: %s" % (i+1, args.seeds, os.path.basename(img_path)))

    gen_img = preprocess_image(img_path)
    preds = [model.predict(gen_img) for model in models]
    labels = [np.argmax(pred[0]) for pred in preds]

    if len(set(labels)) > 1:
        print("다른 예측 발생: %s" % str(labels))
        for m, d in zip(models, model_layer_dicts):
            update_coverage(gen_img, m, d, args.threshold)
        filename = './generated_inputs/already_differ_%s.png' % '_'.join([str(l) for l in labels])
        imsave(filename, deprocess_image(gen_img))
        continue
    else:
        print("공격 시작 - 모든 모델 동일 예측: %s" % str(labels))

    orig_label = labels[0]
    losses = []
    neuron_losses = []

    for idx, (model, layer_dict) in enumerate(zip(models, model_layer_dicts)):
        output_loss = K.mean(model.get_layer('output_layer').output[..., orig_label])
        if idx == args.target_model:
            losses.append(-args.weight_diff * output_loss)
        else:
            losses.append(output_loss)

        layer_name, index = neuron_to_cover(layer_dict)
        neuron_loss = K.mean(model.get_layer(layer_name).output[..., index])
        neuron_losses.append(neuron_loss)

    final_loss = K.mean(sum(losses) + args.weight_nc * sum(neuron_losses))
    grads = normalize(K.gradients(final_loss, models[0].input)[0])
    iterate = K.function([models[0].input] + [m.input for m in models[1:]] + [K.learning_phase()],
                         losses + neuron_losses + [grads])

    for iteration in range(args.grad_iterations):
        percent = int((iteration + 1) * 100.0 / args.grad_iterations)
        sys.stdout.write("\rGradient Ascent 진행 중: %3d%% (%d/%d)" % (percent, iteration + 1, args.grad_iterations))
        sys.stdout.flush()

        inputs = [gen_img] * len(models) + [0]
        result = iterate(inputs)
        grads_value = result[-1][0]

        if args.transformation == 'light':
            grads_value = np.clip(grads_value, -0.1, 0.1)
        elif args.transformation == 'occl':
            mask = np.zeros_like(grads_value)
            x0, y0 = args.start_point
            dx, dy = args.occlusion_size
            mask[:, x0:x0+dx, y0:y0+dy, :] = 1
            grads_value *= mask
        elif args.transformation == 'blackout':
            grads_value = np.zeros_like(grads_value)

        gen_img += grads_value * args.step
        new_preds = [m.predict(gen_img) for m in models]
        new_labels = [np.argmax(p[0]) for p in new_preds]

        if len(set(new_labels)) > 1:
            print("\n공격 성공 - 모델 예측 결과: %s" % str(new_labels))
            for m, d in zip(models, model_layer_dicts):
                update_coverage(gen_img, m, d, args.threshold)
            filename = './generated_inputs/%s_%s.png' % (args.transformation, '_'.join([str(l) for l in new_labels]))
            imsave(filename, deprocess_image(gen_img))
            break
    else:
        print("\n공격 실패 - 모든 모델 동일 예측 유지됨: %s" % str(new_labels))
