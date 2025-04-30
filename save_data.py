# -*- coding: utf-8 -*-

from keras.datasets import cifar10
from keras.preprocessing.image import array_to_img
import os

save_dir = './test/center/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

(_, _), (x_test, _) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.

for i in range(30):  # 300장만 저장
    img = array_to_img(x_test[i])
    filename = 'test_img_%03d.png' % i  #  Python 2.7 스타일 포맷팅
    img.save(os.path.join(save_dir, filename))

print(" CIFAR-10 테스트 이미지 30장 저장 완료.")
