# -*- coding: utf-8 -*-
import os, sys, time
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import optimizers
from keras.layers import Input
from datetime import datetime

# 하이퍼파라미터
img_width, img_height = 32, 32
batch_trainsize = 32
batch_testsize = 32
learningrate = 1e-3
momentum = 0.8
num_classes = 10

# 학습 반복 범위
start_idx = 1
end_idx = 6  # model6까지 생성

# 데이터 로드
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# 반복 학습
for i in range(start_idx, end_idx):
    prev_model_path = './trained_models/resnet50model{}.h5'.format(i)
    next_model_path = './trained_models/resnet50model{}.h5'.format(i+1)

    print('\n[INFO] Loading {} → Training 1 epoch → Saving {}'.format(prev_model_path, next_model_path))
    
    if not os.path.isfile(prev_model_path):
        print('[ERROR] Model not found: {}'.format(prev_model_path))
        sys.exit(1)
    
    model = load_model(prev_model_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=learningrate, momentum=momentum),
                  metrics=['accuracy'])

    t = time.time()
    model.fit(X_train, y_train,
              batch_size=batch_trainsize,
              epochs=1,
              validation_data=(X_test, y_test),
              verbose=1)
    print('[INFO] 1 epoch training done in {:.2f} sec.'.format(time.time() - t))

    model.save(next_model_path)
    print('[INFO] Saved: {}'.format(next_model_path))
    del model
