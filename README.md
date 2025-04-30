# DeepXplore with CIFAR-10 (ResNet50)

This project demonstrates DeepXplore's differential testing on CIFAR-10 using multiple ResNet50 models fine-tuned with slight variations.

---

## 1. 환경 설정
```bash
conda create -n deepxplore python=2.7 -y
conda activate deepxplore
pip install tensorflow==1.3.0 keras==2.0.8 Pillow==6.2.2 h5py==2.10.0 opencv-python==3.4.2.17


git clone https://github.com/Do-Gun/Deepxplore-CIFAR10.git
cd Deepxplore-CIFAR10
```


## 2. 모델 준비 (이미 준비 해놨기에, Testing 과정에서는 넘어감)
초기 모델은 https://github.com/kusiwu/Resnet50-Cifar10-Python-Keras epoch 20 가중치 사용

이후 아래 스크립트로 1 epoch씩 fine-tuning 하여 실험에 사용할 5개 모델 생성됨

trained_models 파일 내부

0) resnet50model1.h5 (epoch: 20, 실험에 사용하지 않는 초기 모델 가중치)
1) resnet50model2.h5 (epoch: 20 + 1, 실험에 사용)
2) resnet50model3.h5 (epoch: 20 + 2, 실험에 사용)
3) resnet50model4.h5 (epoch: 20 + 3, 실험에 사용)
4) resnet50model5.h5 (epoch: 20 + 4, 실험에 사용) 
5) resnet50model6.h5 (epoch: 20 + 5, 실험에 사용)
**→ 총 5개의 ResNet50 사용**

새로운 가중치 만들고 싶은 경우만 (Testing 과정에서는 필요 없음)
```bash
python python finetuning.py
```

[INFO] Loading ./trained_models/resnet50model1.h5 → Training 1 epoch → Saving ./trained_models/resnet50model2.h5
Train on 50000 samples, validate on 10000 samples
Epoch 1/1
50000/50000 [==============================] - 259s - loss: 0.3801 - acc: 0.8702 **- val_loss: 0.6599 - val_acc: 0.7982**
[INFO] 1 epoch training done in 263.26 sec.
[INFO] Saved: ./trained_models/resnet50model2.h5

[INFO] Loading ./trained_models/resnet50model2.h5 → Training 1 epoch → Saving ./trained_models/resnet50model3.h5
Train on 50000 samples, validate on 10000 samples
Epoch 1/1
50000/50000 [==============================] - 257s - loss: 0.3028 - acc: 0.8968 **- val_loss: 0.6997 - val_acc: 0.7986**
[INFO] 1 epoch training done in 261.11 sec.
[INFO] Saved: ./trained_models/resnet50model3.h5

[INFO] Loading ./trained_models/resnet50model3.h5 → Training 1 epoch → Saving ./trained_models/resnet50model4.h5
Train on 50000 samples, validate on 10000 samples
Epoch 1/1
50000/50000 [==============================] - 255s - loss: 0.2322 - acc: 0.9218 **- val_loss: 0.7134 - val_acc: 0.7994**
[INFO] 1 epoch training done in 259.07 sec.
[INFO] Saved: ./trained_models/resnet50model4.h5

[INFO] Loading ./trained_models/resnet50model4.h5 → Training 1 epoch → Saving ./trained_models/resnet50model5.h5
Train on 50000 samples, validate on 10000 samples
Epoch 1/1
50000/50000 [==============================] - 260s - loss: 0.1821 - acc: 0.9362 **- val_loss: 0.7458 - val_acc: 0.8034**
[INFO] 1 epoch training done in 264.48 sec.
[INFO] Saved: ./trained_models/resnet50model5.h5

[INFO] Loading ./trained_models/resnet50model5.h5 → Training 1 epoch → Saving ./trained_models/resnet50model6.h5
Train on 50000 samples, validate on 10000 samples
Epoch 1/1
50000/50000 [==============================] - 257s - loss: 0.1387 - acc: 0.9541 **- val_loss: 0.7932 - val_acc: 0.8026**
[INFO] 1 epoch training done in 261.72 sec.
[INFO] Saved: ./trained_models/resnet50model6.h5

## 3. Test Data (이미 준비해놨기에, Testing 과정에서는 넘어감)
CIFAR-10에서 30개의 테스트 데이터를 추출 (test/center 파일에 이미 저장해놓음)
```bash
python save_data.py
```


## 4. 실행
```bash
KERAS_BACKEND=tensorflow python gen_diff.py light 3 0.5 0.1 10 500 0.2 -t 0
```
![image](https://github.com/user-attachments/assets/95054c8d-c08f-480e-8996-f9c20032cf1c)


## 5. 실험 결과
![image](https://github.com/user-attachments/assets/8b8d8320-c866-4990-bae6-0c59bf234320)
![image](https://github.com/user-attachments/assets/102c0101-e2db-4905-8641-e6a8714c0da2)

1) 각 모델의 초기 예측부터 “동일하지 않은 경우” 존재 (3, 7, 8)

→ 공격을 시도하지 않아도 서로 다른 예측이 발생함

**→ 모델 간 weight가 실험을 위해 충분히 다름을 의미, DeepXplore 세팅 유효성을 정성적으로 확인**

2) 각 모델의 초기 예측이 “동일한 경우”에만 DeepXplore 적용
3) 
**→ Target model(model2.h5, index0)가 다른 예측 결과를 내보냄 (공격 성공) (1, 6, 9, 10)**

→ 공격 실패 경우도 존재함 (2)

→ 공격 시 Target model이 아닌 모델이 다른 예측을 보이는 케이스도 존재했음 (4, 5)













