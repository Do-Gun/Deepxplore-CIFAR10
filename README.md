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


## 2. 모델 준비
초기 모델은 https://github.com/kusiwu/Resnet50-Cifar10-Python-Keras epoch 20 가중치 사용
이후 아래 스크립트로 1 epoch씩 fine-tuning 하여 실험에 사용할  5개 모델 생성

trained_models 파일 내부
0) resnet50model1.h5 (epoch: 20, 실험에 사용하지 않는 초기 모델 가중치)
1) resnet50model2.h5 (epoch: 20 + 1)
2) resnet50model3.h5
3) resnet50model4.h5
4) resnet50model5.h5
5) resnet50model6.h5




python finetuning.py
