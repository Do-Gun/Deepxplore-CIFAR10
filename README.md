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
