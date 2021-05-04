# Quantization Aware Training Modules

[1]을 참고하여 Cifar-100 데이터셋을 위한 ResNet18,34,50에 QAT 모듈을 적용함.

```
git clone https://github.com/Kojungbeom/quantization.git
cd quantization
```

<br>

## Requirements

- Python

- CUDA 11.1
- [Pytorch](https://pytorch.org/get-started/previous-versions/) 1.7.1+cu110
- Torchvision 0.8.2+cu110
- Tensorboard 2.4.1
- Torchsummary 1.5.1

<br>

## Usage

### 1. List of Modules

1. [DoReFa](https://arxiv.org/abs/1606.06160)
   - quant_utils/quant_dorefa.py
2. [PACT](https://arxiv.org/abs/1805.06085)
   - quant_utils/quant_pact.py
3. [SQuantizer](https://arxiv.org/abs/1812.08301)
   - quant_utils/quant_sparse.py

<br>

### 2. List of Models

- **qresnet**: ResNet with DoReFa-Net + PACT
- **sqresnet**: ResNet with Squantizer + PACT
- **Nos_qresnet**: ResNet with Squantizer(without Sparsification) + PACT

<br>

### 3. How to Train

```bash
usage: train.py [-net network] [-b batch_size] [-wbit bit_precision]
                [-abit bit_precision] [-sigma sigma] [-delay delay]
              
optional arguments:
  -net			Name of network [resnet18, resnet34, resnet50,
  								 qresnet18, qresnet34, qresnet50,
  								 sqresnet18, sqresnet34, sqresnet50,
  								 Nos_qresnet18, Nos_qresnet34, Nos_qresnet50]
  -b 			batch size, Default: 256
  -wbit			bit-precision of weight, Default: 8
  -abit			bit-precision of activation, Default: 8
  -sigma		Sparsity controlling factor in squantizer, Default: 0.0
  -delay		Delay epoch to let weights stabilize at the start of training.
                Commonly epoch / 3, Default: 70 
```

- EXAMPLE

```bash
# Train qresnet50, wbit 8, abit 8
python train.py -net qresnet50 -wbit 8 -abit 8 -gpu

# Train sqresnet50, wbit 8, abit 8, sigma 0, delay 70
python train.py -net sqresnet50 -wbit 8 -abit 8 -sigma 0.0 -delay 70 -gpu

# Train Nos_qresnet50, wbit 8, abit 8, delay 70
python train.py -net Nos_qresnet50 -wbit 8 -abit 8 -delay 70 -gpu
```

<br>

### 4. How to Test

```bash
usage: test.py [-net network] [-weight weight_file] [-b batch_size] 
               [-wbit bit_precision] [-abit bit_precision] [-sigma sigma]
              
optional arguments:
  -net			Name of network [resnet18, resnet34, resnet50,
  								 qresnet18, qresnet34, qresnet50,
  								 sqresnet18, sqresnet34, sqresnet50,
  								 Nos_qresnet18, Nos_qresnet34, Nos_qresnet50]\
  -weight 		Path to the weight file					
  -b 			Batch size, Default: 256
  -wbit			Bit-precision of weight
  -abit			Bit-precision of activation
  -sigma		Sparsity controlling factor in squantizer
```

- EXAMPLE

```bash
# test qresnet50, wbit 4, abit 4, the weight file is in qresnet50/weight.pth
python test.py -net qresnet50 -wbit 4 -abit 4 -gpu -weight checkpoint/qresnet18/weight.pth

# test sqresnet50, wbit 4, abit 4, sigma 0.0, the weight file is in sqresnet50/weight.pth
python test.py -net sqresnet50 -wbit 4 -abit 4 -sigma 0.0 -gpu -weight checkpoint/sqresnet18/weight.pth

# test Nos_qresnet50, wbit 4, abit 4, the weight file is in Nos_qresnet50/weight.pth
python test.py -net Nos_qresnet50 -wbit 4 -abit 4 -gpu -weight checkpoint/sqresnet18/weight.pth
```

- Tensorboard

```
tensorboard --logdir runs/"name of network"
```



## Reference

[1] https://github.com/weiaicunzai/pytorch-cifar100

[2] https://github.com/cornell-zhang/dnn-gating

[3] https://github.com/zzzxxxttt/pytorch_DoReFaNet

[4] https://arxiv.org/abs/1606.06160

[5] https://arxiv.org/abs/1805.06085

[6] https://arxiv.org/abs/1812.08301

