# Quantization Module
I implemented quantization module by referring to several repositories.

Please let me know where i make mistake.

<br>

## List of Modules
1. [DoReFa](https://arxiv.org/abs/1606.06160)
2. [PACT](https://arxiv.org/abs/1805.06085)
3. [SQuantizer](https://arxiv.org/abs/1812.08301)

4. ...



```
git clone https://github.com/Kojungbeom/quantization.git
cd quantization
python train.py -net qresnet18 -wbit 4 -abit 4 -gpu
```





## Usage 

```bash
usage: train.py [-net network] [-b batch_size] [-wbit bit_precision]
                [-abit bit_precision] [-sigma sigma] [-delay delay]
              
optional arguments:
  -net			Name of network [resnet18, resnet34, resnet50,
  								 qresnet18, qresnet34, qresnet50,
  								 sqresnet18, sqresnet34, sqresnet50,
  								 Nos_qresnet18, Nos_qresnet34, Nos_qresnet50]
  -b 			batch size, Default: 256
  -wbit			bit-precision of weight
  -abit			bit-precision of activation
  -sigma		Sparsity controlling factor in squantizer
  -delay		Delay epoch to let weights stabilize at the start of training
```

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





## Reference

[1] https://github.com/zzzxxxttt/pytorch_DoReFaNet

[2] https://github.com/cornell-zhang/dnn-gating

