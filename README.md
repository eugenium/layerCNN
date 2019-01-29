# Greedy Layerwise CNN

This is a peliminary research code.

Code for experiments on greedy supervised layerwise CNNs

## Imagenet
Imagenet experiments for 1-hidden layer use the standalone imagenet_single_layer.py

Imagenet experiments for k=2+ can be run with imagenet.py

Note k in the paper corresponds to nlin in the code


To obtain the results for Imagenet

k=3 
```
python IMAGENER_DIR -j THREADS  imagenet.py --ncnn 8 --nlin 2 

```

k=2 

```
python IMAGENER_DIR -j THREADS imagenet.py --ncnn 8 --nlin 1 

```

k=1 model
```
python IMAGENER_DIR -j THREADS  imagenet_single_layer.py --ncnn 8

```
### VGG-11

The VGG-11 model was trained with a new refactored and more modular codebase different from the codebase used for the above models and is thus run from the standalone directory 
refactored_imagenet/

To train the VGG-11 with k=3

```
python imagenet_greedy.py IMAGENER_DIR -j THREADS --arch vgg11_bn --half --dynamic-loss-scale

```
to train the baseline:

```
python imagenet.py IMAGENER_DIR -j THREADS --arch vgg11_bn --half --dynamic-loss-scale

```

### Linear Separability
Linear separability experiments are in linear_separability folder. A notebook is included that produces the plots. to run different settings


This will create and train a model, using K non-linearity, F features and the model is stored in checkpoint.

```
python cifar.py --ncnn 5 --nlin K --feature_size F
```

This will use the model "filename", to train probes on top of these at layer "j"
```
python train_lr.py filename j

```

### CIFAR experiments
CIFAR experiments can be reproduced using cifar.py

The CIFAR-10 models can be trained:

k=3 (~91.7) 
```
python cifar.py --ncnn 4 --nlin 2 --feature_size 128 --down [1] --bn 1

```

k=2 (~90.4)

```
python cifar.py --ncnn 4 --nlin 1 --feature_size 128 --down [1] --bn 1

```

k=1 (~88.3) 
```
python cifar.py --ncnn 5 --nlin 0 --feature_size 256 

```

