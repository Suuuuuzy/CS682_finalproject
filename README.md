# CS682_finalproject

## Semi-supervised Image classification on TinyImageNet with pretext tasks

We try three pretrain tasks: Jigsaw puzzle with jigResNet, colorization with DeepLabV3 using ResNet-50 as backbone, SimCLR.



### Pretraining

#### JigResNet and SimCLR

The pretraining code for jigResNet and SimCLR are written and run in the jupyter notebooks in `notebooks`

#### Colorization

The code to run colorization are written in Python and run on Google Cloud Platform.

- Before start runing, `mkdir data` `cd data` `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip` `unzip -q tiny-imagenet-200.zip` to get and unzip the Tiny ImageNet data

- `generate_partial_data.py` is used to generate partial of data. 
  - Usage: `python generate_partial_data.py  original_path new_path percetage_of_data` 
  - This will randomly copy the same percentage of train and validation data from the oroginal dataset.
- `python main.py --mode pretrain --col_test`



### Baseline training

We train ResNet-50 on 10% data of the Tiny ImageNet from scratch to perform as the baseline.

`python main.py --mode baseline_train`



### Fine-tune

#### JigResNet and SimCLR

In notebooks

#### Colorization

`python3 main.py --mode finetune  --pretrain_task colorization --pretrained_model  pretrain_test_tiny-imagenet-200-0002.pth --batch_size 16`

