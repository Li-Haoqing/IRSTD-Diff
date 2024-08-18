
# IRSTD-Diff: Generative Infrared Small Target Detection with Diffusion Model

## Requirement
     Python 3.7.10
     torch 1.10.1

## Training

``python scripts/train.py --data_name NUDT --data_dir input data direction --out_dir output data direction --image_size 256 --lr 1e-4 --batch_size 4``

In default, the _'.pt'_ will be saved at `` ../results/ `` 

## Sampling

``python scripts/sample.py --data_name NUDT --data_dir input data direction --out_dir output data direction --model_path saved model --image_size 256``

In default, the samples will be saved at `` ../sample/ `` 

## Valuating

``python utils/val.py --img_size 256 --output_path '../sample/' --batch-size 1``


## Datasets
Dataset folder should be like:

https://github.com/RuiZhang97/ISNet
~~~
IRSTD-1k
└───imges
│       │   XDU0.png
│       │   XDU1.png
│       │  ...
└───masks
│       │   XDU0.png
│       │   XDU1.png
│       │  ...
└───trainval.txt
└───test.txt
~~~
https://github.com/YimianDai/sirst
~~~
NUAA-SIRST
└───idx_320
│       │   trainval.txt
│       │   test.txt
└───idx_427
│       │   trainval.txt
│       │   test.txt
└───imges
│       │   Misc_1.png
│       │   Misc_2.png
│       │  ...
└───masks
│       │   Misc_1_pixels0.png
│       │   Misc_2_pixels0.png
│       │  ...
~~~
https://github.com/YeRen123455/Infrared-Small-Target-Detection
~~~
NUDT-SIRST
└───imges
│       │   000001.png
│       │   000002.png
│       │  ...
└───masks
│       │   000001.png
│       │   000002.png
│       │  ...
└───trainval.txt
└───test.txt
~~~


## Citation

     @ARTICLE{10601187,
       author={Li, Haoqing and Yang, Jinfu and Xu, Yifei and Wang, Runshi},
       journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
       title={Mitigate Target-Level Insensitivity of Infrared Small Target Detection via Posterior Distribution Modeling}, 
       year={2024},
       volume={17},
       pages={13188-13201},
       doi={10.1109/JSTARS.2024.3429491}}
