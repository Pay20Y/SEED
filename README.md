# SE_ASTER

## Introduction
This is the implementation of the paper "SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition"
This code is based on the [aster.pytorch](https://github.com/ayumiymk/aster.pytorch), we sincerely thank ayumiymk for his awesome repo and help.

## How to use
### Env
```
PyTorch == 1.1.0
torchvision == 0.3.0
fasttext == 0.9.1
```
Details can be found in requirements.txt

### Train
##### Prepare your data
- Download the pretrained language model (bin) from [here](https://fasttext.cc/docs/en/crawl-vectors.html)
- Update the path in the lib/tools/create_all_synth_lmdb.py
- Run the lib/tools/create_all_synth_lmdb.py
- Note: it may result in large storage space, you can modify the datasets/dataset.py to generate the word embedding in an online way

##### Run
- Update the path in train.sh, then
```
sh train.sh
```

### Test
- Update the path in the test.sh, then
```
sh test.sh
```

## Experiments
### Evaluation on benchmarks
* You can downlod the benchmark datasets from [BaiduYun](https://pan.baidu.com/s/1Z4aI1_B7Qwg9kVECK0ucrQ) (key: nphk) shared by clovaai in this [repo](https://github.com/clovaai/deep-text-recognition-benchmark).

|     Checkpoint  | IIIT5K | IC13-1015  | IC13-857 | IC15-1811 | IC15-2077  | SVT | SVTP  |  CUTE  |
|:-----------------:|:------:|:----------:|:--------:|:------:|:----------:|:---:|:-----:|:------:|
|  [OneDrive](https://1drv.ms/u/s!AvXu1eY3TODqjlhQgTXrkd4wj11D?e=d3eBQw) [BaiduYun](https://pan.baidu.com/s/1JHlDCDYDV4VBn7oZ5_JPyw)(key: x54e)  |  93.4  |  93.5 | 94.5|   79.8    | 75.8     |88.4 |82.0  | 84.0   |
### Evalution with lexicons
* Existing methods replace the predicted word with the nearest lexicon word under the metric of edit distance (ED). With the semantic information, we can choose the most semantics similar (SS) word based on the nearest edit distance.

| Methods | IIIT5K-50  |  IIIT5K-1K  | SVT-50  |  IC13   |   IC15  |
|:-------:|:----------:|:-----------:|:-------:|:--------:|:--------:|
|  ED    |  99.06    |   97.87   |  96.36  | 97.44   |   87.76 |
| ED + SS |  <b>99.27 |  <b>97.93  | <b>96.45| <b>97.64 |<b>88.07  |
  
### About the word embedding
 * Directly use word embedding from the pre-trained LM during training and inference.

| IIIT5K | IC13  | IC15-1811 | IC15-2077  | SVT  | SVTP  |  CUTE  |
|:------:|:-----:|:---------:|:----------:|:----:|:-----:|:------:|
|  94.6  |  93.8 |  85.0    |   79.6   | 90.9 |  84.2 |  85.4  |
  
### Exploration on global information

* We try to use [Aggregation Cross-Entropy](https://github.com/summerlvsong/Aggregation-Cross-Entropy) as the global information instead of the semantics. This part of code will be released in next few days.
 
| IIIT5K | IC13  | IC15-1811 | IC15-2077  | SVT  | SVTP  |  CUTE  |
|:------:|:-----:|:---------:|:----------:|:----:|:-----:|:------:|
|  93.8 |  91.3 |  78.7    |   -   | 90.1 |  81.6 |  81.9  |

## Citation
```
@inproceedings{qiao2020seed,
  title={{SEED}: Semantics enhanced encoder-decoder framework for scene text recognition},
  author={Qiao, Zhi and Zhou, Yu and Yang, Dongbao and Zhou, Yucan and Wang, Weiping},
  booktitle={CVPR},
  year={2020},
}
@article{shi2018aster,
  title={{ASTER}: An attentional scene text recognizer with flexible rectification},
  author={Shi, Baoguang and Yang, Mingkun and Wang, Xinggang and Lyu, Pengyuan and Yao, Cong and Bai, Xiang},
  journal={TPAMI},
  volume={41},
  number={9},
  pages={2035--2048},
  year={2018},
  publisher={IEEE}
}
```