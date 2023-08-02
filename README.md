# Jigsaw: Learning to Assemble Multiple Fractured Objects

---

This is a minimal PyTorch implementation of paper: [Jigsaw: Learning to Assemble Multiple Fractured Objects](https://arxiv.org/abs/2305.17975).


## Installation

---

This repository is developed and tested with 
Ubuntu 18.04, CUDA 11.0, Python 3.8, Pytorch 1.10.2, 
Pytorch Lightning 1.9.1, and Pytorch Geometric 2.0.4.

1. Create an anaconda environment `assembly`:
    ```shell
    conda env create -f environment.yaml
    conda activate assembly
    ```
2. Install custom CUDA ops for Chamfer distance:
    ```shell
    cd utils/chamfer
    pip install -e .
    ```

## Datasets

---

We support [Breaking Bad Dataset](https://breaking-bad-dataset.github.io/), please refer
to the dataset website for more information about the data-processing.

After processing the data, you shall create a folder `data` with structure like:
```
data
├── breaking_bad
│   ├── everyday
│   │   ├── BeerBottle
│   │   │   ├── ...
│   │   ├── ...
├── everyday.train.txt
├── everyday.val.txt
└── ...
```
Only `everyday` subset is necessary for training. 
If you want to test the `artifact` and `other` subsets, the structure follows the same.

### Run the Experiment

---

For training, run
```shell
python train_matching.py --cfg path/to/your/yaml
```
You shall replace `path/to/your/yaml` by path to your configuration file, e.g.
```shell
python train_matching.py --cfg experiments/jigsaw_250e_cosine.yaml
```
For evaluation, run
```shell
python eval_matching.py --cfg path/to/your/eval_yaml
```
Default configuration files are stored in `experiments/`
and you are welcomed to try your own configurations. 
If you find a better yaml configuration, 
please let us know by raising an issue or a PR,
and we will update the benchmark!

## Pretrained Weights

---

To use the pretrained weights, first download the weight file, 
then add
```yaml
WEIGHT_FILE: path/to/your/weight_file.ckpt
```
to your configuration file.

## Acknowledgement

---

We would like to thank and acknowledge referenced codes from:

* [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch): an open sourced repository for deep graph matching algorithms.
* [Multi-part-assembly](https://github.com/Wuziyi616/multi_part_assembly): breaking bad benchmark code. 
* [Pointcept](https://github.com/Pointcept/Pointcept): providing point transformer structure.

## Citation

---

If you find this repository useful in your research, please cite
```
@misc{lu2023jigsaw,
      title={Jigsaw: Learning to Assemble Multiple Fractured Objects}, 
      author={Jiaxin Lu and Yifan Sun and Qixing Huang},
      year={2023},
      eprint={2305.17975},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}```
