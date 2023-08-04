# Jigsaw: Learning to Assemble Multiple Fractured Objects

---

This repository contains a minimal PyTorch implementation of the paper 
"[Jigsaw: Learning to Assemble Multiple Fractured Objects](https://arxiv.org/abs/2305.17975)".


## Installation

---

This repository has been developed and tested 
with Ubuntu 18.04 and CUDA 11.0. 
To set up the required environment, follow these steps:

1. Create a new Anaconda environment named `assembly`:
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

We provide support for the 
[Breaking Bad Dataset](https://breaking-bad-dataset.github.io/).
For more information about data processing, please refer to the dataset website.

After processing the data, ensure that you have a folder named `data` with the following structure:
```
data
├── breaking_bad
│   ├── everyday
│   │   ├── BeerBottle
│   │   │   ├── ...
│   │   ├── ...
│   ├── everyday.train.txt
│   ├── everyday.val.txt
│   └── ...
└── ...
```
Only the `everyday` subset is necessary for training. 
If you want to test the `artifact` and `other` subsets, the structure should follow the same pattern.

### Run the Experiment

---

For training, run
```shell
python train_matching.py --cfg path/to/your/yaml
```
Replace `path/to/your/yaml` with the path to your configuration file, for example:
```shell
python train_matching.py --cfg experiments/jigsaw_250e_cosine.yaml
```
The results will be stored to the directory `results/MODEL_NAME/`.

For evaluation, run
```shell
python eval_matching.py --cfg path/to/your/eval_yaml
```
Default configuration files are stored in the `experiments/` directory, 
and you are encouraged to try your own configurations. 
If you discover a better configuration, 
please let us know by raising an issue or a pull request, 
and we will update the benchmark accordingly!

## Pretrained Weights

---
To use the pretrained weights, 
download the weight file and 
add the following YAML configuration to your configuration file:
```yaml
WEIGHT_FILE: path/to/your/weight_file.ckpt
```

## Acknowledgement

---

We would like to express our gratitude to the authors of the following repositories, from which we referenced code:

* [Multi-part-assembly](https://github.com/Wuziyi616/multi_part_assembly): breaking bad benchmark code. 
* [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch): an open sourced repository for deep graph matching algorithms.
* [Pointcept](https://github.com/Pointcept/Pointcept): providing point transformer structure.

We would also like to thank the authors of all the packages we use.
We welcome any valuable suggestions for improving our repository.


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
}
```
