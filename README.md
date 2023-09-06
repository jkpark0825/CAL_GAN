# CAL_GAN

### Abstract

#### Getting started

- Clone this repo.
```bash
git clone https://github.com/csjliang/LDL
cd LDL
```

- Install dependencies. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
pip install -r requirements.txt
```

- Prepare the training and testing dataset by following this [instruction](datasets/README.md).
- Prepare the pre-trained models by following this [instruction](experiments/README.md).

#### Training

First, check and adapt the yml file ```options/train/LDL/train_Synthetic_LDL.yml``` (or ```options/train/LDL/train_Realworld_LDL.yml``` for real-world image super-resolution), then

- Single GPU:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/LDL/train_Synthetic_LDL.yml --auto_resume
```
or
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python realesrgan/train.py -opt options/train/LDL/train_Realworld_LDL.yml --auto_resume
```

- Distributed Training:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 basicsr/train.py -opt options/train/LDL/train_Synthetic_LDL.yml --launcher pytorch --auto_resume
```
or 
```bash
PYTHONPATH=":${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train/LDL/train_Realworld_LDL.yml --launcher pytorch --auto_resume
```

Training files (logs, models, training states and visualizations) will be saved in the directory ```./experiments/{name}```

#### Testing

First, check and adapt the yml file ```options/test/LDL/test_LDL_Synthetic_x4.yml``` (or ```options/test/LDL/test_LDL_Realworld_x4.yml``` for real-world image super-resolution), then

- Calculate metrics and save visual results for synthetic tasks:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/LDL/test_LDL_Synthetic_x4.yml
```

- Save visual results for real-world image super-resolution:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/LDL/test_LDL_Realworld_x4.yml
```

Evaluating files (logs and visualizations) will be saved in the directory ```./results/{name}```

The Training and testing steps for scale=2 are similar.

#### Get Quantitative Metrics

First, check and adapt the settings of the files in [metrics](scripts/metrics), then (take PSNR as an example) run
```bash
PYTHONPATH="./:${PYTHONPATH}" python scripts/metrics/table_calculate_psnr_all.py
```
Other metrics are similar.

### License

This project is released under the Apache 2.0 license.

### Citation
```
@inproceedings{jie2022LDL,
  title={Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution},
  author={Liang, Jie and Zeng, Hui and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

### Acknowledgement
This project is built based on the excellent [BasicSR](https://github.com/xinntao/BasicSR) project.

### Contact
Should you have any questions, please contact me via `liang27jie@gmail.com`.
