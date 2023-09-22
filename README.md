# CAL_GAN

### Abstract

#### Getting started

- Clone this repo.
```bash
git clone https://github.com/jkpark0825/CAL_GAN
cd CAL_GAN
```

- Install dependencies. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
pip install -r requirements.txt
```

- Prepare the training and testing dataset.
- Prepare the pre-trained models (PSNR oritented models) from https://github.com/xinntao/ESRGAN.

#### Training
If you prefer not to train the model, you can simply obtain the pretrained model by downloading it from this link: https://drive.google.com/file/d/1qWGpKE0sUKnB1Rd5_1KOdCvi5xRNUgQ9/view?usp=sharing.

Adapt yml file:  ```options/trainCAL_GAN/*.yml``` 

- Single GPU:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/CAL_GAN/*.yml --auto_resume
```

- Distributed Training:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 basicsr/train.py -opt options/train/CAL_GAN/*.yml --launcher pytorch --auto_resume
```

Training files  will be saved in the directory ```./experiments/{name}```

#### Testing

Adapt the yml file ```options/test/CAL_GAN/*.yml``` 

- Save visual results for real-world image super-resolution:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/CAL_GAN/*.yml
```

Evaluating files (logs and visualizations) will be saved in the directory ```./results/{name}```


### Acknowledgement
This code is based on 
[BasicSR](https://github.com/xinntao/BasicSR) project.
