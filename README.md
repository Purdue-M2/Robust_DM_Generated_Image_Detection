# Robust DM-generated images detection with CLIP
Santosh, Li Lin, Xin Wang, Shu Hu
_________________
This repository is the official implementation of our paper [Robust Diffusion Model-Generated Image Detection with CLIP](https://arxiv.org/abs/2404.12908)

### 1. Data Preparation
* Download the dataset from the [HuggingFace](https://huggingface.co/datasets/elsaEU/ELSA_D3)
* After downloading the parquet files, tranform them into h5 files.
```python
python get_data.py
```
* Use [CLIP ViT L/14](https://github.com/openai/CLIP) to extract image and text features and save them into h5 folder (e.g., clip_train/0000.h5 and clip_val/0000.h5) by executing [h5_process.py](./h5_process.py).
Note: [h5_process.py](./h5_process.py) file uses [clip_feature.py](./clip_feature.py) to extract images and text features.
```python
python h5_process.py
```

### 2. Train the model 
* load, 'train' folder for train_dataset in [train.py](./train.py); load 'val' folder for val_dataset in [train.py](./train.py).

```python
    train_dataset = DFADDataset('train')
    val_dataset = DFADDataset('val')
```

```python
python train.py
```

* Use CVaR & AUC Loss

```python
 model_trainer(loss_type='auc', alpha=0.1,  batch_size=2048, num_epochs=32)
```
Tune **gamma** on CVaR + AUC loss to find the best tradeoff hyperparameter

## Citation
Please kindly consider citing our papers in your publications. 
```bash
@inproceedings{Santosh2024robust,
      title={Robust Diffusion Model-Generated Image Detection with CLIP}, 
      author={Santosh and Li Lin and Xin Wang and Shu Hu},
      booktitle={1st Workshop on New Trends in AI-Generated Media and Security},
      year={2024},
}
```
  
