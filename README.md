# Crack-Segmentation

Semantic Segmentation is an essential task in Computer Vision, which has to do with
segmenting the objects of certain categories within a given frame. This task can be applied to
undertaker other tasks such as, Autonomous Vehicles, Medical Diagnosis, Farming, etc. 
In this case, semantic segmentation was executed for Crack Segmentation, which is critical
for assessment of Structural Health and its Monitoring. It can also server to measure the width
and length of the crack for even further assessment of the structural health.

A benchmark study of multiple State-of-the-Art and latest DL models for Crack Segmentation is conducted.
In particular, a light-weight yet effective model named EfficientCrackNet was developed and implemented. This is
a very recent model that was developed after LMM-Net, another model that was developed for the same
purpose. EfficientCrackNet outperforms LMM-Net in various metrics including Precision and mean 
Intersection over Union (mIoU) despite its simplicity. 

**_U-Net_** [Paper](https://arxiv.org/abs/1505.04597) <br />

<div align="center">
<img src="https://github.com/user-attachments/assets/6c3a0a9a-add5-4410-bbef-18acf811c41e" width=85% height=85%>
</div><br />

**_LMM-Net_** [Paper](https://ieeexplore.ieee.org/document/10539282) <br />

<div align="center">
<img src="https://github.com/user-attachments/assets/cfcc5955-3ea3-4abc-ac28-795a65861f90" width=85% height=85%>
</div><br />

**_EfficientCrackNet_** [Paper](https://arxiv.org/abs/2409.18099) <br />

<div align="center">
<img src="https://github.com/user-attachments/assets/10b59e5c-5ecf-48bc-98a4-e18dd7686681" width=85% height=85%> 
</div><br />

The dataset used for Crack Segmentation is DeepCrack, a public dataset of concrete surface cracks.
EfficientCrackNet achieved 4.44% higher Precision and 0.77& higher mIoU scores over LMM-Net.

## Commands
### Training
```python
python main_dev.py --data_dir (data directory) --model_name (name of model to run) --epochs (# of epochs) --alpha (alpha decrease value for Dice Loss) --data_name (name of data to run) --run_num (# of run)
```

### Evaluation
```python
python eval.py --data_dir (data directory) --model_name (name of model to run) --data_name (name of data to run) --run_num (# of run)
```
