<h1 align="center">
IDSNN:Image Deraining Using Spiking Neural Network
</h1>
<p align="center">
    Project of AI3610 Brain-Inspired Intelligence, 2024 Fall, SJTU
    <br />
    <a href="https://github.com/zzctmd"><strong>Zichen Zou</strong></a>
    <br />
</p>

> **Abstract:** 
Recently, spiking neural networks (SNNs) have demonstrated substantial potential in computer vision tasks.
In this project, we improve an Efficient Spiking Deraining Network, called ESDNet.Our work is motivated by the observation that the restoration effect of real images is not outstanding enough, and the introduction of perceptual loss can significantly enhance the visual effect of real images after deraining. Therefore, we replace the loss with a spatial loss containing Edge Aware Dists Loss and conduct training on more real datasets.By this way, our improved ESDNet can effectively detect and analyze the characteristics of rain streaks by  using Edge Aware Dists Loss. This also enables better guidance for the deraining process and facilitates high-quality image reconstruction. 

## 🛠️ Requirements
Please run the following commands:
1. Create a new conda environment
```
conda create -n ESDNet python=3.10
conda activate ESDNet 
```
2. Install dependencies
```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib scikit-image opencv-python numpy einops natsort tqdm lpips tensorboardX pyiqa
```
3.prepare the spiking jelly latest version 0.0.0.0.15
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
```

## Datasets
Download the datasets from the [交大云盘](https://jbox.sjtu.edu.cn/l/n1hqFJ) and put them into data folder for training and testing.
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain12</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>Rain1200</th>
    <th>RW-Data</th>
    <th>GT-Rain</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>交大云盘</td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/n1hqFJ">Download</a> </td>
    <td align="center"> <a href="https://jbox.sjtu.edu.cn/l/n1hqFJ">Download</a> </td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/n1hqFJ">Download</a> </td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/n1hqFJ">Download</a> </td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/n1hqFJ">Download</a> </td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/n1hqFJ">Download</a> </td>
  </tr>
</tbody>
</table>

## 🤖 Pre-trained Models
Download the Pre-trained Models from [交大云盘](https://jbox.sjtu.edu.cn/l/31U9lt) and put each of them into Corresponding Subfolders of pretrained_models folder for testing.
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>Rain1200</th>
    <th>GT-Rain</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>交大云盘</td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/31U9lt">Download</a> </td>
    <td align="center"> <a href="https://jbox.sjtu.edu.cn/l/31U9lt">Download</a> </td>
    <td > <a href="https://jbox.sjtu.edu.cn/l/31U9lt">Download</a> </td>
    <td > <a href="https://jbox.sjtu.edu.cn/l/31U9lt">Download</a> </td>
  </tr>
</tbody>
</table>

## 🚀 Training,Testing and Evaluation

### Train
Run the following script to train a model:
```sh
python train.py --train_dir <Your trainset> --val_dir <Your validation set>
```

### Test
Run the following script to test the trained model and save the output in the result folder.
```sh
python test.py --data_path <Your testset> --save_path <result saved path> --weights <model path>
```
### Evaluation
Run the following script to evaluate the output picture:
```sh
python evaluation.py --generated_images_path <Recovered images path> --target_path <Ground-truth -path>
```

### 💡 Visual Results
The stored visual results are obtained by using the improved model trained on Rain1200 to restore the rainy images of each dataset respectively.
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain12</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>Rain1200</th>
    <th>RW-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>交大云盘</td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/WHH0q3">Download</a> </td>
    <td align="center"> <a href="https://jbox.sjtu.edu.cn/l/WHH0q3">Download</a> </td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/WHH0q3">Download</a> </td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/WHH0q3">Download</a> </td>
    <td> <a href="https://jbox.sjtu.edu.cn/l/WHH0q3">Download</a> </td>
  </tr>
</tbody>
</table>

## Primary Tasks and Intermediate Tasks
According to the following hints, the primary and intermediate parts of our major assignment project can be completed.

>For Primary Task

1.Download the  MNIST dataset from [交大云盘](https://jbox.sjtu.edu.cn/l/v1DzA5) and put it into the `./Primary Task/data` folder

2.Run these commands for training

```
cd Primary Task
python SNN_MNIST.py
```

3.The best version is also saved in [交大云盘](https://jbox.sjtu.edu.cn/l/v1DzA5), with an test accuracy rate of 0.9885.

>For Intermediate Task

1.Download the colorized-MNIST-master from [交大云盘](https://jbox.sjtu.edu.cn/l/71i7wM) and put it into the `./Intermediate Task` folder

2.Run the `./Intermediate Task/Transformer_SNN_Colored MNIST.ipynb` 

3.The model version is also saved in [交大云盘](https://jbox.sjtu.edu.cn/l/71i7wM)
## 🚨 Notes

1. Send e-mail to zzcnb123456@sjtu.edu.cn if you have critical issues to be addressed.
2. The real world datasets GT-Rain training effect is not very good ,because there are a large number of homogeneous photos in the training and testing sets.

## 👍 Acknowledgment

This code is based on the [Restormer](https://github.com/swz30/Restormer), [spikingjelly](https://github.com/fangwei123456/spikingjelly),[DFOSD](https://github.com/JianzeLi-114/DFOSD),[ESDNet](https://github.com/MingTian99/ESDNet) and [C2PNet](https://github.com/YuZheng9/C2PNet). Thanks for their awesome work.
