# MViTDepth

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

## ⚙️Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0
pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8
pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image
pip3 install mmcv-full==1.3.0 mmsegmentation==0.11.0  
pip install timm einops IPython
```
 We ran our experiments with PyTorch 1.9.0, CUDA 11.1, Python 3.7 and Ubuntu 18.04. 

 Note that our code is built based on [Monodepth2](https://github.com/nianticlabs/monodepth2). However, we only use it for Monocular videos training and estimation.


## 

## 💾KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.

The above conversion command creates images which match our experiments, where KITTI `.png` images were converted to `.jpg` on Ubuntu 16.04 with default chroma subsampling `2x2,1x1,1x1`.
We found that Ubuntu 18.04 defaults to `2x2,2x2,2x2`, which gives different results, hence the explicit parameter in the conversion command.

You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.

## ⏳Training

Pre-trained MonoViT can be avaliable at [here](https://github.com/zxcqlf/MonoViT) 

By default models and tensorboard event files are saved to `~/tmp/<model_name>`.
This can be changed with the `--log_dir` flag.

**Monocular training:**

```shell
python --model_name model_name --num_layers 18 --decoder_channel_scale [200,100,50] --encoder_mobilevit ["xxs", "xs", "s"]
```

The decoder_channel_scale means

| decoder channel scale | decoder channels for each stage |
| :-------------------: | :-----------------------------: |
|          200          |     {16, 32, 64, 128, 256}      |
|          100          |      {8, 16, 32, 64, 128}       |
|          50          |       {4, 8, 16, 32, 64}        |

The encoder_mobilevit means the backbone network of MobileViTv1

| Name |    Encoder    |
| :--: | :-----------: |
| xxs  | MobileViT_xxs |
|  xs  | MobileViT_xs  |
|  s   |  MobileViT_s  |



## 📊KITTI evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```
...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --decoder_channel_scale [200,100,50] --encoder_mobilevit ["xxs", "xs", "s"] --eval_mono
```
If you train your own model with our code you are likely to see slight differences to the publication results due to randomization in the weights initialization and data loading.

An additional parameter `--eval_split` can be set.
The three different values possible for `eval_split` are explained here:

| `--eval_split`        | Test set size | For models trained with... | Description  |
|-----------------------|---------------|----------------------------|--------------|
| **`eigen`**           | 697           | `--split eigen_zhou` (default) or `--split eigen_full` | The standard Eigen test files |
| **`eigen_benchmark`** | 652           | `--split eigen_zhou` (default) or `--split eigen_full`  | Evaluate with the improved ground truth from the [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) |
| **`benchmark`**       | 500           | `--split benchmark`        | The [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) test files. |

Because no ground truth is available for the new KITTI depth benchmark, no scores will be reported  when `--eval_split benchmark` is set.
Instead, a set of `.png` images will be saved to disk ready for upload to the evaluation server.



Our depth estimation results on the KITTI dataset in `192 x 640` as follows:

|     Method     | $Abs\space Rel\downarrow$ | $Sq\space Rel\downarrow$ | $RMSE\downarrow$ | $RMSE\space log\downarrow$ | $\delta <1.25\uparrow$ | $\delta < 1.25^2\uparrow$ | $\delta < 1.25^3\uparrow$ |
| :------------: | :-----------------------: | :----------------------: | :--------------: | :------------------------: | :--------------------: | :-----------------------: | :-----------------------: |
|   MViTDepth    |           0.106           |          0.766           |      4.507       |           0.180            |         0.891          |           0.964           |           0.983           |
| MViTDepth small |           0.107           |          0.788           |      4.579       |           0.183            |         0.887          |           0.962           |           0.983           |
| MViTDepth tiny |           0.111           |          0.825           |      4.731       |           0.187            |         0.881          |           0.961           |           0.983           |



Our depth estimation results on the KITTI dataset in `320 x 1024` as follows:

|    $Method$    | $Abs\space Rel\downarrow $ | $Sq\space Rel\downarrow $ | $RMSE\downarrow $ | $RMSE\space log\downarrow $ | $\delta <1.25\uparrow $ | $\delta < 1.25^2\uparrow $ | $\delta < 1.25^3\uparrow $ |
| :------------: | :------------------------: | :-----------------------: | :---------------: | :-------------------------: | :---------------------: | :------------------------: | :------------------------: |
|   MViTDepth    |           0.102            |           0.756           |       4.415       |            0.177            |          0.899          |           0.966            |           0.984            |
| MViTDepth small |           0.104            |           0.772           |       4.460       |            0.180            |          0.893          |           0.965            |           0.984            |
| MViTDepth tiny |           0.109            |           0.818           |       4.603       |            0.184            |          0.887          |           0.963            |           0.983            |



We also provide models in different sizes to accommodate various edge devices

|    $Method$    | $decoder\space channel$ | $Abs\space Rel\downarrow $ | $Sq\space Rel\downarrow $ | $RMSE\downarrow $ | $RMSE\space log\downarrow $ | $\delta <1.25\uparrow $ | $\delta < 1.25^2\uparrow $ | $\delta < 1.25^3\uparrow $ |
| :------------: | :---------------------: | :------------------------: | :-----------------------: | :---------------: | :-------------------------: | :---------------------: | :------------------------: | :------------------------: |
|   MViTDepth    |            2            |           0.105            |           0.753           |       4.474       |            0.180            |          0.892          |           0.965            |           0.984            |
| MViTDepth small |            2            |           0.106            |           0.785           |       4.555       |            0.182            |          0.889          |           0.963            |           0.983            |
| MViTDepth tiny |            2            |           0.110            |           0.829           |       4.675       |            0.187            |          0.884          |           0.962            |           0.982            |
|   MViTDepth    |            1            |           0.106            |           0.766           |       4.507       |            0.180            |          0.891          |           0.964            |           0.983            |
| MViTDepth small |            1            |           0.107            |           0.788           |       4.579       |            0.183            |          0.887          |           0.962            |           0.983            |
| MViTDepth tiny |            1            |           0.111            |           0.825           |       4.731       |            0.187            |          0.881          |           0.961            |           0.983            |
|   MViTDepth    |           0.5           |           0.106            |           0.736           |       4.523       |            0.180            |          0.888          |           0.963            |           0.984            |
| MViTDepth small |           0.5           |           0.108            |           0.755           |       4.604       |            0.182            |          0.883          |           0.962            |           0.984            |
| MViTDepth tiny |           0.5           |           0.114            |           0.862           |       4.789       |            0.189            |          0.874          |           0.959            |           0.983            |



Our various models complexit as follow:

|    $Method$    | $decoder\space channel$ | $Parameters$ | $GFLOPs$ |
| :------------: | :---------------------: | :----------: | :------: |
|   MViTDepth    |            2            |     8.1M     |   6.7    |
| MViTDepth small |            2            |     4.3M     |   4.7    |
| MViTDepth tiny |            2            |     3.1M     |   3.2    |
|   MViTDepth    |            1            |     6.3M     |   4.7    |
| MViTDepth small |            1            |     2.8M     |   2.8    |
| MViTDepth tiny |            1            |     1.8M     |   1.5    |
|   MViTDepth    |           0.5           |     5.5M     |   4.0    |
| MViTDepth small |           0.5           |     2.3M     |   2.2    |
| MViTDepth tiny |           0.5           |     1.3M     |   0.9    |



## Contact us 

Contact us to avaliable model pre-trained weights: 2222108036@stmail.ujs.edu.cn



## Acknowledgement

Thanks the authors for their works:

[Monodepth2](https://github.com/nianticlabs/monodepth2)

[MonoViT](https://github.com/zxcqlf/MonoViT)

[MobileViTv1](https://github.com/apple/ml-cvnets) 

