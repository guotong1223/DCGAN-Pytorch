# DCGAN-Pytorch
这是CDGAN使用Pytorch实现的代码
## 依赖
```
python 3.6
pytorch 1.1
```
## 训练方式
1.下载数据集，解压后在当前目录下创建一个data目录，该目录下存放下载的数据集,本代码的结果基于网上爬取的动漫数据集，
[下载链接](https://pan.baidu.com/s/1eSifHcA) 提取码：g5qa,
其目录结构如下所示：
```
/path/to/anime
    -> faces
        -> 000001.jpg
        -> 000002.jpg
        -> 000003.jpg
        -> 000004.jpg
           ...
```
输入图像的大小可以任意设定，但不宜过大，因为可能效果不好。之所以可以设定任意大小的输入图像，是因为本代码中有如下代码：
```
ds_size = opt.img_size // 2 ** 4
self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
```
2.然后修改代码中的数据目录：
```
parser.add_argument("--dataroot", type=str, default="data/your_datatset", help="interval between image sampling")
```
3.训练
```
python dcgan_faces_tutorial.py
```
## 结果

![](https://github.com/lovepiano/DCGAN-Pytorch/blob/master/93200.png)
