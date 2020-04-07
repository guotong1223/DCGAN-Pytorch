# DCGAN-Pytorch
这是CDGAN使用Pytorch实现的代码
## 依赖
python 3.6
pytorch 1.1
## 训练方式
1.下载数据集，解压后在当前目录下创建一个data目录，该目录下存放下载的数据集celeba，其目录结构如下所示：
```
/path/to/celeba
    -> img_align_celeba
        -> 188242.jpg
        -> 173822.jpg
        -> 284702.jpg
        -> 537394.jpg
           ...
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
如下是使用所爬取到的动漫头像作为数据集训练的结果：

![](https://github.com/lovepiano/DCGAN-Pytorch/blob/master/93200.png)
