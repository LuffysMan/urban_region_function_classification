# 城市区域功能分类

## 简介
代码是在两位大佬的基础上改的, 唯一的改进在于处理数据的时候使用了多线程, 加快了速度; 但是提交的准确率并没有提升; 虽然做的不好, 但是多多少少还是做了一些工作, 尤其是多线程处理数据的部分, 还有比较通用的模型框架, 以后也可以用到, 在此基础上改进.

## 快速起步
### 1.1 依赖的库
```
tensorflow-gpu==1.13.1
opencv-python
pandas 
```
### 1.2 数据准备
将数据放在data文件夹下，如下所示：
- data/test/image/xxxxxx.jpg
- data/test/visit/xxxxxx.txt
- data/train/image/00x/xxxxxx_00x.jpg
- data/train/visit/xxxxxx_00x.txt

把压缩文件解压后手动调整下, 放成上面的结构.

### 1.3 数据转换
把visit数据转换为26x24x7的矩阵，原版没有用多线程需要1个小时, 我修改为10个线程同时转换，大概要10分钟左右.
```
python -m dataset.visit2array_mp
```
转换后的数据存储在:
- data/train/npy
- data/test/npy

### 1.4 生成tfrecord
将图片, 文本和标签做成tfrecord; 其中训练集被分为10个tfrecord, 便于交叉验证; 测试集单独生成一个tfrecord;
```
python -m datast/tfrecord -t=0      #生成训练集record
python -m datast/tfrecord -t=1      #生成测试机record
```
生成的tfrecord存储在：
- data/train/record/record_x
- data/test/record/record_0

加载数据集的时候使用了tensorflow的高层api-tf.data API, 比较方便

### 1.5 训练
```
python train.py
```
为了调参方便，每组实验存在不同的文件夹里。
需要输入显卡的编号和文件夹名称，比如：
```
device id: 0
model id: 001
```

查看tensorboard：
```
cd model/
tensorboard --logdir=./
```

### 1.6 测试
```
python predict.py
```
生成的结果存储在result文件夹下, 后缀为指定的模型id：
- result/result_xxx.txt





