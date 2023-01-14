# 基于Vision Transformer的去雾算法研究与实现

**李昕 2201821 自然语言处理大作业**



> 本次实验是**基于已有模型代码Uformer（Uformer: A General U-Shaped Transformer for Image Restoration (CVPR 2022)）进行改进的**
>
> 主要针对Uformer中出现的：==1.注意力操作复杂度较高；2.对数据集没有重复利用== 问题进行解决
>
> > **针对每个问题，都进行了改进，并附有相应的结果：**
> >
> > 问题1的改进是用到了Informer提出的==概率系数自注意力机制==，**虽然恢复效果有所降低，但是大幅减少了自注意力复杂度**；
> >
> > 问题2的改进使用到了==对比正则化==**让恢复的图像接近无雾图像，并远离带雾图像**，以达到更好的恢复结果
>
> 
>
> **详细的运行流程在第2章描述；实验结果以及消融实验分别在第3章和第4章描述；而在研究过程中，还发现了修改后的不足之处，在第5章有描述**
>
> 
>
> **数据集的下载地址（第1章），对数据集的预处理（第2章） 以及 训练好的模型权重（第3章）都有详细的操作解释**



>在我提交的代码中，所有的关键代码都有详细的注释，包括但不限于：
>
>**详细的维度注释，每个关键代码块的具体含义，函数的输入输出，特殊类和函数的输入输出的示例**，这样在阅读的时候会特别方便



## 1. 数据集

![image-20230114112409563](https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202301141124722.png)

### 1.1 NH-HAZE

数据集下载：	https://competitions.codalab.org/competitions/22236#participate-get_data

> Train：1-40；Test：41-45

我们引入了NH-HAZE，一个非均匀的真实数据集，有成对真实的模糊和相应的无雾图像。因此，非均匀雾霾数据集的存在对于图像去雾场是非常重要的。

它代表第一个真实的图像去模糊数据集与非均匀的模糊和无模糊（地面真实）配对图像

为了补充之前的工作，在本文中，我们介绍了NH-HAZE，这是第一个具有非均匀模糊和无雾（地面真实）图像的真实图像去模糊数据集。

### 1.2 NTIRE 2019

DENSE-haze是一个真实的数据集，包含密集（均匀）模糊和无烟雾（地面真实）图像

官方地址：

https://data.vision.ee.ethz.ch/cvl/ntire19/#:~:text=Datasets%20and%20reports%20for%20NTIRE%202019%20challenges

https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/

另一个下载地址：

https://www.kaggle.com/rajat95gupta/hazing-images-dataset-cvpr-2019?select=GT

> Train：1-45；Test：51-55

## 2. 模型运行过程

### 2.0 模型介绍

> 在文件夹`/Uformer_ProbSparse/`下存放模型代码

![image-20220606174010393](https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206061740524.png)

<hr/>

![image-20220606173917583](https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206070243742.png)



<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206061750486.svg" alt="网络架构图" style="zoom:50%;" />

> 参考代码：https://github.com/ZhendongWang6/Uformer 

<hr/>

![image-20220606174029064](https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206061740201.png)

> 参考代码：https://github.com/zhouhaoyi/Informer2020

<hr/>

![image-20220606174044925](https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206061740064.png)

> 参考代码：https://github.com/GlassyWu/AECR-Net

### 2.1 预处理数据 --- 把训练数据图像切分成大小为256*256的小图

下载数据集存放在：

```/home/dell/桌面/TPAMI2022/Dehazing/#dataset/NH_haze/```

内含两个文件夹：`train  test`

对训练数据集处理：

```
python3 generate_patches_SIDD.py --src_dir /home/dell/桌面/TPAMI2022/Dehazing/#dataset/NH_haze/train --tar_dir /home/dell/桌面/NLP/Datasets/NH-HAZE/train_patches
```



### 2.2 训练代码My_train.py

```
python3 ./My_train.py --arch Uformer --nepoch 270 --batch_size 32 --env My_Infor_CR --gpu '1' --train_ps 128 --train_dir /media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/NLP/Datasets/NH-HAZE/train_patches --val_dir /media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/NLP/Datasets/NH-HAZE/test_patches --embed_dim 32 --warmup
```

如果要继续对模型进行训练：`--pretrain_weights` 设置预训练权重路径，我的模型预训练权重在My_best_model文件夹下，以数据集划分不同预训练权重

并添加参数 `--resume`

 

训练所有参数设置在option.py文件种，主要的参数含义：

* `--train_ps` 训练样本的补丁大小，默认为128，指多大的patches输入到模型中
* `--train_dir` `--val_dir` 训练和测试文件夹，文件夹下包含两个文件夹gt和hzay，分别包含无雾图片集和带雾图片集
* `--batch_size` 设置Batch_size，默认为3
* `--is_ab` **是否使用n a对比损失，默认为False（使用）
* `--w_loss_vgg7`对比损失使用的权重，默认为1
* `--w_loss_CharbonnierLoss`  CharbonnierLoss 所占权重，默认为1**

 

### 2.3 测试代码test_long_GPU.py和预训练权重

> 训练权重：
>
> 链接：https://pan.baidu.com/s/1a1YPTGSNa0R6I-qiTNir0A 
> 提取码：y422
>
> 模型预训练权重：将百度网盘中的`Uformer_ProbSparse/My_best_model`文件夹放到`Uformer_ProbSparse`文件夹下，里面包含4大数据集下的权重

```
python3 ./test_long_GPU.py
```

测试流程：

在My_train.py文件中，为了训练速度考虑，我们是在每个patch上进行的测试，但patch上测试结果不等于在整图上测试的结果，因此该文件是对模型在整图上结果进行测试，论文中的结果与该测试结果一致

由于代码的特殊设置，需要让输入的图片的长和宽为  `--train_ps`   的整数倍，如果不够足，则要进行扩展

主要参数解释：

* `--input_dir` **设置测试的文件夹，文件夹下包含两个文件夹gt和hzay，分别包含无雾图片集和带雾图片集**

* `--train_ps`训练样本的补丁大小，默认为128，指多大的patches输入到模型中

* 代码中的: L表示图像需要拓展长和宽为多大

  例如：输入是1200 \* 1600，patch size = 128时，L = 1664

  L需要为128倍数，且要大于输入图像的长和宽，需要根据输入图像进行调整，例如：NH-HAZE数据集上的为L = 1664
  
  



## 3. 实验结果

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206061736084.png" alt="image-20220606173637944" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206222346976.png" alt="image-20220606173646976" style="zoom: 80%;" />

结果分析：

> **根据恢复图的结果，我们发现在部分图上的效果并不是特别优异**
>
> ***可以很好的反应Vision Transformer的劣势：*该架构虽然全局建模能力强，但局部建模能力没有CNN强，因此当输入某物体占大部分空间时，恢复结果容易受到其影响；因此可以在之后改进中使用CNN和Transformer组合模型，共同对全局和局部进行建模。**



## 4. 消融实验

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206061737233.png" alt="image-20220606173744093" style="zoom:80%;" />



## 5. 总结展望

![image-20220606173837155](https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202206061738337.png)
