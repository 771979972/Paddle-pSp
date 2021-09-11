﻿﻿﻿# PSP_cn
*****
### English|简体中文
* PSP
    * 一、简介
    * 二、复现结果
    * 三、数据集
    * 四、环境依赖
    * 五、预训练模型
    * 六、快速开始
         * 训练
         * 测试
    * 七、代码结构与详细说明
         * 代码结构
         * 参数说明
    * 八、模型信息

# **一、简介**

***
本项目基于paddlepaddle框架复现pixel2style2pixel(pSp).pSp框架基于一个新颖的编码器网络，直接生成一系列风格向量，这些向量被馈送到一个预训练的风格生成器中，形成扩展的W+潜在空间。该编码器可以直接重建真实的输入图像.
#### **论文**
* [1] Richardson E ,  Alaluf Y ,  Patashnik O , et al. Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation[J].  2020.
#### **参考项目**
* [https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
#### **项目aistudio地址**
* notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/2331440](https://aistudio.baidu.com/aistudio/projectdetail/2331440)

# **二、复现结果**

#### **指标（在CelebA-HQ上测试）**

|  模型  | LPIPS  |Similarity|MSE|
|  :----:| :----: |:----:|:----:|
|论文|0.17|0.56|0.03|
|Pytorch模型|0.15|0.57|0.03|
|Paddle模型|0.17|0.57|0.03|

#### **视觉对比**

|  论文模型结果  | Paddle复现结果  |
|  :----:| :----: |
|![1](examples/1.png)|<img src="inference/inference_coupled/052329.jpg" alt="052329" style="zoom: 25%;" />|
|![1](examples/2.png)|<img src="inference/inference_coupled/179349.jpg" alt="1" style="zoom: 25%;" />|
|![1](examples/3.png)|<img src="inference/inference_coupled/145789.jpg" alt="1" style="zoom:25%;" />|

# **三、数据集**

训练集下载： [FFHQ训练集](https://github.com/NVlabs/ffhq-dataset)。图片数据保存`FFHQ/`。

测试集下载：[CelebA-HQ](https://aistudio.baidu.com/aistudio/datasetdetail/49226)。图片数据保存在`CelebA_test/`。

# **四、环境依赖**

硬件：GPU、CPU

框架：PaddlePaddle >=2.0.0

# **五、预训练模型**

下载后将模型的参数保存在`work\pretrained_models\`中

|  模型(文件名)   | Description  |
|  ----  | ----  |
| FFHQ StyleGAN(stylegan2-ffhq-config-f.pdparams) | StyleGAN 在FFHQ上训练，来自 [rosinality](https://github.com/rosinality/stylegan2-pytorch) ，输出1024x1024大小的图片 |
| IR-SE50 Model(model_ir_se50.pdparams)| IR SE 模型，来自 [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) 用于训练中计算ID loss。 |
| CurricularFace Backbone(CurricularFace_Backbone.paparams)    |     预训练的 CurricularFace model，来自 [HuangYG123](https://github.com/HuangYG123/CurricularFace) 用于Similarity的评估。  |
|   AlexNet(alexnet.pdparams和lin_alex.pdparams)    |   	用于lpips loss计算。    |
| StyleGAN Inversion(psp_ffhq_inverse.pdparams)      |   pSp trained with the FFHQ dataset for StyleGAN inversion.    |

链接：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg) 提取码：m3nb

pSp encoder预训练模型下载：

|模型|Description|
|--|--|
|StyleGAN Inversion(psp_ffhq_inverse.pdparams)|pSp trained with the FFHQ dataset for StyleGAN inversion.|

# **六、快速开始**

#### **编译算子**

	python scripts/compile_ranger.py
#### **训练**

	python scripts/train.py \
	--dataset_type=ffhq_encode \
	--exp_dir=exp/test \
	--workers=0 \
	--batch_size=8 \
	--test_batch_size=8 \
	--test_workers=0 \
	--val_interval=2500 \
	--save_interval=5000 \
	--encoder_type=GradualStyleEncoder \
	--start_from_latent_avg \
	--lpips_lambda=0.8 \
	--l2_lambda=1 \
	--id_lambda=0.1 \
	--optim_name=ranger

#### **测试**

#### **inferernce**

```
python scripts/inference.py \
--exp_dir=inference \
--checkpoint_path=pretrained_models/psp_ffhq_inverse.pdparams \
--data_path=CelebA_test \
--test_batch_size=8 \
--test_workers=4
```

#### **计算其他指标**
* 计算LPIPS

    python scripts/calc_losses_on_images.py \
    --mode lpips \
    --data_path=inference/inference_results \
    --gt_path=CelebA_test

* 计算MSE

	python scripts/calc_losses_on_images.py \
	--mode l2 \
	--data_path=inference/inference_results \
	--gt_path=CelebA_test

* 计算Similarity

	python scripts/calc_id_loss_parallel.py \
	--data_path=inference/inference_results \
	--gt_path=CelebA_test

# **七、代码结构与详细说明**

#### **代码结构**

```
├─config          # 配置
├─data            #数据集加载
   ├─CelebA_test  # 测试数据图像
├─logs            #日志
   ├─train        # 训练日志
   ├─test         # 测试日志
├─models          # 模型
   ├─encoders    # 编码器
   ├─loss        # 损失函数
   ├─mtcnn       #     
   ├─stylegan2   #       
   ├─utils       # 编译算子
├─scripts         #算法执行
    trian         #训练
    inference     #测试
├─utils           # 工具代码
│  README.md      #英文readme
│  README_cn.md   #中文readme
```

#### **参数说明**

|  参数   | 默认值  |
| --  | -- |
| config  | None, 必选 |

# **八、模型信息**

模型的总体信息如下：

|信息|说明|
|--|--|
| 框架版本| Paddle 2.1.2 |
| 应用场景|图像生成|
|支持硬件|GPU / CPU|