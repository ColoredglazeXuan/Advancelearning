## Pytorch

​		Pytorch主要使用torch.nn包来构建神经网络。

​		`torch.Tensor` - 一个多维数组，支持诸如`backward()`等的自动求导操作，同时也保存了张量的梯度。

​		`nn.Module` - 神经网络模块。是一种方便封装参数的方式，具有将参数移动到GPU、导出、加载等功能，包含各个层和一个`forward(input)`方法，该方法返回`output`，forward函数需要自己定义。

​		`nn.Sequential`-顺序容器，用来添加构成神经网络的layers，现有的layers-`nn.Linear`（全连接层）、`nn.ReLU`（ReLU激活函数）、`nn.Conv2D`（2维卷积层）。

​		`nn.Parameter` - 张量的一种，当它作为一个属性分配给一个`Module`时，它会被自动注册为一个参数。

​		另外nn包中还有很多损失函数，用于计算loss的值，包括`nn.MSELoss`、`nn.CrossEntropyLoss`等

​		`backward()`，反向传播计算梯度。

​		`torch.optim`包，用于设置Learning rate、Optimization方式、参数更新等，包括SGD、Adamw等。

​		`zero_grad()`用于重置梯度，用于下一次的update。

​		`DataLoader(dataset, batch_size, shuffle=True)`，载入数据集，并根据batch_size划分batch，仅在training时shuffle为true，让batch生产随机

## PaddlePaddle

​		飞桨（PaddlePaddle）是一个深度学习框架（类似Pytorch？）

​		paddle：飞桨的主库，paddle 根目录下保留了常用API的别名，当前包括：paddle.tensor、paddle.device目录下的所有API。

​		paddle.nn：组网相关的API，包括 Linear、卷积 Conv2D、循环神经网络LSTM、损失函数CrossEntropyLoss、激活函数ReLU等。

​		init函数：在类的初始化函数中声明每一层网络的实现函数。在房价预测任务中，只需要定义一层全连接层，模型结构和**第1.3节**保持一致。

​		forward函数：在构建神经网络时实现前向计算过程，并返回预测结果，在本任务中返回的是房价预测结果。

​		用.train()和.eval()改变模型的执行状态。

​		model.prepare：用于定义模型训练参数，如优化器`paddle.optimizer.SGD`、损失函数`paddle.nn.MSELoss`等。

​		model.fit：用于模型训练，并指定相关参数，如训练轮次`epochs`，批大小`batch_size`。

​		model.evaluate：用于在测试集上评估模型的损失函数值和评价指标。

​		PaddlePaddle和Pytorch包的对应关系

|        Pytorch         |       PaddlePaddle       |                     说明                     |
| :--------------------: | :----------------------: | :------------------------------------------: |
|        torch.nn        |        Paddle.nn         |        包括了神经网络相关的大部分函数        |
|       nn.Module        |         nn.Layer         | 搭建网络时集成的父类，包含了初始化等基本功能 |
|      torch.optim       |     Paddle.optimizer     |                  训练优化器                  |
| torchvision.transforms | paddle.vision.transforms |             数据预处理、图片处理             |
|  torchvision.datasets  |  paddle.vision.datasets  |              数据集的加载与处理              |

​		网络结构

|       PyTorch        |     PaddlePaddle     |                    说明                    |
| :------------------: | :------------------: | :----------------------------------------: |
|      nn.Conv2d       |      nn.Conv2D       |                 2维卷积层                  |
|    nn.BatchNorm2d    |    nn.BatchNorm2D    |         Batch Normalization 归一化         |
|       nn.ReLU        |       nn.ReLU        |                ReLU激活函数                |
|     nn.MaxPool2d     |     nn.MaxPool2D     |               二维最大池化层               |
| nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool2D | 自适应二维平均池化（只用给定输出形状即可） |
|      nn.Linear       |      nn.Linear       |                  全连接层                  |
|    nn.Sequential     |    nn.Sequential     |          顺序容器，用来添加layers          |
|    torch.flatten     |    paddle.flatten    |                  展平处理                  |
|    torch.softmax     |    paddle.softmax    |                 softmax层                  |

​		数据加载与处理

|             PyTorch             |          PaddlePaddle           |       说明       |
| :-----------------------------: | :-----------------------------: | :--------------: |
|       transforms.Compose        |       transforms.Compose        |   图片处理打包   |
|  transforms.RandomResizedCrop   |  transforms.RandomResizedCrop   |     随机裁剪     |
| transforms.RandomHorizontalFlip | transforms.RandomHorizontalFlip |   随机水平翻转   |
|       transforms.ToTensor       |       transforms.ToTensor       | 转化为tensor格式 |
|      transforms.Normalize       |      transforms.Normalize       |    数据标准化    |
|      datasets.ImageFolder       |     datasets.DatasetFolder      | 指定数据集文件夹 |
|   torch.utils.data.DataLoader   |      paddle.io.DataLoader       |    加载数据集    |

​		模型训练

|       PyTorch       |     PaddlePaddle      |                             说明                             |
| :-----------------: | :-------------------: | :----------------------------------------------------------: |
|     (net).train     |      (net).train      |                           训练模式                           |
|   (loss).backward   |    (loss).backward    |                         反向传递误差                         |
|     optim.Adam      |      optim.Adam       | Adam优化器，注意paddlepaddle中的参数分别为parameters和learning _rate，与PyTorch中是不同的 |
| (optimizer).no_grad | (optimizer).zero_grad |                           梯度清零                           |
|     torch.save      |    paddle.jit.save    |                                                              |
|     (net).eval      |      (net).eval       |                           预测模式                           |

​		模型预测

|     PyTorch     |   PaddlePaddle   |     说明     |
| :-------------: | :--------------: | :----------: |
| torch.unsqueeze | paddle.unsqueeze | 增加数据维度 |
|  torch.no_grad  |  paddle.no_grad  |  不计算梯度  |

​		设备指定

|   PyTorch    |   PaddlePaddle    |   说明   |
| :----------: | :---------------: | :------: |
| torch.device | paddle.set_device | 指定设备 |

