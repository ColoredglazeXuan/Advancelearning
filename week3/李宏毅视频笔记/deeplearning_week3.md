## CNN（Convolutional Neural Networks，卷积神经网络，专用在影像识别上）

### 形成原因：		

​		一张图片是3维的tensor，红绿蓝三色，三个channel，因此输入一个100X100分辨率的图片，在Fully Connected Network里一个neuron就需要$3*10^4$个Weight，整个Layer和model的参数就更多了，所以针对影像识别，需要减少Weight的数量。

### CNN主要特点

##### Typical Setting

​		不同于Fully Connected Network，CNN中，一个neuron（用于识别物体的一个部分）只需要关注一个Receptive field（图片中一个指定的区域），一个neuron的Weight数量就只需要关注这个Receptive field的输入。Receptive field对应的neuron可以有多个（用于识别不同物体的部分）。

​		Receptive field的设置可以任意决定，一般情况下是3X3的Receptive field，同一层新的Receptive field根据Stride（自己设置，一般是1、2）的大小向右边移动进行设置，超出的padding部分一般设置为0（也有其他的方法）。

##### Observation

​		检测同一个物体部分的neuron共享参数（parameter sharing），因为识别同一个物体部分的neuron虽然对应的Receptive field不同，但是工作相同，可以减少重复。

​		因此，所有的Receptive field对应相同的识别功能neuron的都相同，每个neuron的参数叫做Filter。

​		Pooling：将图片缩小化，减少运算量，因为每个neuron对所有Receptive field进行运算后得出的结果还是一个矩阵，可以看出一张图片，所以Pooling就是分割output矩阵后，选取各个小矩阵的一个值，重新构成input。（存在缺陷，不能对识别很细微的东西进行Pooling）

##### Convolutional Layer

​		图片首先设置Receptive field，每个neuron对所有Receptive field进行运算后得出的结果还是一个矩阵，每个neuron的output就是一个channel，可以看成一个新的图片。同时，形成的新的图片再用同一个neuron进行一次计算，可以视作是一个neuron照顾到了一个更大的范围。



## RNN（Recurrent Neural Network，循环神经网络，有记忆力的network）

### 形成原因

​		比如说，使用Slot Filling进行词汇识别，可能存在理解出错的情况，比如：不能识别是出发地或者目的地。

（Slot Filling，将一个句子拆分成多个输入x，一个输入x包含一个单词，进行词汇识别，一个单词用一个向量表示（1-of-N encoding），范围之外的单词用Dimension for "other"（加入other来表示超出之外的词）/Word hashing（用一些片断的组合来表示一个词，构成失败则表示其他词）表示。如果一句话中只包含一个地点，难以判断是目的地或者出发地（不会检测“出发”或者“到”这两个词，识别的单词一般仅包含“地名”和“时间”？））

### RNN的主要特点

​		因此RNN在每个hidden Layer输出时，不仅传输到下一层，结果还会被存在memory中，memory会影响下一次的hidden Layer的输出（Elman Network）/每次output Layer的输出存到memory中（Jordan Network，因为output Layer的输出和结果有关联，所以一般效果更好），这样一句话中的“到台北”和“去台北”的输出也是不一样的了。

​		Bidirection RNN：既有正向的network还有逆向的network，每个词正逆向的hidden Layer一起决定一个output Layer的输出。（相当于获取到了整个输入的信息）



##### 存在问题

​	RNN存在一个问题，training时，因为memory会影响后面的结果输出，迭代次数过多的情况下gradient的值变化会很大，（gradient vanishing，比如：y的1000次方，输入是1、1.01和0.99、0.01的gradient差距很大），导致learning rate的设置不好确定。解决方法就是使用LSTM（因为memory的值不是直接会被更新，而是要根据其他的三个Gate值决定，完成输入、输出和重置）

​		Long Short-term Memory（LSTM）：memory仅在input Gate打开时进行修改，仅在output Gate打开时将memory中的值传输到input中，forget Gate决定什么时候把memory中的值forget掉，这三个值由机器自己学习决定。

​		GRU（Gated Recurrent Unit，和LSTM类似，但是减少了一个Gate的计算）：旧的不去，新的不来，input memory的同时会清理掉以前的memory，input gate和forget gate是绑定的
