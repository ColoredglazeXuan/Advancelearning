## Self-attention

#### 产生原因

​		在Fully connected network中，输入相同，输出也相同，因此做语句识别时，输入同一个词，只能得到一个词性，然而实际可能不相同（I saw a saw）

#### 特点

​		Self-attention在进行训练时，会将输入进行一定的处理，生成新的输入向量，用于训练，新的输入向量生成会考虑所有的输入（一整个句子，相较于RNN，因为memory的生成和保存时间原因，Self-attention考虑的情况更全面）。具体过程如下：

![QQ截图20240728154230](.\picture\QQ截图20240728154230.png)

​		其中$q = W^q *a$、$k = W^k *a$、$v = W^v *a$，$α_{i,j} = q^T_i*k_j$、$α_{i,j} '= exp(α_{i,j})/\sum_jexp(α_{i,j})$、$b^i = \sum_iα_{i,j}'*v^i$。其中的a、q、k、v和b都是向量，W都是矩阵，α则是数值，所以最后通过b和v的值判断输入之间的相关性（q和k越相似，向量的内积越大，α越大，最后b和对应的v相似度越大，生成$α'$的Soft-max函数可以修改成ReLU）。

![QQ截图20240728160342](.\picture\QQ截图20240728160342.png)

#### 其他

##### Multi-head Self-attention

​		实际进行识别时，识别类型不止语句，因此输入可能存在不同的相关性，所以在Self-attention中进行多个相似性计算，产生多个q、k、v（同一个相似性计算的向量互相进行计算，也即是$q^{i,1}$只和$k^{i,1}$进行计算，最后所有的b在进行一定的处理得到最后的输出。

![QQ截图20240728160808](.\picture\QQ截图20240728160808.png)

##### Positional Encoding

​		此前的Self-attention中没有考虑位置的关系，比如同一个词在句首和句末的影响是相同的，但是一个词在句首是动词的可能性就比较低，所以Positional Encoding加入了位置向量positional vector $e^i$，只要在Self-attention前输入$a^i$时，加入$e^i$就行。



##### Batch Normalization

​		training过程中，因为输入的不同feature $x^i$的值存在一些feature一直很大、一些feature一直很小，最后导致不同Weight的改变量对最后结果的影响一部分很大、一部分很小，最后因为数据集本身的原因，导致training过程中不同Weight对loss的影响差距较大，不同的dimension影响不同，导致training困难。

​		Feature Normalization是对不同dimension的输入x进行标准化$x = \frac{x - m}{σ}$，

![QQ截图20240728163823](.\picture\QQ截图20240728163823.png)

​		而在Deep Learning中前一层的output也是下一层的input，所以在Deep Learning中每一层的input都需要进行Feature Normalization，$z = \frac{z - μ}{σ}$，（因为μ和σ都是向量，所以进行标准化时需要向量中对应的值进行计算，）

![QQ截图20240728164148](.\picture\QQ截图20240728164148.png)

​		Batch Normalization就是考虑在使用batch进行数据集的分割时，仅对一个batch内的feature进行处理，并且仅适用于batch_size比较大的时候。

​		testing部分因为不存在batch，数据集的量不一定，所以进行Normalization时，μ的值之间拿training过程中生成的值，$μ = P*ū + （1-P）u^t$，ū的值是training过程中所有μ值的平均数，$μ^t$是最后一次training的μ值。



## Sequence-to-Sequence（Seq2Seq）

#### 产生原因

​		因为在一些语言的翻译时，output的长度不一定以及部分语言没有文字，需要翻译成可理解的文字。Seq2Seq就是让机器自己决定output的长度。

### 特点

​		Seq2Seq的架构主要是两部分Encoder、Decoder，比如说Transformer的架构：

![QQ截图20240728170405](.\picture\QQ截图20240728170405.png)

### Encoder

​		Encoder的输入和输出都是一堆向量，中间包括多个block（一个Self-attention和一个Fully Connected network）

![QQ截图20240728172818](.\picture\QQ截图20240728172818.png)

​		在transformer中多增加了residual connection（将input加入到output中）、Layer normalization（对同一个向量进行normalization，而不是不同dimension（维度））以及positional encoding（在Encoder的第一步中加入，然后才进入block）

![QQ截图20240728173002](.\picture\QQ截图20240728173002.png)

### Decoder

​		Decoder包括AT（Autoregressive）和NAT（Non-Autoregressive）

​		AT首先给Decoder输入Encoder的output和BEGIN信号（特殊符号，一维向量）、然后Decoder开始输出表示一个字（词）的向量（包括所有的字或者词以及END信号），这个词再输入到Decoder中，直到输出END。

![QQ截图20240728174407](.\picture\QQ截图20240728174407.png)

​		Decoder的block和Encoder相似，其中Self-attention被修改为了Masked Self-attention（计算b向量时，不再考虑之前的输入a的资讯，也就是生成b1值用a1计算，计算b2用a1和a2，b4则考虑a1、a2、a3、a4（因为Decoder中下一步的a是上一步的output，需要产生后才能计算））

![QQ截图20240728174530](.\picture\QQ截图20240728174530.png)

​		NAT输入Encoder的output和多个BEGIN信号（BEGIN的数量通过Encoder的input或者output来生成或者直接输入大量BEGIN，输出END后的字就不需要了），直接输出句子。相较于AT，NAT平行化计算，能够比较快的输出，但是training的质量没有AT好。

### Cross attention

​		Cross attention是连接Encoder和Decoder的桥梁，本质上是一个Self-attention，让Decoder进入每一个block时能够获取到Encoder的输出信息（Decoder每多一个输入（也就是上一步的output）会多计算一个向量，也即是下一个Fully connected network的输入）。

![QQ截图20240728180230](.\picture\QQ截图20240728180230.png)

### Teacher Forcing

​		在Decoder部分，修改输入，不再是前一步的output作为输入，而是把正确答案一步步作为输入。（但是存在Testing过程中看到的自身的输入，导致部分错误）