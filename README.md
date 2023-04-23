# Fuzzing-Deep-Learning-Libraries-via-Large-Language-Models
## 对一篇论文的小规模复现 [论文地址](https://arxiv.org/abs/2212.14834 "悬停显示")
### 方法实现
---
#### 用codex初始代码种子库（具体代码实现查看codex.py）
1. 选取主流深度学习库——`tensorflow`（由于本仓库只实现小规模数据复现，故只选取tensorflow库的`核心api接口`进行自动化代码测试）
2. 选择的`核心api接口`主要有以下三类：  
* 基本运算：tf.expand_dims、tf.split、tf.concat、tf.cast、tf.reshape、tf.equal、tf.matmul、tf.argmax、tf.squeeze  
- 搭建网络：tf.nn.conv2d、tf.nn.max_pool、tf.nn.avg_pool、tf.nn.relu、tf.nn.dropout、tf.nn.l2_normalize、tf.nn.batch_normalization、tf.nn.l2_loss、tf.nn.softmax_cross_entropy_with_logits  
* 训练优化：tf.train.Saver、tf.train.Saver.restore、tf.train.GradientDescentOptimizer(0.01).minimize(loss)、tf.train.exponential_decay(learning_rate=1e-2, global_step=sample_size/batch, decay_rate=0.98, staircase=True)、tf.train.string_input_producer(string_tensor, num_epochs, shuffle=True)、tf.train.shuffle_batch(tensors=[example, label], batch_size, capacity, min_after_dequeue)、tf.train.Coordinator()、tf.train.start_queue_runners(sess, coord)   
3. 输入提示词，获得上述各个api接口应用codex模型自动生成的的代码种子库  
以下选取tf.nn作为目标api接口进行实例展示：  
输入提示：  
  0)create a function named f to do the following steps   
  1)Import TensorFlow 2.10.0  
  2)Generate input data    
  3)Call the API tf.nn.conv2d(input,filters,strides, padding,data_format='NHWC',dilations=None,name=None)  
代码生成：  
      ```python   
      def f():  
          import tensorflow as tf  
          input_data = tf.random.normal([1, 2, 2, 1])  
          filters = tf.random.normal([2, 2, 1, 1])  
          strides = [1, 1, 1, 1]  
          padding = 'SAME'  
          output = tf.nn.conv2d(input_data, filters, strides, padding, data_format='NHWC', dilations=None, name=None)  
          return output
---
#### 用incoder突变代码种子库（具体代码实现查看incoder.py）
1. 变异算子主要有四类——`参数`、`前缀`、`后缀`和`方法`
2. 变异算子的选择  
将如何为每个目标api选择最合适的突变算子问题视作`Bernoulli bandit problem`，并且利用经典的`Thompson Sampling (TS) algorithm`来选择突变算子。
3. 自动变异测试代码  
根据上述利用codex模型自动生成的`种子代码`结合上步选择出的`变异算子`进行`代码突变`展示：  
    * 目标api：`tf.nn.conv2d`  
    - 选择的变异算子：`方法`
    * 代码掩盖：
        ```python
        def f():
          import tensorflow as tf
          input_data = tf.<insert>([1, 2, 2, 1])
          filters = tf.<insert>([2, 2, 1, 1])
          strides = [1, 1, 1, 1]
          padding = 'SAME'
          output = tf.nn.conv2d(input_data, filters, strides, padding, data_format='NHWC', dilations=None, name=None)
          return output
    - 代码突变：
        ```python 
        def f():
          import tensorflow as tf
          input_data = tf.random_uniform([1, 2, 2, 1])
          filters = tf.random_uniform([2, 2, 1, 1])
          strides = [1, 1, 1, 1]
          padding = 'SAME'
          output = tf.nn.conv2d(input_data, filters, strides, padding, data_format='NHWC', dilations=None, name=None)
          return output

4. 评判自动化生成的测试代码  
    * 数据流图的深度：自动变异后的测试代码数据流图深度越深该测试代码得分越高
    - api接口的数量：自动变异后的测试代码包含的不同api接口数量越多得分越高
---
#### 结合coder和incoder的模糊检测流程（具体代码实现查看fuzzing.py）
1. 确定`目标api`和`时间预算`（在多长时间内对选择的目标api进行代码突变）
2. 利用codex模型初始化生成所选的目标api的代码种子库
3. 初始化每个突变算子的先验概率
4. 进入时间循环内
5. 评判自动变异生成的代码，根据得分选择当前的种子代码
6. 选择变异算子
7. 对当前种子
     



