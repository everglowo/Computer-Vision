# Computer Vision PJ1
Three Layer Nerual Network

├──data                     # CIFAR-10数据集
├──utils.py                 # 载入数据 & 保存、加载训练好的模型 &可视化神经网络的辅助函数
├──layers.py                # 输入层 & 全连接层 & 激活函数
├──training-core.py         # 优化器 & 损失函数 & 训练神经网络所需的类
├──model.py                 # 传入参数，构建神经网络模型
├──evaluate.py              # 评估模型
├──accuracy.py              # 计算accuracy

├──activation_experiment.py # 对比activation function性能
├──optimizer_experimen.py   # 对比optimizer性能
├──plot                     # 以上两个实验的实验结果
├──main.py                  # 寻找最优超参数

    
├──model                    # 存放训练好的模型参数
├──readme.md                # 关于实验代码的说明
