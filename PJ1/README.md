# Computer Vision PJ1
Three Layer Neural Network

```
├── data                     # CIFAR-10数据集
├── utils.py                 # 载入数据 & 保存、加载训练好的模型 & 可视化神经网络loss,accuracy的辅助函数
├── layers.py                # 输入层 & 全连接层 & 激活函数
├── training-core.py         # 优化器 & 损失函数 & 训练神经网络所需的类
├── model.py                 # 传入参数，构建神经网络模型
├── evaluate.py              # 评估模型
├── accuracy.py              # 计算accuracy

├── activation_experiment.py # 对比activation function性能
├── optimizer_experiment.py  # 对比optimizer性能
├── plot                     # 以上两个实验的实验结果
├── main.py                  # 寻找最优超参数,两个全连接层权重的热力图、分布图函数

├── model                    # 存放训练好的模型参数
├── readme.md                # 关于实验代码的说明
```

训练模型
```
python main.py
```

如果想改变超参数，直接在main.py里修改即可
```
# 最优参数  
learning_rates = [0.01]
n_neurons = [512]
l2_weights = [0.00001]
momentums = [0.9] 

# 超参数
n_epoch = 15  # 增加训练轮数，因为CIFAR-10较复杂
batch_size = 128  # 小批量更新，减少每次的梯度波动
# 选择使用的优化器
use_sgd_momentum = True  # 改为 True 使用SGD Momentum, False则使用普通SGD
best_val_acc = 0.0
```
加载并测试训练好的模型
```
model = load_model('./result/best_cifar10_model.pkl')
evaluator = Evaluator(model)
evaluator.eval(X_test, y_test, batch_size=128)
```
