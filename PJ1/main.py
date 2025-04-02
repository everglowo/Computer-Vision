import numpy as np
from utils import DataHandler, load_model,save_model, Visualizer
from model import Model
from layers import FullyConnectedLayer, ActivationReLU, ActivationSoftmax, InputLayer
from training_core import SGD, SGDMomentum, CELoss, Trainer
from evaluate import Evaluator
from accuracy import Accuracy
import matplotlib.pyplot as plt 

np.random.seed(0)

# Create dataset
data_handler = DataHandler('./data')
X_train, y_train, X_test, y_test = data_handler.load_cifar10()  # 加载 CIFAR-10 数据集
X, y, X_val, y_val = data_handler.split_validation(X_train, y_train, val_ratio=0.2)  # 增加训练集和验证集的比例
X, X_test = data_handler.scale(X, X_test)  # 数据归一化
X_val = data_handler.scale(X_val)  # 验证集归一化



def create_model(n_neuron_layer=512, learning_rate=0.05,
                 decay=1e-4, moment=0.9, l2_reg_weight=0.0001):
    model = Model()

    # 网络结构
    model.add(InputLayer())  # 自动处理输入形状 (32,32,3) -> 展平为3072
    
    # 第一层: 3072 -> 512
    model.add(FullyConnectedLayer(32*32*3, n_neuron_layer, l2_reg_weight))
    model.add(ActivationReLU())
    
    # 第二层: 512 -> 512
    model.add(FullyConnectedLayer(n_neuron_layer, n_neuron_layer, l2_reg_weight))
    model.add(ActivationReLU())
    
    # 输出层: 512 -> 10
    model.add(FullyConnectedLayer(n_neuron_layer, 10, l2_reg_weight))
    model.add(ActivationSoftmax())

    print(f"### Model architecture: 3072 -> {n_neuron_layer} -> {n_neuron_layer} -> 10")

    # 设置优化器、损失函数和评估指标
    optimizer = SGDMomentum(learning_rate, decay, moment) if use_sgd_momentum else SGD(learning_rate, decay)
    model.set_items(
        loss=CELoss(),
        optimizer=optimizer,
        accuracy=Accuracy()
    )
    model.finalize()

    return model

# 最优参数  
learning_rates = [0.01]
n_neurons = [512]
l2_weights = [0.00001]
momentums = [0.9] 

# 超参数
n_epoch = 5  # 增加训练轮数，因为CIFAR-10较复杂
batch_size = 128  # 小批量更新，减少每次的梯度波动
# 选择使用的优化器
use_sgd_momentum = True  # 改为 True 使用SGD Momentum, False则使用普通SGD
best_val_acc = 0.0

# for lr in learning_rates:
#     for neurons in n_neurons:
#         for l2 in l2_weights:
#             for mom in momentums:
#                 print(f"\n### Training with lr={lr}, neurons={neurons}, l2={l2}, momentum={mom}")
                
#                 # Create model with the optimal hyperparameters
#                 model = create_model(n_neuron_layer=neurons,
#                                      learning_rate=lr,
#                                      decay=1e-4,
#                                      moment=mom,
#                                      l2_reg_weight=l2)
                
#                 # Train the model
#                 trainer = Trainer(model)
#                 trainer.train(X, y, epochs=n_epoch, batch_size=batch_size,
#                               print_every=200, val_data=(X_val, y_val))

# model = load_model('./result/best_cifar10_model.pkl')
# evaluator = Evaluator(model)
# evaluator.eval(X_test, y_test, batch_size=128)
                
model = create_model()

# Train the model
print(f"\n### Training with lr={learning_rates}, neurons={n_neurons}, l2={l2_weights}, momentum={momentums}")
trainer = Trainer(model)
trainer.train(X, y, epochs=n_epoch, batch_size=batch_size,
              print_every=100, val_data=(X_val, y_val))

model.set_parameters(model.best_weights)

# Test the model
evaluator = Evaluator(model)
evaluator.eval(X_test, y_test, batch_size=128)

# Plot loss, accuracy and learning rate
trainer.plot_figs()

#Visualize weights
Visualizer.visualize_first_fc_layer(model)

Visualizer.visualize_second_fc_layer(model)

# Save the parameters and the model
model.save_params('./result/best_cifar10_params.pkl')
save_model(model, './result/best_cifar10_model.pkl')
print("### Model saved.")
