import numpy as np
from utils import DataHandler, save_model, Visualizer
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

# 超参数
n_epoch = 5  # 增加训练轮数，因为CIFAR-10较复杂
batch_size = 128  # 小批量更新，减少每次的梯度波动

# 选择使用的优化器
use_sgd_momentum = True  # 改为 True 使用SGD Momentum, False则使用普通SGD

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

# # # 粗调
# # learning_rates = [0.01,0.05,0.1]
# # n_neurons = [256,512]
# # l2_weights = [0.00001,0.0001]
# # momentums = [0.9] 

# # 精调
# # learning_rates = [0.01,0.05]
# # n_neurons = [512]
# # l2_weights = [1e-5,5e-5]
# # momentums = [0.9]

# 最优参数  
learning_rates = [0.01]
n_neurons = [512]
l2_weights = [0.00001]
momentums = [0.9] 


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

# Save the parameters and the model
model.save_params('./result/best_cifar10_params.pkl')
save_model(model, './result/best_cifar10_model.pkl')
print("### Model saved.")

def visualize_first_fc_layer(model):
    # 提取第一个全连接层（索引0）
    fc_layer = model.layers[1]  # layers[0]是InputLayer，layers[1]是第一层FC
    
    # 获取权重矩阵（3072输入 → 512输出）
    weights = fc_layer.weights
    
    # 创建画布
    plt.figure(figsize=(15, 6))
    
    # ------------------
    # 子图1：权重热力图
    # ------------------
    plt.subplot(1, 2, 1)
    
    # 使用对称的颜色范围（以0为中心）
    vmax = np.percentile(np.abs(weights), 99)  # 排除1%的极端值
    heatmap = plt.imshow(weights.T,  # 转置矩阵使输入维度在x轴
                        cmap='coolwarm',
                        aspect='auto',
                        vmin=-vmax,
                        vmax=vmax)
    
    plt.colorbar(heatmap, label='Weight Value')
    plt.title('First FC Layer Weights (3072 → 512)\nColor Range: ±{:.3f}'.format(vmax))
    plt.xlabel('Input Pixels (Flattened 32x32x3)')
    plt.ylabel('Hidden Neurons')
    
    # ------------------
    # 子图2：权重分布
    # ------------------
    plt.subplot(1, 2, 2)
    
    # 计算统计指标
    mean = weights.mean()
    std = weights.std()
    
    # 绘制直方图（排除极端值）
    n, bins, patches = plt.hist(weights.flatten(), 
                               bins=100, 
                               range=(-vmax, vmax),
                               color='teal',
                               edgecolor='white',
                               alpha=0.7)
    
    # 添加统计标注
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
    plt.axvline(mean + std, color='orange', linestyle=':', label=f'±1σ ({std:.4f})')
    plt.axvline(mean - std, color='orange', linestyle=':')
    
    plt.title('Weight Distribution\nL2 Reg: {}'.format(fc_layer.l2_reg_weight))
    plt.xlabel('Weight Value')
    plt.ylabel('Count (log scale)')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 在训练后调用
visualize_first_fc_layer(model)
def visualize_second_fc_layer(model):
    # 提取第二个全连接层（注意层索引可能需要调整）
    # 根据你的模型结构，假设 layers[3] 是第二个FC层（索引从0开始）
    # InputLayer → FC1 → ReLU → FC2 → ReLU → FC3 → Softmax
    fc_layer = model.layers[3]
    
    weights = fc_layer.weights
    
    plt.figure(figsize=(16, 6))
    
    # ===================================================================
    # 子图1：权重矩阵可视化（512x512）
    # ===================================================================
    plt.subplot(1, 2, 1)
    
    # 使用动态范围（排除极端值）
    abs_max = np.percentile(np.abs(weights), 99.9)
    matrix_view = weights[:256, :256]  # 仅显示前256x256区域
    
    im = plt.imshow(matrix_view, 
                   cmap='PiYG',  # 改用双色系增强对比
                   aspect='equal', 
                   vmin=-abs_max,
                   vmax=abs_max)
    
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"FC Layer 2 Weight Submatrix\n(First 256x256 of 512x512)\nTotal Range: [{weights.min():.3f}, {weights.max():.3f}]")
    plt.xlabel("Input Neurons (Layer 1 Output)")
    plt.ylabel("Output Neurons (Layer 2)")
    
    # ===================================================================
    # 子图2：权重分布与稀疏性分析
    # ===================================================================
    plt.subplot(1, 2, 2)
    
    # 计算统计量
    positive_ratio = (weights > 0).mean() * 100
    dead_neurons = (np.abs(weights).sum(axis=0) == 0).sum()  # 列和为0的神经元
    
    # 双坐标轴分析
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 主坐标：分布直方图
    n, bins, _ = ax1.hist(weights.flatten(), 
                         bins=200, 
                         range=(-abs_max, abs_max),
                         color='darkcyan',
                         alpha=0.6,
                         log=True)
    
    # 副坐标：累积分布
    cumulative = np.cumsum(n) / np.sum(n)
    ax2.plot(bins[1:], cumulative, 'r--', linewidth=2)
    
    # 标注关键阈值
    for percentile in [25, 50, 75, 95]:
        value = np.percentile(weights, percentile)
        ax1.axvline(value, color='grey', linestyle=':', alpha=0.5)
        ax1.text(value, n.max()*0.8, f'{percentile}%', rotation=90, ha='right')
    
    ax1.set_title(
        f"Distribution | L2: {fc_layer.l2_reg_weight}\n"
        f"Pos/Neg Ratio: {positive_ratio:.1f}% | Dead: {dead_neurons} neurons"
    )
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Count (log)", color='darkcyan')
    ax2.set_ylabel("Cumulative %", color='red')
    
    plt.tight_layout()
    plt.show()

# 调用函数
visualize_second_fc_layer(model)