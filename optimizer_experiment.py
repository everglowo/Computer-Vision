import numpy as np
import matplotlib.pyplot as plt
from utils import DataHandler
from model import Model
from layers import FullyConnectedLayer, ActivationReLU, ActivationSoftmax, InputLayer
from training_core import SGD, SGDMomentum, RMSProp, Adam, CELoss, Trainer
from evaluate import Evaluator
from accuracy import Accuracy

np.random.seed(0)
data_handler = DataHandler('./data')
X_train, y_train, X_test, y_test = data_handler.load_cifar10()
X, y, X_val, y_val = data_handler.split_validation(X_train, y_train, val_ratio=0.2)
X, X_test = data_handler.scale(X, X_test)
X_val = data_handler.scale(X_val)

n_epoch = 15
batch_size = 128
learning_rate = 0.01
n_neuron_layer = 512
l2_reg_weight = 0.00001
momentum = 0.9

def create_model(optimizer):
    """创建固定使用ReLU+ReLU的模型，指定优化器"""
    model = Model()
    model.add(InputLayer())
    model.add(FullyConnectedLayer(32*32*3, n_neuron_layer, l2_reg_weight))
    model.add(ActivationReLU())
    model.add(FullyConnectedLayer(n_neuron_layer, n_neuron_layer, l2_reg_weight))
    model.add(ActivationReLU())
    model.add(FullyConnectedLayer(n_neuron_layer, 10, l2_reg_weight))
    model.add(ActivationSoftmax())
    
    model.set_items(
        loss=CELoss(),
        optimizer=optimizer,
        accuracy=Accuracy()
    )
    model.finalize()
    return model

# 定义要比较的优化器
optimizers = {
    'SGD': SGD(learning_rate=learning_rate),
    'SGDMomentum': SGDMomentum(learning_rate=learning_rate, momentum=momentum),
    'RMSProp': RMSProp(learning_rate=learning_rate),
    'Adam': Adam(learning_rate=learning_rate)
}

# 存储结果
optimizer_results = {
    'train_loss': {},
    'val_loss': {},
    'val_acc': {}
}

# 训练并记录结果
for opt_name, optimizer in optimizers.items():
    print(f"\n=== Training with {opt_name} ===")
    model = create_model(optimizer)
    trainer = Trainer(model)
    trainer.train(X, y, epochs=n_epoch, 
                 batch_size=batch_size, val_data=(X_val, y_val))
    
    # 保存训练过程中的指标
    optimizer_results['train_loss'][opt_name] = trainer.loss_train
    optimizer_results['val_loss'][opt_name] = trainer.loss_val
    optimizer_results['val_acc'][opt_name] = trainer.acc_val

# 绘制优化器比较图
def plot_optimizer_comparison(results):
    plt.figure(figsize=(15, 5))
    
    # 训练损失对比
    plt.subplot(1, 2, 1)
    for opt_name, losses in results['train_loss'].items():
        plt.plot(losses, label=opt_name, linewidth=2)
    plt.title('Training Loss Comparison (ReLU+ReLU)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 验证损失对比
    plt.subplot(1, 2, 2)
    for opt_name, losses in results['val_loss'].items():
        plt.plot(losses, label=opt_name, linewidth=2)
    plt.title('Validation Loss Comparison (ReLU+ReLU)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("优化器对比图已保存为 optimizer_comparison.png")

# 调用绘图函数
plot_optimizer_comparison(optimizer_results)