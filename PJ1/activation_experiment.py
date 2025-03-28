import numpy as np
import matplotlib.pyplot as plt
from utils import DataHandler
from model import Model
from layers import FullyConnectedLayer, ActivationReLU, ActivationSigmoid, ActivationTanh, ActivationSoftmax, InputLayer
from training_core import SGD, SGDMomentum, CELoss, Trainer
from evaluate import Evaluator
from accuracy import Accuracy

np.random.seed(0)
data_handler = DataHandler('./data')
X_train, y_train, X_test, y_test = data_handler.load_cifar10()
X, y, X_val, y_val = data_handler.split_validation(X_train, y_train, val_ratio=0.2)
X, X_test = data_handler.scale(X, X_test)
X_val = data_handler.scale(X_val)

# 使用小子集
X_small, y_small = X[:5000], y[:5000]
X_val_small, y_val_small = X_val[:1000], y_val[:1000]
X_test_small, y_test_small = X_test[:1000], y_test[:1000]

n_epoch = 10
batch_size = 128
learning_rate = 0.05
n_neuron_layer = 256
l2_reg_weight = 0.0001
momentum = 0.9
use_sgd_momentum = False

def create_model(act_fn1, act_fn2):
    model = Model()
    model.add(InputLayer())
    model.add(FullyConnectedLayer(32*32*3, n_neuron_layer, l2_reg_weight))
    model.add(act_fn1())
    model.add(FullyConnectedLayer(n_neuron_layer, n_neuron_layer, l2_reg_weight))
    model.add(act_fn2())
    model.add(FullyConnectedLayer(n_neuron_layer, 10, l2_reg_weight))
    model.add(ActivationSoftmax())

    optimizer = SGDMomentum(learning_rate, 1e-4, momentum) if use_sgd_momentum else SGD(learning_rate, 1e-4)
    model.set_items(loss=CELoss(), optimizer=optimizer, accuracy=Accuracy())
    model.finalize()
    return model

activation_functions = [ActivationReLU, ActivationSigmoid, ActivationTanh]
results_acc = {}
results_loss = {}
all_losses = {}
all_accuracies = {}

for act_fn1 in activation_functions:
    for act_fn2 in activation_functions:
        model = create_model(act_fn1, act_fn2)
        trainer = Trainer(model)
        # 调用train方法进行训练
        trainer.train(X_small, y_small, epochs=n_epoch, batch_size=batch_size, val_data=(X_val_small, y_val_small))
        model.set_parameters(model.best_weights)
        evaluator = Evaluator(model)
        acc_test, loss_test = evaluator.eval(X_test_small, y_test_small, batch_size=128)
        results_acc[f"{act_fn1.__name__} + {act_fn2.__name__}"] = acc_test
        results_loss[f"{act_fn1.__name__} + {act_fn2.__name__}"] = loss_test
        # 保存每个组合训练过程中的验证损失和准确率
        all_losses[f"{act_fn1.__name__} + {act_fn2.__name__}"] = trainer.loss_val
        all_accuracies[f"{act_fn1.__name__} + {act_fn2.__name__}"] = trainer.acc_val


def plot_activation_comparison(all_losses, all_accuracies):
    import os
    if not os.path.exists('activation_plots'):
        os.makedirs('activation_plots')
    
    # 使用当前可用的样式替代 'seaborn'
    available_styles = plt.style.available
    preferred_styles = ['seaborn-v0_8', 'ggplot', 'seaborn', 'default']
    selected_style = next((style for style in preferred_styles if style in available_styles), 'default')
    plt.style.use(selected_style)
    
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.rcParams['font.size'] = 12
    
    for combo_name in all_losses.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 绘制损失曲线
        ax1.plot(all_losses[combo_name], 'b-', label='Training Loss')
        ax1.set_title(f'{combo_name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 绘制准确率曲线
        ax2.plot(all_accuracies[combo_name], 'r-', label='Validation Accuracy')
        ax2.set_title(f'{combo_name} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f'activation_plots/{combo_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"图表已保存到 activation_plots 目录")

# 调用绘图函数
plot_activation_comparison(all_losses, all_accuracies)
