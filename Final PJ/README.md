<h1 align="center">DATA130051.01 Final Project</h1>
<h3 align="center"> 周璇 李明泽  </h3>

## Contents
- [Task 1: NeRF](#task-1-nerf)
- [Task 2: Instant NGP](#task-2-instant-ngp)
- [Task 3: 3D Gaussian Splatting](#task-3-3d-gaussian-splatting)

# Task 1: NeRF
##  Data 
Refer to this https://blog.csdn.net/qq_45913887/article/details/132731884
##  Training
```
python run_nerf.py --config configs/your_config.txt
```
##  Evalution
```
 python run_nerf.py --config configs/your_config.txt --render_only
```
##  Visualization
```
tensorboard --logdir ./logs/summaries/
```
Then open http://localhost:6006/ in your browser

### 🎥 NeRF Demo








[▶️ Click to watch NeRF video](https://github.com/user-attachments/assets/368b53e1-4012-4522-8e25-b7dee9cc7778)




# Task 2: Instant NGP

# Task 3: 3D Gaussian Splatting
### 🎥 3D Gaussian Splatting Demo
[▶️ Click to watch Splatting video](https://github.com/user-attachments/assets/e17405e6-b9b6-462f-b15b-bc6f410266a3)
