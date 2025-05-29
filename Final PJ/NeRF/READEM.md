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