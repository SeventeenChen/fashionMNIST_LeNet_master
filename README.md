# fashionMNIST_LeNet5

## LeNet5 模型

# ![LeNet5 Model]()

## 模型参数

![Summary]()

## 优化函数

```
Adam
```

## 模型准确率

| 训练集准确率            | 验证集准确率            | 测试集准确率           |
| ----------------------- | ----------------------- | ---------------------- |
| <center>90.27%</center> | <center>90.27%</center> | <center>96.1%</center> |

## 模型训练

```python
CUDA_VISIBLE_DEVICE=0 python train.py
```

**CUDA_VISIBLE_DEVICE**对应显卡序号，本实验所用显卡为一块GeForce GTX 1080 Ti：

```python
nvidia-smi
```

训练时会自动保存验证集准确率较高的pth文件，以供测试

## 模型测试

由于本数据集没有区分测试集和验证集，故每次迭代随机从测试集中取batch_size的样本作为验证集，方便训练过程调整超参数

将测试集中的PATH改为对应保存的pth文件即可

```
CUDA_VISIBLE_DEVICE=0 python test.py
```

## TO DO

- [ ] README.md in English

## 参考

[ashmeet13](https://github.com/ashmeet13)

