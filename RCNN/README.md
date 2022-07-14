
# R-CNN复现

完整训练流程如下：

```
├── 一、准备数据
    训练集：VOC2012train
    验证集：VOC2012val
    测试集：VOC2007test

├── 二、Selective Search ：生成region proposals
    1.微调CNN使用的proposals
    2.训练SVM使用的proposals

├── 三、训练
    1.微调CNN
    2.训练SVM
        使用训练好的CNN作为特征提取器，训练20类SVM
    3.box校正
```

代码说明：
```
├── dataset_prepare
│   ├── selectivesearch.py      # SS算法，生成region proposals
│   ├── finetune_data.py        # 生成训练CNN使用的数据
│   ├── svm_data.py             # 生成训练SVM使用的数据
│   ├── util.py                 # 辅助函数
├── custom_finetune_data.py           # CNN数据加载器
├── custom_svm_data.py                # SVM数据加载器
├── custom_hnm_data.py                # 难负例挖掘，用于训练SVM
├── custom_batch_sampler.py           # 采样
├── train_finetune.py                 # 训练微调CNN
├── train_svm.py                      # 训练SVM
```