# Frozen test protocol

本协议在读取测试标签前保存。测试仅用于一次最终评估，不按结果挑选种子、不调整参数。

- 候选：原始实验的 MAE 线性预测层输出目录中 seed 6201、6202、6203 的已保存检查点，全部为验证选择的 epoch 1。
- 特征：冻结 ResNet18 IMAGENET1K_V1，官方 weights.transforms()，fc=Identity。
- 标准化：每个 checkpoint 保存的训练集 feature_mean、feature_std。
- 对照：训练均值和训练中位数，只从 train.csv 计算。
- 测试：Day 5 预先固定的 test.csv（M2_L_V1_1），不参与拟合或模型选择。
- 报告：每种子与均值±种子间样本标准差的三轴平均 MAE、合并三轴 RMSE、各轴指标、完整时间曲线；不选最好种子。
- 测试录像只有一个，帧间相关；种子间波动不等于独立录像上的置信区间。
- 测试后该录像不再属于未见数据；若继续改进模型，需要新增独立录像做最终验证。
