# Optimization the denoising autoencoder by genetic algorithm for anomaly detection (DAEGA)

针对用户用电异常分析，提出了基于遗传算法优化的DAE用电异常分析方法。

利用用电数据去训练去噪自编码器模型，根据模型重构数据和原始数据计算误差，与设定阈值进行比较确定是否发生用电异常

在两个不同的地区分别进行实验，最终模型的效果如下：

台区1:
- accuracy is:  0.8832876712328767
- f1 score is:  0.9376354160566844
- precision is:  0.8825928784037042

台区2:
- accuracy is:  0.892161911628386
- f1 score is:  0.9425577440061448
- precision is:  0.8913562311922798

更多细节见DAEGA.ipynb