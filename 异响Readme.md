运行方式：



pip install numpy scipy scikit-learn pandas matplotlib joblib



python abnormal\_sound\_fingerprint\_simulation.py --out ./runs/asf\_demo



加大仿真数据量：



python abnormal\_sound\_fingerprint\_simulation.py \\

&#x20; --n-per-class 300 \\

&#x20; --n-unknown 200 \\

&#x20; --duration 2.0 \\

&#x20; --fs 16000 \\

&#x20; --out ./runs/asf\_big



脚本里已经实现了这些模块：



异响数据模拟

normal：正常运行声

friction：摩擦异响

knock：敲击/碰撞异响

squeal：啸叫异响

looseness：松动异响

leak：泄漏异响

bearing：轴承周期冲击异响

resonance：共振异响

scrape：刮擦异响

unknown：不在库未知异响，用于开放集拒识测试

三层级特征框架

Level 1：直接测量特征

包括 RMS、峰值、峰峰值、峭度、偏度、过零率、包络统计、谱质心、谱带宽、谱平坦度、谱熵、滚降频率、主峰频率、频带能量比、自相关峰、包络谱峰等。

Level 2：降维变换特征

使用 log-STFT 谱图和包络谱向量，然后通过 PCA 得到低维时频嵌入。

Level 3：融合指纹特征

将直接特征、直接特征 PCA、时频 PCA embedding 拼接并标准化，形成最终 fusion fingerprint。

异响指纹库

每类异响建立类别原型 centroid。

保存类内方差 diagonal covariance。

计算类内马氏距离阈值。

保存直接特征均值、标准差、典型样本、边界样本。

输出为 fingerprint\_library.json。

识别方法

随机森林已知类分类器。

类别概率 + 指纹原型距离融合判决。

开放集拒识：当概率、融合分数或原型距离不满足阈值时，输出 unknown。

结果导出

运行后输出目录中会生成：

fingerprint\_library.json              # 异响指纹库

asf\_model\_bundle.joblib                # 特征管线 + 分类器 + 指纹库

open\_set\_predictions.csv               # 每条样本的识别明细

metrics.json                           # 准确率、F1、未知类召回率等指标

classification\_report\_closed\_set.txt   # 已知类闭集分类报告

classification\_report\_open\_set.txt     # 开放集识别报告

direct\_feature\_importance.csv          # 直接测量特征重要性

fusion\_feature\_importance.csv          # 融合特征重要性

plots/                                 # 混淆矩阵、指纹空间 PCA 图、特征重要性图

examples\_wav/                          # 每类仿真 wav 样例



脚本中还包含一个单条音频识别函数：



result = recognize\_new\_waveform(

&#x20;   x=waveform,

&#x20;   fs=16000,

&#x20;   model\_bundle\_path="./runs/asf\_demo/asf\_model\_bundle.joblib"

)



print(result.predicted\_label)

print(result.best\_known\_label)

print(result.confidence)

print(result.reject\_reasons)



真实数据接入时，主要替换脚本中的 simulate\_dataset() 部分，把真实 wav 读取后送入：



extract\_direct\_features()

extract\_transform\_features()

ThreeLevelFeaturePipeline

FingerprintLibrary



整体识别框架和指纹库结构不用改。

