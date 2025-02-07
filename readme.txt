'UTF-8'

改善了特征提取方法，将特征提取步骤集成至训练与测试代码中。

模型文件 moirePatternCNN_che.h5 利用 摩尔纹视频图片 与 10001training 训练得到。

可直接运行 train.py 训练模型，新训练得到的模型文件将以 moirePatternCNN.h5 保存。

可直接运行运行 test.py测试模型。测试阶段若判断出摩尔纹图片，其最可能出现摩尔纹的位置会被框出并保存至'./moire pattern'文件夹。

