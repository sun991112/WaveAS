# 最终版

现在目录只保留 4 个核心文件：

- [main.py](</C:/Users/sunyi/Desktop/DTI-1.0/最终版/main.py>)
  - 唯一入口
  - 串起特征生成、训练、交叉验证、结果保存

- [features.py](</C:/Users/sunyi/Desktop/DTI-1.0/最终版/features.py>)
  - 论文 2.2 的特征生成
  - 包括：
    - ChemBERTa 药物语义特征
    - ESM 蛋白语义特征与 residue embedding
    - GINE 药物结构编码
    - backbone + spatial/contact 蛋白结构编码
    - 属性重构
    - PageRank pooling
    - PCA 压缩

- [wavegc_model.py](</C:/Users/sunyi/Desktop/DTI-1.0/最终版/wavegc_model.py>)
  - 论文 2.3 / 2.4 的模型
  - 包括：
    - `SineEncoding`
    - `EigenEncoding`
    - `WaveGCSpectralGenerator`
    - `WaveGCSpectralBlock`
    - `WaveGCSpectralLinkPredictor`

- [trainer.py](</C:/Users/sunyi/Desktop/DTI-1.0/最终版/trainer.py>)
  - 训练与评估流程
  - 包括：
    - DTI 数据读取
    - 特征对齐
    - 相似图构建
    - 负采样
    - 光谱上下文构建
    - 交叉验证
    - 训练 loop
    - 指标计算

## 运行

只跑特征生成：

```bash
py -3 最终版/features.py --dataset davis --output_path outputs/davis_paper_features.npz
```

完整训练：

```bash
py -3 最终版/main.py --dataset davis --file_path Data/dti_lists/davis/dti.csv --node_feature_path outputs/davis_paper_features.npz --output_dir outputs/davis_paper_run
```

如果不传 `--node_feature_path`，`main.py` 会先自动生成特征。
