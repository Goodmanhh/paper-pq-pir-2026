# Privacy-Preserving Vector Retrieval for RAG: An Experimental Evaluation


## 研究问题

RAG 系统使用向量检索，但现有隐私保护技术（PIR/SE）不支持最近邻语义。本项目通过实验回答：

1. 隐私保护向量检索比明文 ANN 慢多少？
2. 隐私保护是否降低检索质量（Recall@k）？
3. 查询与结果的通信开销是多少？

## 对比路线

| 路线 | 实现 | 角色 |
|---|---|---|
| 明文基线 | HNSW（hnswlib） | 性能上限 |
| HE 路线 | CKKS（TenSEAL）加密内积 | 隐私保护主流 |
| PIR 路线 | LWE 检索 | 查询隐私方案 |

## 目录

```
baseline/      明文 HNSW 基线
he_ckks/       CKKS 加密相似度
pir_lwe/       LWE-PIR 检索
data/          数据集与生成脚本
experiments/   实验脚本
results/       结果表格与图
docs/          AIRS 研究日志（构建证据）
```

## 复现

见各目录 README 与 experiments/ 下的脚本。
