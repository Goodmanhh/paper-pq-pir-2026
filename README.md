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

## 实验内容

- **主对比实验**：三条路线的延迟、召回率、通信开销、客户端存储
- **规模扩展实验**：1万→10万条记录、128→256 维的性能变化
- **参数研究**：CKKS 加密阶数（4096/8192/16384）对延迟和响应大小的影响
- **ef-召回率权衡**：HNSW 搜索广度 ef 对召回率和延迟的影响
- **真实数据实验**：20 Newsgroups + TF-IDF 的三路线对比

## 目录

- `baseline/`：明文 HNSW 基线
- `he_ckks/`：CKKS 加密相似度（含参数研究）
- `pir_lwe/`：LWE-PIR 检索
- `experiments/`：扩展实验、真实数据实验、绘图脚本
- `results/`：实验数据（JSON）与结果图
- `docs/`：AIRS 研究日志（构建证据）

## 复现

```bash
python3 baseline/baseline_hnsw.py
python3 he_ckks/ckks_inner_product.py
python3 pir_lwe/lwe_pir.py
python3 experiments/scaling_experiments.py
python3 experiments/ef_tradeoff.py
python3 experiments/real_data_tfidf.py
python3 experiments/make_figures.py
python3 experiments/new_figures.py
