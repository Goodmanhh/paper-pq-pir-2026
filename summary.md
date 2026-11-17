# 三条路线实验结果汇总（AIRS 生成）

| 指标 | 明文 HNSW | CKKS 加密内积 | LWE-PIR |
|---|---|---|---|
| 单查询延迟 | 0.2394 ms | 23.326 s | 103.9152 ms（服务器）+ 0.2206 s（客户端） |
| Recall@10 | 0.9852 | 1.0 | 精确下标（误差 0.000597） |
| 查询语义 | ANN 相似度 | ANN 相似度 | 仅精确下标 |
| 查询密文 | 明文 | 323.46 KB | LWE 样本 |
| 响应大小 | 10 条记录 | 1175.36 MB | 1 条记录 |
| 客户端存储 | 无 | CKKS 上下文+密钥 | LWE 密钥 |

图：latency_comparison.png、communication_comparison.png
