import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import os

# 初始化总体真实值和预测值
true_all = []
pred_all = []

# 遍历每对文件，分别计算指标
for i in range(1, 19):
    # 读取真实值和预测值
    true_df = pd.read_csv(f'Cascaded_Retrieval-Augmented_Fine-Tuning/dataset/csv-data/test/location_{i}.csv')
    pred_df = pd.read_csv(f'Cascaded_Retrieval-Augmented_Fine-Tuning/test/result/CRAFT-Qwen3-4B/output{i}.csv',header=None)

    # 获取真实值（最后一列）并转换为布尔值
    true_values = true_df.iloc[:, -1].astype(bool).tolist()


    pred_values = (
    pred_df.iloc[:, 0]
    .str.extract(r'(?is)</think>\s*(.*)$', expand=False)
    .str.extract(
        r'(?is)the\s+answer\s+is.*?([01])',
        # r'(?is)the\s+answer\s+is\s*[:：]?\s*([01])\b',
        expand=False
    )
    .fillna(-1)
    .astype(int)
    .tolist()
    )


    print(true_values)
    print(pred_values)



#     #累积总体数据
    true_all.extend(true_values)
    pred_all.extend(pred_values)

#    每个文件的评估指标
    acc = accuracy_score(true_values, pred_values)
    rec = recall_score(true_values, pred_values, zero_division=0)
    prec = precision_score(true_values, pred_values, zero_division=0)

    # 计算混淆矩阵：TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(true_values, pred_values, labels=[False, True]).ravel()

    print(f"[location_{i}]")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")


# 总体指标
overall_acc = accuracy_score(true_all, pred_all)
overall_rec = recall_score(true_all, pred_all, zero_division=0)
overall_prec = precision_score(true_all, pred_all, zero_division=0)
tn, fp, fn, tp = confusion_matrix(true_all, pred_all, labels=[False, True]).ravel()

print("[Overall]")
print()
print(f"Accuracy:  {overall_acc:.4f}")
print(f"Recall:    {overall_rec:.4f}")
print(f"Precision: {overall_prec:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")





