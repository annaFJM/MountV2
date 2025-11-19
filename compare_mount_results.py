#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较挂载结果和真实标签的脚本
"""

import json
import os
import sys
import re
import csv


def read_jsonl(filepath):
    """读取JSONL文件并返回数据字典"""
    data_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                # 将data_id统一转换为字符串，以便匹配
                data_id = str(item['data_id'])
                data_dict[data_id] = item
    return data_dict


def compare_results(prediction_file, ground_truth_file, output_dir):
    """
    比较预测结果和真实标签
    
    Args:
        prediction_file: 预测结果文件路径
        ground_truth_file: 真实标签文件路径
        output_dir: 输出目录
    """
    # 读取数据
    print(f"读取预测结果文件: {prediction_file}")
    predictions = read_jsonl(prediction_file)
    
    print(f"读取真实标签文件: {ground_truth_file}")
    ground_truths = read_jsonl(ground_truth_file)
    
    # 提取文件名（不含扩展名）
    filename = os.path.basename(prediction_file)
    base_name = filename.replace('.jsonl', '')
    
    # 准备统计数据
    total_count = 0
    correct_count = 0
    wrong_count = 0
    null_count = 0
    
    # 准备CSV数据
    csv_data = []
    
    # 遍历所有预测结果
    for data_id in predictions:
        total_count += 1
        
        pred_item = predictions[data_id]
        pred_target = pred_item.get('target_node_id')
        
        # 获取真实标签
        if data_id in ground_truths:
            true_item = ground_truths[data_id]
            true_target = str(true_item.get('target_node_id')) if true_item.get('target_node_id') is not None else None
        else:
            true_target = None
        
        # 判断结果
        if pred_target is None or pred_target == 'null':
            result = '空值'
            null_count += 1
            pred_target_str = 'null'
        else:
            pred_target_str = str(pred_target)
            if true_target and pred_target_str == true_target:
                result = '正确'
                correct_count += 1
            else:
                result = '错误'
                wrong_count += 1
        
        # 添加到CSV数据
        csv_data.append({
            'data_id': data_id,
            'predicted_id': pred_target_str,
            'true_id': true_target if true_target else 'N/A',
            'result': result
        })
    
    # 计算正确率
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    # 计算除去空值的正确率
    non_null_count = total_count - null_count
    accuracy_without_null = (correct_count / non_null_count * 100) if non_null_count > 0 else 0
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入TXT文件
    txt_file = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"总条数: {total_count}\n")
        f.write(f"正确条数: {correct_count}\n")
        f.write(f"错误条数: {wrong_count}\n")
        f.write(f"空值条数: {null_count}\n")
        f.write(f"正确率: {accuracy:.2f}%\n")
        f.write(f"除去空值正确率: {accuracy_without_null:.2f}%\n")
    
    print(f"\n统计信息已保存到: {txt_file}")
    print(f"总条数: {total_count}")
    print(f"正确条数: {correct_count}")
    print(f"错误条数: {wrong_count}")
    print(f"空值条数: {null_count}")
    print(f"正确率: {accuracy:.2f}%")
    print(f"除去空值正确率: {accuracy_without_null:.2f}%")
    
    # 写入CSV文件
    csv_file = os.path.join(output_dir, f"{base_name}.csv")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['data_id', 'predicted_id', 'true_id', 'result'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"详细结果已保存到: {csv_file}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python compare_mount_results.py <prediction_file>")
        print("\n示例:")
        print("  python compare_mount_results.py /home/thl/2025Fall/LLM_Mount_KG/results/mount_result_20251027_134736.jsonl")
        sys.exit(1)
    
    prediction_file = sys.argv[1]
    ground_truth_file = "/home/thl/2025Fall/LLM_Mount_KG/V4/test_data_from_KG/ground_truth.jsonl"
    output_dir = "/home/thl/2025Fall/LLM_Mount_KG/mount_results"
    
    # 检查文件是否存在
    if not os.path.exists(prediction_file):
        print(f"错误: 预测文件不存在: {prediction_file}")
        sys.exit(1)
    
    if not os.path.exists(ground_truth_file):
        print(f"错误: 真实标签文件不存在: {ground_truth_file}")
        sys.exit(1)
    
    # 执行比较
    compare_results(prediction_file, ground_truth_file, output_dir)
    print("\n处理完成!")


if __name__ == "__main__":
    main()