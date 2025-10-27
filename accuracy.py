#!/usr/bin/env python
import json
import os

# --- 1. 定义文件路径 ---

# 您的 "结果" 文件 (来自 LLM 挂载)
RESULT_FILE_PATH = "/home/thl/2025Fall/LLM_Mount_KG/results/mount_result_20251027_082525.jsonl"

# 您的 "答案" 文件 (来自 generate_test_data.py)
GROUND_TRUTH_PATH = "/home/thl/2025Fall/LLM_Mount_KG/test/data/ground_truth.jsonl"


def load_ground_truth(filepath):
    """
    加载 "答案" 文件到一个字典中。
    键: data_id (强制转为 str)
    值: target_node_id (强制转为 str)
    """
    print(f"--- 正在加载“答案”文件: {filepath} ---")
    ground_truth_map = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    # 关键：将 data_id 和 target_node_id 都转为 str，以实现统一比较
                    data_id = str(record.get("data_id"))
                    target_id = str(record.get("target_node_id"))
                    
                    if data_id != "None":
                        ground_truth_map[data_id] = target_id
                        
                except json.JSONDecodeError:
                    print(f"⚠️ 警告：跳过“答案”文件中的无效 JSON 行: {line}")

    except FileNotFoundError:
        print(f"❌ 错误：“答案”文件未找到: {filepath}")
        return None
    
    print(f"✅ “答案”加载完毕。共 {len(ground_truth_map)} 条记录。")
    return ground_truth_map


def evaluate_results(result_filepath, ground_truth_map):
    """
    逐行读取结果文件，并与“答案”字典进行比对。
    """
    print(f"--- 正在评估结果文件: {result_filepath} ---")
    
    total_records = 0
    correct_records = 0
    
    try:
        with open(result_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                total_records += 1
                try:
                    result_record = json.loads(line)
                    
                    # 关键：将 data_id 和 target_node_id 都转为 str
                    data_id = str(result_record.get("data_id"))
                    # str(None) 会变成 'None'
                    predicted_target_id = str(result_record.get("target_node_id"))

                    # 检查“答案”是否存在
                    if data_id not in ground_truth_map:
                        print(f"⚠️ 警告：data_id {data_id} 在结果文件中，但不在“答案”文件中。跳过...")
                        continue

                    # 获取正确答案
                    correct_target_id = ground_truth_map[data_id]

                    # 比较
                    if predicted_target_id == correct_target_id:
                        correct_records += 1
                    else:
                        print(f"❌ 错误: data_id {data_id}")
                        print(f"   - 预测 (Prediction): {predicted_target_id}")
                        print(f"   - 答案 (Ground Truth): {correct_target_id}")

                except json.JSONDecodeError:
                    print(f"⚠️ 警告：跳过结果文件中的无效 JSON 行: {line}")
                    
    except FileNotFoundError:
        print(f"❌ 错误：结果文件未找到: {result_filepath}")
        return

    # --- 4. 打印最终报告 ---
    print("\n" + "="*30)
    print("--- 评估结果报告 ---")
    print(f"总处理记录 (Total Records): {total_records}")
    print(f"正确挂载 (Correct): {correct_records}")
    
    incorrect_records = total_records - correct_records
    print(f"错误挂载 (Incorrect): {incorrect_records}")

    if total_records > 0:
        accuracy = (correct_records / total_records) * 100
        print(f"\n正确率 (Accuracy): {accuracy:.2f} %")
    else:
        print("\n正确率 (Accuracy): N/A (没有找到记录)")
    print("="*30)


if __name__ == "__main__":
    # 1. 加载答案
    truth_map = load_ground_truth(GROUND_TRUTH_PATH)
    
    if truth_map is not None:
        # 2. 评估结果
        evaluate_results(RESULT_FILE_PATH, truth_map)