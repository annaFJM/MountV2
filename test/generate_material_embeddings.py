#!/usr/bin/env python
# generate_embeddings_from_file_v3.py

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# --- 1. 配置 ---

# 输入文件路径 (您的实际文件路径)
INPUT_JSON_PATH = "/home/thl/2025Fall/LLM_Mount_KG/test/data/embedding_source_data.json"

# 模型路径 (来自你的参考代码)
MODEL_PATH = "/home/thl/models/Qwen3-4B-clustering_1078/checkpoint-936"

# 输出目录 (您可以根据需要修改)
OUTPUT_DIR = "data"
OUTPUT_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "material_embeddings.npy")
OUTPUT_METADATA_PATH = os.path.join(OUTPUT_DIR, "material_metadata.json")

# 推理设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data_from_json(json_path):
    """
    从指定的 JSON 文件加载源数据。
    """
    print(f"--- 正在从 {json_path} 加载数据... ---")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        print(f"--- 成功加载 {len(records)} 条记录。 ---")
        return records
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件: {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ 错误：文件 {json_path} 不是有效的 JSON 格式。")
        return None
    except Exception as e:
        print(f"❌ 加载 JSON 文件时发生未知错误: {e}")
        return None

def format_data_for_embedding(record):
    """
    将 JSON 记录格式化为模型所需的文本。
    此函数会：
    1. 提取 'name' 字段。
    2. 删除 'id', '来源', 'identity' 和 'data' 字段。
    3. 将所有剩余字段格式化为属性字符串。
    """
    
    # 复制一份，避免修改原始数据
    props_copy = record.copy()

    # 提取 'name'，并将其从字典中移除
    material_name = props_copy.pop("name", "未知牌号") 

    # *** 这是关键修改点 ***
    # 显式删除 'id' (UUID), '来源', 'data', 和 'identity' (整数ID)
    # 这样它们就不会被包含在用于 embedding 的 "属性" 字符串中
    props_copy.pop("id", None)       
    props_copy.pop("来源", None) 
    props_copy.pop("data", None)     
    props_copy.pop("identity", None) # <--- 新增：也移除 identity

    # 格式化所有剩余的属性 (如 "工艺", "成分", "结构" 等)
    props_str = ", ".join([f"{k}: {v}" for k, v in props_copy.items() if v is not None and v != ""])
    
    # --- ⭐ 修改点 1: 更改指令文本格式 ---
    # 移除 "材料名称: " 前缀，直接以 name 开头
    instruction_text = f"{material_name}, 材料属性: {props_str}"
    
    return instruction_text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载模型
    print(f"--- 正在从 {MODEL_PATH} 加载模型到 {DEVICE} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto"
        ).to(DEVICE)
        model.eval()
        print("✅ 模型加载成功。")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("   请检查 MODEL_PATH 是否正确以及模型文件是否完整。")
        return

    # 2. 从 JSON 文件加载数据
    source_records = load_data_from_json(INPUT_JSON_PATH)

    if not source_records:
        print("❌ 未获取到任何数据，终止运行。")
        return

    embeddings_list = []
    metadata_list = []

    # 3. 生成 Embeddings
    print("--- 开始生成 Embeddings... ---")
    with torch.no_grad():
        for record in tqdm(source_records, desc="生成向量"):
            
            # --- ⭐ 修改点 2: 检查 'identity' 字段 ---
            # 检查记录是否包含 'identity' 和 'name' 键
            if "identity" not in record or "name" not in record:
                print(f"⚠️ 警告：跳过一条记录，缺少 'identity' 或 'name'。记录: {record}")
                continue
                
            text = format_data_for_embedding(record)
            
            if not text:
                print(f"⚠️ 警告：跳过 'identity' {record.get('identity')}，无法生成文本。")
                continue

            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                
                # 使用 Mean Pooling
                embedding = last_hidden_state.mean(dim=1).squeeze()

                embeddings_list.append(embedding.cpu().numpy())
                
                # --- ⭐ 修改点 3: 更改元数据格式 ---
                # 使用 'identity' (整数ID) 作为主要标识符
                metadata_list.append({
                    "identity": record["identity"],
                    "name": record["name"],
                    "label": "Material" 
                })
            except Exception as e:
                print(f"\n❌ 处理 'identity' {record.get('identity')} 时出错: {e}")

    # 4. 保存结果
    if embeddings_list:
        final_embeddings = np.array(embeddings_list).astype(np.float32)
        np.save(OUTPUT_EMBEDDINGS_PATH, final_embeddings)
        print(f"\n✅ 成功！特征向量已保存至: {OUTPUT_EMBEDDINGS_PATH}")
        print(f"   向量矩阵形状: {final_embeddings.shape}")

        with open(OUTPUT_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=4)
        print(f"✅ 成功！元数据已保存至: {OUTPUT_METADATA_PATH}")
    else:
        print("⚠️ 未生成任何向量。")

    print("\n--- 离线Embedding生成任务完成 ---")

if __name__ == "__main__":
    main()