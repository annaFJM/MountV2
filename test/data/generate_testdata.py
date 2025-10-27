#!/usr/bin/env python
# generate_test_data.py

import os
import json
from neo4j import GraphDatabase
from tqdm import tqdm

# 假设 config.py 在同一个目录下
try:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
except ImportError:
    print("❌ 错误：无法导入 config.py。请确保该文件存在并且与此脚本在同一目录中。")
    exit(1)

# --- 输出文件配置 ---
# 1. test.json: 用于 main_v2.py 的输入
TEST_JSON_PATH = "test.json"
# 2. embedding_source_data.json: 用于生成向量库的数据
EMBEDDING_SOURCE_PATH = "embedding_source_data.json"
# 3. ground_truth.jsonl: 用于验证结果的“答案”
GROUND_TRUTH_PATH = "ground_truth.jsonl"


def deflatten_dict(flat_dict):
    """
    将从Neo4j获取的扁平化属性字典 "反扁平化" 为嵌套的JSON结构。
    例如: 'data_成分比重_Mn' -> data: {'成分比重': {'Mn': ...}}
    
    *** 注意：对于没有 'data_' 前缀的键 (如 'name', '工艺'), 
    *** 它们会保持在顶层，这符合您当前的需求。
    """
    nested_dict = {}
    data_sub_dict = {}

    for key, value in flat_dict.items():
        if key in ('_id', '_meta_id', '_tid'):
            # 1. 处理根级别元数据
            nested_dict[key] = value
        elif key.startswith('data_'):
            # 2. 处理所有 'data_' 开头的键
            # 去掉 'data_' 前缀
            stripped_key = key[5:] 
            parts = stripped_key.split('_')
            
            current_level = data_sub_dict
            
            # 遍历所有部分，除了最后一个
            for part in parts[:-1]:
                # .setdefault() 会获取键的值，如果键不存在，则插入该键并设置默认值
                current_level = current_level.setdefault(part, {})
            
            # 设置最后一个部分的值
            current_level[parts[-1]] = value
        else:
            # 3. 处理其他可能的根级别键 (例如 'name', 'id', '工艺' 等)
            nested_dict[key] = value

    # 即使没有 'data_' 键，也会添加一个空的 'data' 字典
    # 下游脚本 (generate_embeddings) 会处理掉这个空字典
    nested_dict['data'] = data_sub_dict
    return nested_dict


def main():
    """
    主执行函数：连接 Neo4j, 查询数据, 反扁平化, 并写入三个输出文件。
    """
    
    # 1. 定义查询
    # 这个查询找到所有连接到'高熵合金'且有>=2个Entity的Material
    # 然后为每个Material返回两个不同Entity的属性，以及Material的ID
    
    # (查询保持不变，继续使用 id() 来获取整数ID)
    cypher_query = """
    MATCH (c:Class {name: '高熵合金'})<-[:isBelongTo]-(m:Material)
    MATCH (e:Entity)-[:isBelongTo]->(m)
    WITH m, count(e) AS entityCount
    WHERE entityCount >= 2
    WITH m
    MATCH (e:Entity)-[:isBelongTo]->(m)
    WITH m, collect(e) AS entities
    RETURN 
        id(m) AS material_id,
        properties(entities[0]) AS entity_props_for_test,
        id(entities[0]) AS entity_identity_for_test, 
        properties(entities[1]) AS entity_props_for_embedding,
        id(entities[1]) AS entity_identity_for_embedding
    """

    driver = None
    test_set_data = []
    embedding_set_data = []
    ground_truth_records = []

    try:
        # 2. 连接数据库并查询
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Neo4j 数据库连接成功！")
        
        with driver.session() as session:
            print(f"--- 正在执行查询，查找所有连接到'高熵合金'且有 >= 2 个 Entity 的 Material... ---")
            result = session.run(cypher_query)
            
            # 使用 tqdm 显示进度条
            records = list(result)
            print(f"--- 查询完毕！共找到 {len(records)} 条符合条件的 Material 节点。开始处理数据... ---")

            # 3. 处理数据
            for record in tqdm(records, desc="处理节点数据"):
                
                # 捕获所有需要的 ID
                flat_test_props = record['entity_props_for_test']
                test_identity = record['entity_identity_for_test'] # 整数 ID
                
                flat_embedding_props = record['entity_props_for_embedding']
                embedding_identity = record['entity_identity_for_embedding'] # 整数 ID
                
                material_id_str = str(record['material_id']) # 整数 ID 转为字符串

                # 反扁平化数据 (这会保留 'name', 'id' 等在顶层)
                nested_test_entity = deflatten_dict(flat_test_props)
                nested_embedding_entity = deflatten_dict(flat_embedding_props)

                # 将整数 identity 添加到字典中
                nested_test_entity['identity'] = test_identity
                nested_embedding_entity['identity'] = record['material_id']


                # 添加到各自的列表中
                test_set_data.append(nested_test_entity)
                embedding_set_data.append(nested_embedding_entity)
                
                # --- ⭐ 唯一修改点在这里 ---
                # 创建“答案”记录
                # 按照你的要求，data_id 使用 test_identity (整数ID)
                ground_truth_records.append({
                    "data_id": str(test_identity), # 使用 test.json 中对应条目的整数ID
                    "target_node_id": material_id_str, # 使用 Material 节点的整数ID
                    "relation_type": "isBelongTo"
                })

        # 4. 写入输出文件
        print(f"\n--- 数据处理完毕，开始写入文件... ---")

        # 写入 test.json
        with open(TEST_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(test_set_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 成功！测试集已保存到: {TEST_JSON_PATH} (共 {len(test_set_data)} 条记录)")

        # 写入 embedding_source_data.json
        with open(EMBEDDING_SOURCE_PATH, 'w', encoding='utf-8') as f:
            json.dump(embedding_set_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 成功！Embedding源数据已保存到: {EMBEDDING_SOURCE_PATH} (共 {len(embedding_set_data)} 条记录)")

        # 写入 ground_truth.jsonl (注意是 .jsonl, 每行一个JSON对象)
        with open(GROUND_TRUTH_PATH, 'w', encoding='utf-8') as f:
            for record in ground_truth_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"✅ 成功！“答案”文件已保存到: {GROUND_TRUTH_PATH} (共 {len(ground_truth_records)} 条记录)")

        print("\n--- 所有任务完成！ ---")

    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {e}")
    finally:
        # 5. 关闭连接
        if driver:
            driver.close()
            print("🔌 Neo4j 数据库连接已关闭。")

if __name__ == "__main__":
    # 您可能需要先安装 tqdm: pip install tqdm
    main()