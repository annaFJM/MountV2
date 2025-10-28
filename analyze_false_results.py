#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析错误的挂载结果，查询Neo4j获取节点名称
"""

import csv
import os
import sys
from neo4j import GraphDatabase


# Neo4j 数据库配置
NEO4J_URI = "neo4j://10.77.50.200:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "thl123!@#"


class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def get_entity_name(self, entity_id):
        """根据Entity的id查询name"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) WHERE id(n) = $entity_id RETURN n.name AS name",
                entity_id=int(entity_id)
            )
            record = result.single()
            return record["name"] if record else None
    
    def get_material_name(self, material_id):
        """根据Material的id查询name"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:Material) WHERE id(n) = $material_id RETURN n.name AS name",
                material_id=int(material_id)
            )
            record = result.single()
            return record["name"] if record else None


def analyze_false_results(csv_file, output_dir):
    """
    分析错误的挂载结果
    
    Args:
        csv_file: 输入的CSV文件路径
        output_dir: 输出目录
    """
    # 提取文件名
    filename = os.path.basename(csv_file)
    base_name = filename.replace('.csv', '')
    
    # 连接Neo4j
    print("正在连接Neo4j数据库...")
    connector = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_file}")
    false_results = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['result'] == '错误':
                false_results.append(row)
    
    print(f"找到 {len(false_results)} 条错误记录")
    
    # 查询节点名称
    output_data = []
    for i, row in enumerate(false_results, 1):
        data_id = row['data_id']
        predicted_id = row['predicted_id']
        true_id = row['true_id']
        
        print(f"正在处理 {i}/{len(false_results)}: data_id={data_id}")
        
        # 查询Entity节点名称（data_id对应Entity）
        data_name = connector.get_entity_name(data_id)
        
        # 查询Material节点名称（predicted_id和true_id对应Material）
        predicted_name = connector.get_material_name(predicted_id) if predicted_id != 'null' else None
        true_name = connector.get_material_name(true_id) if true_id != 'N/A' else None
        
        output_data.append({
            'data_id': data_id,
            'data_name': data_name if data_name else 'N/A',
            'predicted_id': predicted_id,
            'predicted_name': predicted_name if predicted_name else 'N/A',
            'true_id': true_id,
            'true_name': true_name if true_name else 'N/A',
            'result': '错误'
        })
    
    # 关闭连接
    connector.close()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入CSV文件
    output_file = os.path.join(output_dir, f"false_{base_name}.csv")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['data_id', 'data_name', 'predicted_id', 'predicted_name', 
                     'true_id', 'true_name', 'result']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    
    print(f"\n错误结果分析完成！")
    print(f"输出文件: {output_file}")
    print(f"共处理 {len(output_data)} 条错误记录")


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python analyze_false_results.py <csv_file>")
        print("\n示例:")
        print("  python analyze_false_results.py /home/thl/2025Fall/LLM_Mount_KG/mount_results/mount_result_20251027_134736.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = os.path.dirname(csv_file)
    
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        sys.exit(1)
    
    analyze_false_results(csv_file, output_dir)


if __name__ == "__main__":
    main()