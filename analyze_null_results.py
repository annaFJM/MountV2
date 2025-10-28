#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析空值的挂载结果，从Neo4j和log文件中提取信息
"""

import csv
import os
import sys
import re
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


def extract_date_from_filename(filename):
    """从文件名中提取日期，例如 mount_result_20251027_134736.csv -> 20251027-134736"""
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        return f"{date_part}-{time_part}"
    return None


def extract_qwen_output_from_log(log_file, data_id):
    """
    从log文件中提取指定data_id的补充信息（qwen_output）
    
    Args:
        log_file: log文件路径
        data_id: 要查找的data_id
    
    Returns:
        补充信息字符串，如果未找到返回None
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找包含该data_id的部分
        for i, line in enumerate(lines):
            # 查找 'identity': data_id 的行
            if f"'identity': {data_id}" in line:
                # 向后查找补充信息
                for j in range(i, min(i + 20, len(lines))):
                    if '**补充信息**:' in lines[j] or '- **补充信息**:' in lines[j]:
                        # 提取补充信息内容
                        info_line = lines[j]
                        # 使用正则提取补充信息后的内容
                        match = re.search(r'\*\*补充信息\*\*:\s*(.+)', info_line)
                        if match:
                            return match.group(1).strip()
        
        return None
    except Exception as e:
        print(f"读取log文件时出错: {e}")
        return None


def analyze_null_results(csv_file, output_dir, log_base_dir):
    """
    分析空值的挂载结果
    
    Args:
        csv_file: 输入的CSV文件路径
        output_dir: 输出目录
        log_base_dir: log文件基础目录
    """
    # 提取文件名和日期
    filename = os.path.basename(csv_file)
    base_name = filename.replace('.csv', '')
    
    # 从文件名提取日期
    date_str = extract_date_from_filename(filename)
    if not date_str:
        print("错误: 无法从文件名中提取日期")
        sys.exit(1)
    
    # 构建log文件路径
    log_file = os.path.join(log_base_dir, f"{date_str}.log")
    if not os.path.exists(log_file):
        print(f"警告: log文件不存在: {log_file}")
        log_file = None
    
    # 连接Neo4j
    print("正在连接Neo4j数据库...")
    connector = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_file}")
    null_results = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['result'] == '空值':
                null_results.append(row)
    
    print(f"找到 {len(null_results)} 条空值记录")
    
    # 查询节点名称和提取log信息
    output_data = []
    for i, row in enumerate(null_results, 1):
        data_id = row['data_id']
        predicted_id = row['predicted_id']
        true_id = row['true_id']
        
        print(f"正在处理 {i}/{len(null_results)}: data_id={data_id}")
        
        # 查询Entity节点名称（data_id对应Entity）
        data_name = connector.get_entity_name(data_id)
        
        # 查询Material节点名称（true_id对应Material）
        true_name = connector.get_material_name(true_id) if true_id != 'N/A' else None
        
        # 从log文件提取qwen_output
        qwen_output = None
        if log_file:
            qwen_output = extract_qwen_output_from_log(log_file, data_id)
        
        output_data.append({
            'data_id': data_id,
            'data_name': data_name if data_name else 'N/A',
            'predicted_id': predicted_id,
            'true_id': true_id,
            'true_name': true_name if true_name else 'N/A',
            'result': '空值',
            'qwen_output': qwen_output if qwen_output else 'N/A'
        })
    
    # 关闭连接
    connector.close()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入CSV文件
    output_file = os.path.join(output_dir, f"null_{base_name}.csv")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['data_id', 'data_name', 'predicted_id', 'true_id', 
                     'true_name', 'result', 'qwen_output']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    
    print(f"\n空值结果分析完成！")
    print(f"输出文件: {output_file}")
    print(f"共处理 {len(output_data)} 条空值记录")


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python analyze_null_results.py <csv_file>")
        print("\n示例:")
        print("  python analyze_null_results.py /home/thl/2025Fall/LLM_Mount_KG/mount_results/mount_result_20251027_134736.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = os.path.dirname(csv_file)
    log_base_dir = "/home/thl/2025Fall/LLM_Mount_KG/log"
    
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        sys.exit(1)
    
    analyze_null_results(csv_file, output_dir, log_base_dir)


if __name__ == "__main__":
    main()