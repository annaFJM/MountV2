import os
import pandas as pd
import logging
from neo4j_connector import Neo4jConnector
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from chem_utils import extract_elements_from_text

# 缓存文件夹路径
CACHE_DIR = "Class_material_info"

def generate_class_material_info():
    """
    遍历所有Class节点，如果有Material子节点，生成对应的CSV缓存
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    logging.info("--- 开始预处理：生成 Class-Material 映射缓存 ---")
    neo4j = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    if neo4j.driver is None:
        logging.error("❌ 数据库连接失败，无法预处理。")
        return

    with neo4j.driver.session() as session:
        # 1. 查找所有挂载了 Material 的 Class 节点
        # 查询逻辑：找到所有作为 isBelongTo 目标的 Class 节点
        query_classes = """
            MATCH (m:Material)-[:isBelongTo]->(c:Class)
            RETURN DISTINCT id(c) as class_id, c.name as class_name
        """
        result = session.run(query_classes)
        classes = [(record["class_id"], record["class_name"]) for record in result]
        
        logging.info(f"发现 {len(classes)} 个包含 Material 的 Class 节点。")
        
        for class_id, class_name in classes:
            file_path = os.path.join(CACHE_DIR, f"{class_id}_material_info.csv")
            
            # 查询该 Class 下所有的 Material
            query_materials = """
                MATCH (n:Material)-[:isBelongTo]->(c:Class)
                WHERE id(c) = $cid
                RETURN id(n) as id, n.name as name, labels(n) as labels
            """
            mat_result = session.run(query_materials, cid=class_id)
            
            data_list = []
            for record in mat_result:
                name = record["name"]
                # 提取元素集合
                el_set = extract_elements_from_text(name)
                
                data_list.append({
                    "ID": record["id"],
                    "Name": name,
                    "Label": ";".join(record["labels"]),
                    "set": str(el_set), # 存为字符串方便读取后 eval
                    "element_count": len(el_set)
                })
            
            if data_list:
                df = pd.DataFrame(data_list)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                logging.info(f"✅ 已生成缓存: {file_path} (包含 {len(df)} 条数据)")
    
    neo4j.close()
    logging.info("--- 预处理完成 ---")

if __name__ == "__main__":
    # 手动运行时配置日志
    logging.basicConfig(level=logging.INFO)
    generate_class_material_info()