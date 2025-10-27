import os
from neo4j import GraphDatabase

# --- 配置 (通常从 config.py 文件加载) ---
NEO4J_URI = "neo4j://10.77.50.200:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "thl123!@#"
OUTPUT_FILENAME = "neo4j_statistics.txt"

class Neo4jStatistics:
    """用于连接Neo4j并获取统计数据的类"""

    def __init__(self, uri, user, password):
        """初始化数据库驱动"""
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("✅ Neo4j 数据库连接成功！")
        except Exception as e:
            print(f"❌ Neo4j 连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j 数据库连接已关闭。")

    def get_single_count(self, query):
        """执行一个Cypher查询并返回单个计数值"""
        if not self.driver:
            return 0
        with self.driver.session() as session:
            result = session.run(query)
            single_record = result.single()
            return single_record[0] if single_record else 0

    def run_statistics(self):
        """执行所有统计查询并将结果写入文件"""
        print("--- 正在执行统计查询... ---")
        
        # 查询1: 总共多少Material节点
        total_materials_query = "MATCH (m:Material) RETURN count(m)"
        total_materials_count = self.get_single_count(total_materials_query)

        # 查询2: 连着“高熵合金”的多少Material节点
        hea_materials_query = "MATCH (m:Material)-[:isBelongTo]->(c:Class {name: '高熵合金'}) RETURN count(m)"
        hea_materials_count = self.get_single_count(hea_materials_query)

        # 查询3: 连着“高熵合金”的Material节点下有Entity节点的有多少个
        hea_materials_with_entity_query = """
            MATCH (e:Entity)-[:isBelongTo]->(m:Material)-[:isBelongTo]->(c:Class {name: '高熵合金'}) 
            RETURN count(DISTINCT m)
        """
        hea_materials_with_entity_count = self.get_single_count(hea_materials_with_entity_query)

        # 查询4: Entity >= 2 的有多少个
        hea_materials_with_2plus_entities_query = """
            MATCH (e:Entity)-[:isBelongTo]->(m:Material)-[:isBelongTo]->(c:Class {name: '高熵合金'})
            WITH m, count(e) AS entityCount
            WHERE entityCount >= 2
            RETURN count(m)
        """
        hea_materials_with_2plus_entities_count = self.get_single_count(hea_materials_with_2plus_entities_query)

        print("--- 查询完成，正在生成报告... ---")
        
        # 写入文件
        try:
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                f.write("Neo4j 知识图谱统计报告\n")
                f.write("="*40 + "\n")
                f.write(f"1. Material节点总数: {total_materials_count}\n")
                f.write("\n")
                f.write("--- “高熵合金” 相关统计 (Entity->Material->Class) ---\n")
                f.write(f"2. 连接到“高熵合金”的Material节点数: {hea_materials_count}\n")
                f.write(f"3. 连接到“高熵合金”且其下至少有1个Entity节点的Material数: {hea_materials_with_entity_count}\n")
                f.write(f"4. 连接到“高熵合金”且其下有 >= 2个Entity节点的Material数: {hea_materials_with_2plus_entities_count}\n")
            
            print(f"✅ 成功！统计数据已保存到: {os.path.abspath(OUTPUT_FILENAME)}")

        except Exception as e:
            print(f"❌ 写入文件失败: {e}")


def main():
    """主执行函数"""
    try:
        stats_generator = Neo4jStatistics(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        stats_generator.run_statistics()
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        if 'stats_generator' in locals() and stats_generator.driver:
            stats_generator.close()

if __name__ == "__main__":
    main()