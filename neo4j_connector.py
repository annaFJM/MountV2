from neo4j import GraphDatabase
import json

# 扁平化嵌套字典
def flatten_dict(data_dict, parent_key='', separator='_'):
    """
    递归扁平化嵌套字典
    """
    items = []
    for key, value in data_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            # 递归处理嵌套字典
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    
    return dict(items)

class Neo4jConnector:
    """Neo4j数据库连接和操作类"""
    
    def __init__(self, uri, user, password):
        """初始化数据库连接"""
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF")
            self.driver.verify_connectivity()
            print("✅ Neo4j 数据库连接成功！")
        except Exception as e:
            print(f"❌ Neo4j连接失败: {e}")
    
    def close(self):
        """关闭数据库连接"""
        if self.driver is not None:
            self.driver.close()
            print("🔌 Neo4j 数据库连接已关闭。")

    def get_outbound_by_id(self, id, relation):
        """
        获取指定关系的出边的节点id
        """
        if self.driver is None:
            return [],[],[]
        
        with self.driver.session() as session:
            try:
                query = f"""
                    MATCH (n)-[r:{relation}]->(m)
                    WHERE id(n) = {id}
                    RETURN id(m) AS id, m.name AS name, labels(m) AS label
                """
                result = session.run(query)
                target_ids = []
                target_name = []
                target_label = []
                for record in result:
                    target_ids.append(record["id"])
                    target_name.append(record["name"])
                    target_label.append(record["label"][0])

                return target_ids, target_name, target_label

            except Exception as e:
                print(f"❌ 获取节点labels时出错: {e}")
                return [], [], []
    
    def get_inbound_by_id(self, id, relation):
        """
        获取指定关系的出边的节点名称
        """
        if self.driver is None:
            return []
        
        with self.driver.session() as session:
            try:
                query = f"""
                    MATCH (n)-[r:{relation}]->(m)
                    WHERE id(m) = {id}
                    RETURN id(n) AS id, n.name AS name, labels(n) AS label
                """
                result = session.run(query)
                
                target_ids = []
                target_name = []
                target_label = []
                for record in result:
                    target_ids.append(record["id"])
                    target_name.append(record["name"])
                    target_label.append(record["label"][0])

                return target_ids[:5], target_name[:5], target_label[:5]
                # todo: 这里默认返回前5个，实际需要返回最接近的5个
            
            except Exception as e:
                print(f"❌ 获取节点labels时出错: {e}")
                return []
    
    def get_information_by_id(self, id):
        """
        获取指定id的名称信息，辅助LLM判断接下来去哪里
        """
        if self.driver is None:
            return []
        
        with self.driver.session() as session:
            try:
                query = f"""
                    MATCH (n)
                    WHERE id(n) = {id}
                    RETURN n.name
                """
                result = session.run(query)
                single_result = result.single()
                return single_result[0] if single_result else -1
            except Exception as e:
                print(f"❌ 获取节点labels时出错: {e}")
                return []

    
    def create_entity(self, data_dict):
        """
        对给定的材料创建一个entity对象,返回新对象的id
        """
        data_dict = flatten_dict(data_dict)

        with self.driver.session() as session:
            try:
                # 使用参数化查询
                query = """
                    CREATE (n:Entity $properties)
                    RETURN id(n)
                """
                print(query)
                result = session.run(query, properties=data_dict)
                single_result = result.single()
                return single_result[0] if single_result else -1
            except Exception as e:
                print(f"创建节点失败: {e}")
                return -1
    
    def create_relation(self, entity_id, material_id):
        """
        给entity对象和material对象创建关系，完成挂载
        """

        if self.driver is None:
            print("创建关系失败: 未连接到数据库")
            return 
        
        with self.driver.session() as session:
            try:
                query = f"""
                    MATCH (n), (m)
                    WHERE id(n) = {entity_id} AND id(m) = {material_id}
                    CREATE (n)-[r:isBelongTo]->(m)
                    RETURN n,r,m
                """
                session.run(query)
            except Exception as e:
                print(f"创建关系失败: {e}")
    
    def drop_relation(self, id):
        """
        删除关系
        """

        if self.driver is None:
            print("删除失败: 未连接到数据库")
            return 
        
        with self.driver.session() as session:
            try:
                query = f"""
                    MATCH (n)
                    WHERE id(n) = {id}
                    DETACH DELETE n
                """
                session.run(query)
                print("成功删除")
            except Exception as e:
                print(f"删除: {e}")


if __name__ == "__main__":
    import os
    import sys
    import unicodedata
    sys.path.append(os.getcwd())
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    neo4j = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    outbound_ids, outbound_names, outbound_labels = neo4j.get_outbound_by_id("0","include")
    print(outbound_ids)
    print(outbound_names)
    print(outbound_labels)
    inbound_ids, inbound_names, inbound_labels = neo4j.get_inbound_by_id("31","isBelongTo")
    print(inbound_ids[:10])
    print(inbound_names[:10])
    print(inbound_labels[:10])
    data_dict = {
        "_id": "67467b38f10142971f536726",
        "_meta_id": 15984631,
        "_tid": 1007,
        "data": {
            "成分": {
                "Fe" : 1.0,
                "Mn" : 1.0,
                "Co" : 0.5
            }
        }
    }
    new_id = neo4j.create_entity(data_dict)
    print(new_id)
    neo4j.create_relation(new_id, "25258")
    neo4j.drop_relation(new_id)
    info = neo4j.get_information_by_id("1")
    print(info)