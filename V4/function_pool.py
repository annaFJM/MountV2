import json
import logging
import numpy as np
from numpy.linalg import norm
import os
import pandas as pd
from chem_utils import extract_elements_from_data_item, calculate_jaccard_similarity

class FunctionPool:
    def __init__(self, neo4j_connector, llm_client, material_embeddings, material_metadata):
        """
        初始化功能池，注入依赖项
        :param neo4j_connector: Neo4j连接实例
        :param llm_client: OpenAI/Dashscope 客户端实例
        :param material_embeddings: 加载的 numpy 向量库
        :param material_metadata: 加载的元数据列表
        """
        self.neo4j = neo4j_connector
        self.client = llm_client
        self.material_embeddings = material_embeddings
        self.material_metadata = material_metadata
        self.cache_dir = "Class_material_info"

        self.id_to_index_map = {}
        if self.material_metadata:
            for idx, item in enumerate(self.material_metadata):
                nid = item.get('identity') 
                if nid is not None:
                    self.id_to_index_map[nid] = idx

    def get_include_outbound(self, id):
        """获取所有include出边并格式化信息"""
        outbound_ids, outbound_names, outbound_labels = self.neo4j.get_outbound_by_id(id, "include")
        result_list = []
        
        for i in range(len(outbound_ids)):
            _, extra_out_names, __ = self.neo4j.get_outbound_by_id(outbound_ids[i], "include")
            _, extra_in_names, __ = self.neo4j.get_inbound_by_id(outbound_ids[i], "isBelongTo")
            
            info = f"- 选项{chr(ord('A')+i)}: {outbound_names[i]},id为{outbound_ids[i]},节点属性为{outbound_labels[i]}"
            
            # 补充include出边信息
            if len(extra_out_names) > 0:
                info += "该下游节点代表性的include出边有："
                info += ",".join(extra_out_names[:3]) + ","
            else:
                info += "该下游节点没有include出边。"
            
            # 补充isBelongTo入边信息
            if len(extra_in_names) > 0:
                info += "该下游节点代表性的isBelongTo入边有："
                info += ",".join(extra_in_names[:3]) + ","
            else:
                info += "该下游节点没有isBelongTo入边。"
            
            result_list.append(info)

        return "\n".join(result_list)

    def format_isbelongto_inbound(self, inbound_ids, inbound_names, inbound_labels):
        """
        格式化 isBelongTo 入边信息 (原 get_isbelongto_inbound)
        注意：只处理前5条数据
        """
        result_list = []
        
        for i in range(min(len(inbound_ids), 5)):
            _, extra_out_names, __ = self.neo4j.get_outbound_by_id(inbound_ids[i], "include")
            _, extra_in_names, __ = self.neo4j.get_inbound_by_id(inbound_ids[i], "isBelongTo")
            
            info = f"- 选项{chr(ord('A')+i)}: {inbound_names[i]},id为{inbound_ids[i]},节点属性为{inbound_labels[i]}"
            
            if len(extra_out_names) > 0:
                info += "该下游节点代表性的include出边有："
                info += ",".join(extra_out_names[:3]) + ","
            else:
                info += "该下游节点没有include出边。"
                
            if len(extra_in_names) > 0:
                info += "该下游节点代表性的isBelongTo入边有："
                info += ",".join(extra_in_names[:3]) + ","
            else:
                info += "该下游节点没有isBelongTo入边。"
                
            result_list.append(info)
        
        return "\n".join(result_list)

    def calculate_cosine_similarity(self, vec_a, vec_b):
        """计算两个向量的余弦相似度"""
        if vec_a is None or vec_b is None:
            return 0.0
        norm_a = norm(vec_a)
        norm_b = norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def match_materials_by_elements(self, class_id, data_item):
        """
        【最终方案】基于两阶段过滤，返回向量相似度最高的 Top 5 名称（不带分数）。
        1. 找出元素相似度最高的所有候选。
        2. 在这些最高相似度中，找出向量相似度最高的 Top 5。
        """
        csv_path = os.path.join(self.cache_dir, f"{class_id}_material_info.csv")
        
        if not os.path.exists(csv_path):
            logging.warning(f"⚠️ 未找到节点 {class_id} 的元素缓存文件。")
            return None
            
        # 1. 提取输入数据的元素
        input_elements = extract_elements_from_data_item(data_item)
        if not input_elements:
            logging.info("⚠️ 输入数据无有效元素，跳过元素匹配。")
            return None 
            
        logging.info(f"--- 正在进行元素匹配 (筛选 Top 5 名称)，输入元素: {input_elements} ---")
        
        try:
            # 2. 读取与计算 Jaccard (Element Score)
            df = pd.read_csv(csv_path)
            df['set_obj'] = df['set'].apply(lambda x: eval(x) if isinstance(x, str) else set())
            df['elem_score'] = df['set_obj'].apply(lambda x: calculate_jaccard_similarity(input_elements, x))
            
            # 3. 【阶段一】找出最高的元素匹配度 Max Score 并过滤
            candidates = df[df['elem_score'] > 0].copy()
            if candidates.empty:
                logging.info("--- 元素匹配无结果 (无交集) ---")
                return None

            max_elem_score = candidates['elem_score'].max()
            # 过滤：只保留达到 Max Score 的候选
            best_elem_candidates = candidates[candidates['elem_score'] == max_elem_score].copy()
            
            logging.info(f"--- 元素匹配最高分: {max_elem_score:.4f}。共有 {len(best_elem_candidates)} 个候选进入第二阶段。---")

            # 4. 计算向量分数 (只计算 Stage 1 选出的候选)
            input_embedding = self.get_embedding_for_data(data_item)
            if input_embedding is not None:
                input_embedding = input_embedding.flatten()
            
            # 辅助函数：计算向量分数。无向量时返回 -1.0 作为最低分。
            def get_vec_score(row_id):
                node_id_int = int(row_id)
                if input_embedding is not None and self.material_embeddings is not None:
                    if node_id_int in self.id_to_index_map:
                        idx = self.id_to_index_map[node_id_int]
                        target_vec = self.material_embeddings[idx]
                        return self.calculate_cosine_similarity(input_embedding, target_vec)
                return -1.0 

            # 计算向量分数
            best_elem_candidates['vec_score'] = best_elem_candidates['ID'].apply(get_vec_score)
            
            # 5. 【阶段二】排序并取 Top 5 (优先向量分，其次元素分)
            
            # 排序：先按向量分降序，然后按元素分降序（确保即使是 -1.0 的也按元素分排序）
            final_matches = best_elem_candidates.sort_values(
                by=['vec_score', 'elem_score', 'ID'], 
                ascending=[False, False, True]
            ).head(5)

            # 6. 格式化输出 (只提供 Name, ID, Label, 无分数)
            result_list = []
            for i, (_, row) in enumerate(final_matches.iterrows()):
                node_id = row['ID']
                node_name = row['Name']
                node_label = row['Label']
                
                # 记录 Log，方便调试
                logging.info(f"   -> Top {i+1}: {node_name} (Elem: {row['elem_score']:.4f}, Vec: {row['vec_score']:.4f})")
                
                # 格式化输出为简单选项，交给 LLM 决定
                info = (f"- 选项{chr(ord('A')+i)}: {node_name}, id为{node_id}, 节点属性为{node_label}")
                result_list.append(info)
                
            if not result_list:
                 return None
                 
            return "\n".join(result_list)

        except Exception as e:
            logging.error(f"❌ 元素匹配过程中出错: {e}", exc_info=True)
            return None
        
    def get_embedding_for_data(self, data_item):
        """为传入的单个材料数据生成embedding (调用 API)"""
        if self.client is None:
            logging.error("❌ 运行时 Embedding 失败：API 客户端未初始化。")
            return None

        props_copy = data_item.copy()
        material_name = props_copy.pop("name", "未知牌号") 
        # 清理不需要 embedding 的字段
        for key in ["id", "来源", "data", "identity"]:
            props_copy.pop(key, None)
        
        props_str = ", ".join([f"{k}: {v}" for k, v in props_copy.items() if v is not None])
        text = f"{material_name}, 材料属性: {props_str}"
        
        try:
            completion = self.client.embeddings.create(
                model="text-embedding-v4",
                input=text
            )
            return np.array(completion.data[0].embedding)
        except Exception as e:
            logging.error(f"❌ 运行时 Embedding API 调用失败: {e}")
            return None

    def recall_top5_materials(self, current_node_id, data_item):
        """向量召回Top-5最相似的Material节点"""
        if self.material_embeddings is None or not self.material_metadata:
            return "向量召回功能不可用（预计算的向量库未加载）。请使用 get_include_outbound 或 get_isbelongto_inbound。"

        # 1. 生成查询向量
        try:
            query_embedding = self.get_embedding_for_data(data_item)
            if query_embedding is None:
                return "为新数据生成向量失败。请使用其他工具。"
        except Exception as e:
            logging.info(f"❌ 为新数据生成向量时出错: {e}")
            return f"为新数据生成向量时出错: {e}。请使用其他工具。"

        # 2. 计算余弦相似度
        query_embedding = query_embedding.flatten()
        query_norm = norm(query_embedding)
        if query_norm == 0:
            logging.info("⚠️ 警告: 查询向量的范数为0。")
            return "查询向量生成失败（范数为0）。请使用其他工具。"
            
        db_norms = norm(self.material_embeddings, axis=1)
        db_norms[db_norms == 0] = 1e-6 
        
        similarities = np.dot(self.material_embeddings, query_embedding) / (db_norms * query_norm)
        
        # 3. 获取Top-5
        top_5_indices = np.argsort(similarities)[-5:][::-1]
        
        # 4. 格式化结果
        result_list = []
        for i, idx in enumerate(top_5_indices):
            if idx < len(self.material_metadata):
                meta_item = self.material_metadata[idx] 
                sim_score = similarities[idx]
                
                node_id = meta_item.get("identity")
                node_name = meta_item.get("name")
                node_label = meta_item.get("label")
                
                info = f"- 选项{chr(ord('A')+i)} (相似度: {sim_score:.4f}): {node_name}, id为{node_id}, 节点属性为{node_label}"
                info += " (该节点为向量召回结果)"
                result_list.append(info)

        if not result_list:
            return "向量召回未找到任何结果。请使用其他工具。"
            
        return "\n".join(result_list)

    def mount_data(self, id, data, f_out):
        """将结果写入文件"""
        try:
            data_id = data.get('identity', 'UNKNOWN_ID')
            target_node_id = id
            relation_name = "isBelongTo"

            result_record = {
                "data_id": data_id,
                "target_node_id": target_node_id,
                "relation_type": relation_name
            }

            f_out.write(json.dumps(result_record, ensure_ascii=False) + '\n')
            return f"成功记录到文件：data {data_id} -> [{relation_name}] -> node {target_node_id}"

        except Exception as e:
            logging.info(f"❌ 写入文件失败: {e}")
            return f"写入文件失败：{e}"

    def get_supplementary_info_from_llm(self, data_item):
        """调用LLM获取材料的补充摘要信息"""
        try:
            material_name = data_item.get("name", "未知")
            composition = data_item.get("成分", "未知")

            if material_name == "未知" and composition == "未知":
                return "N/A"
                
            prompt_content = f"""
1. 请识别材料的成分。
2. 分析各成分的性质。
3. 基于材料各成分的性质理解这可能是什么样的材料。

name: '{material_name}'
成分: '{composition}'

请用100字以内告诉我这是什么具体的材料类型，包括简单的判断原因。
"""
            logging.info(f"--- 正在调用LLM获取补充信息 (Name: {material_name})... ---")
            
            response = self.client.chat.completions.create(
                model="qwen3-max",
                messages=[
                    {"role": "system", "content": "你是一个材料科学专家，请简明扼要地分析材料。"},
                    {"role": "user", "content": prompt_content}
                ]
            )
            
            summary = response.choices[0].message.content
            logging.info(f"--- 成功获取补充信息: {summary} ---")
            return summary

        except Exception as e:
            logging.warning(f"⚠️ 调用LLM获取补充信息失败: {e}")
            return "获取补充信息失败"