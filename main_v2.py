import requests
import json
import os
import sys
from volcenginesdkarkruntime import Ark
from datetime import datetime
import numpy as np 
from numpy.linalg import norm 
import torch 
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from tqdm import tqdm
sys.path.append(os.getcwd())
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    DATA_FILE_PATH, 
    # DEEPSEEK_API_KEY , DEEPSEEK_BASE_URL,
    # VOLCANO_API_KEY, VOLCANO_REGION,
    RESULT_DIR, RESULT_FILE_PREFIX
)
from neo4j_connector import Neo4jConnector

# --- MODIFICATION: Configuration for Multithreading ---
MAX_WORKERS = 5  # Concurrent threads, adjust based on your API limits (5-10 is a good start)

# --- MODIFICATION: API Key Management ---
# Load API keys from environment variables or a list
API_KEYS = [
    os.environ.get("ARK_API_KEY"),
    # "your_second_api_key",  # If you have a second key, uncomment
    # "your_third_api_key",   # A third one
]
API_KEYS = [key for key in API_KEYS if key]  # Filter out any None values

if not API_KEYS:
    print("❌ Error: No API keys found. Please set the ARK_API_KEY environment variable.")
    sys.exit(1)

# Create a list of API clients
clients = [
    Ark(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=key)
    for key in API_KEYS
]

# Thread-safe counters for rotating API keys
api_client_counter = 0
api_client_lock = Lock()

def get_next_api_client():
    """
    Rotates through the available API clients in a thread-safe manner.
    """
    global api_client_counter
    with api_client_lock:
        client = clients[api_client_counter]
        api_client_counter = (api_client_counter + 1) % len(clients)
        return client
# --- End of MODIFICATION ---

# 数据库接口初始化
neo4j = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# 文件写入锁
file_lock = Lock()

# API轮询计数器和锁
api_counter = 0
api_lock = Lock()

# --- 1. 加载运行时Embedding模型 --- # 
RUNTIME_EMBED_MODEL_PATH = "/home/thl/models/Qwen3-4B-clustering_1078/checkpoint-936" 
RUNTIME_EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNTIME_EMBED_TOKENIZER = None
RUNTIME_EMBED_MODEL = None

try:
    print(f"--- 正在加载运行时Embedding模型到 {RUNTIME_EMBED_DEVICE} ---")
    RUNTIME_EMBED_TOKENIZER = AutoTokenizer.from_pretrained(RUNTIME_EMBED_MODEL_PATH)
    RUNTIME_EMBED_MODEL = AutoModel.from_pretrained(
        RUNTIME_EMBED_MODEL_PATH, 
        torch_dtype="auto"
    ).to(RUNTIME_EMBED_DEVICE).eval()
    print(f"✅ 成功加载用于运行时推理的Embedding模型。")
except Exception as e:
    print(f"⚠️ 警告：未能加载运行时Embedding模型: {e}")
    print("     recall_top5_materials 功能将不可用。")

# --- 2. 加载预计算的向量库 --- # 
EMBEDDINGS_DB_PATH = "/home/thl/2025Fall/LLM_Mount_KG/embedding/data/material_embeddings.npy"
EMBEDDINGS_METADATA_PATH = "/home/thl/2025Fall/LLM_Mount_KG/embedding/data/material_metadata.json"

# 测试用
# EMBEDDINGS_DB_PATH = "/home/thl/2025Fall/LLM_Mount_KG/test/data/material_embeddings.npy"
# EMBEDDINGS_METADATA_PATH = "/home/thl/2025Fall/LLM_Mount_KG/test/data/material_metadata.json"

MATERIAL_EMBEDDINGS = None
MATERIAL_METADATA = []
MATERIAL_ID_TO_METADATA = {} # 用于快速查找

try:
    MATERIAL_EMBEDDINGS = np.load(EMBEDDINGS_DB_PATH)
    with open(EMBEDDINGS_METADATA_PATH, 'r', encoding='utf-8') as f:
        MATERIAL_METADATA = json.load(f)
        # 创建一个 ID -> 元数据 的映射，方便使用
        MATERIAL_ID_TO_METADATA = {item['id']: item for item in MATERIAL_METADATA}
        
    print(f"✅ 成功加载 {len(MATERIAL_METADATA)} 条预计算的Material向量。")
except Exception as e:
    print(f"⚠️ 警告：未能加载预计算的Material向量: {e}")
    print(f"     路径: {EMBEDDINGS_DB_PATH}, {EMBEDDINGS_METADATA_PATH}")
    print("     recall_top5_materials 功能将不可用。")

def get_include_outbound(id):
    # 获取所有include出边
    outbound_ids, outbound_names, outbound_labels = neo4j.get_outbound_by_id(id, "include")
    result_list = []
    # 补充include出边节点的信息
    for i in range(len(outbound_ids)):
        _, extra_out_names, __ = neo4j.get_outbound_by_id(outbound_ids[i],"include")
        _, extra_in_names, __ = neo4j.get_inbound_by_id(outbound_ids[i],"isBelongTo")
        info = f"- 选项{chr(ord('A')+i)}: {outbound_names[i]},id为{outbound_ids[i]},节点属性为{outbound_labels[i]}"
        if len(extra_out_names) > 0:
            info += "该下游节点代表性的include出边有："
            count = 0
            for j in range(len(extra_out_names)):
                if count == 3:
                    break
                info += f"{extra_out_names[j]},"
                count += 1
        else:
            info += "该下游节点没有include出边。"
        if len(extra_in_names) > 0:
            info += "该下游节点代表性的isBelongTo入边有："
            count = 0
            for j in range(len(extra_in_names)):
                if count == 3:
                    break
                info += f"{extra_in_names[j]},"
                count += 1
        else:
            info += "该下游节点没有isBelongTo入边。"
        result_list.append(info)
        # print(info)

    return "\n".join(result_list)

def get_isbelongto_inbound(inbound_ids, inbound_names, inbound_labels):
    # 这个函数现在是一个格式化工具
    # 它只处理传入的数据，只处理前5条
    result_list = []
    
    # 只迭代前5个，即使传入了更多
    for i in range(min(len(inbound_ids), 5)):
        _, extra_out_names, __ = neo4j.get_outbound_by_id(inbound_ids[i], "include")
        # BUG 修复：这里应该是 get_inbound_by_id
        _, extra_in_names, __ = neo4j.get_inbound_by_id(inbound_ids[i], "isBelongTo") 
        info = f"- 选项{chr(ord('A')+i)}: {inbound_names[i]},id为{inbound_ids[i]},节点属性为{inbound_labels[i]}"
        if len(extra_out_names) > 0:
            info += "该下游节点代表性的include出边有："
            count = 0
            for j in range(len(extra_out_names)):
                if count == 3:
                    break
                info += f"{extra_out_names[j]},"
                count += 1
        else:
            info += "该下游节点没有include出边。"
        if len(extra_in_names) > 0:
            info += "该下游节点代表性的isBelongTo入边有："
            count = 0
            for j in range(len(extra_in_names)):
                if count == 3:
                    break
                info += f"{extra_in_names[j]},"
                count += 1
        else:
            info += "该下游节点没有isBelongTo入边。"
        result_list.append(info)
    
    return "\n".join(result_list)

# 原函数

def get_embedding_for_data(data_item):
    """
    (辅助函数)
    为传入的单个材料数据（新数据）生成embedding。
    """
    if not RUNTIME_EMBED_MODEL or not RUNTIME_EMBED_TOKENIZER:
        print("❌ 运行时Embedding模型未加载。")
        return None
    
    # 扁平化 (来自 neo4j_connector.py)
    def flatten_dict(data_dict, parent_key='', separator='_'):
        items = []
        for key, value in data_dict.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(flatten_dict(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        return dict(items)

    # 将data_item（字典）转换为用于embedding的字符串
    # 这一步的格式必须与 generate_material_embeddings.py 中的 format_data_for_embedding 一致！
    
    properties_to_embed = data_item.get('data', data_item)
    
    # 扁平化并移除元数据
    flat_properties = flatten_dict(properties_to_embed)
    flat_properties.pop('_id', None) 
    flat_properties.pop('_meta_id', None)
    flat_properties.pop('_tid', None)
    
    # MGE18_标题 可能是最好的 "name" 替代品
    material_name = flat_properties.pop("MGE18_标题", "未知牌号")
    
    props_str = ", ".join([f"{k}: {v}" for k, v in flat_properties.items()])
    
    # *** 确保这个格式与离线脚本完全一致 ***
    text = f"材料名称: {material_name}, 材料属性: {props_str}"
    
    with torch.no_grad():
        inputs = RUNTIME_EMBED_TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(RUNTIME_EMBED_DEVICE)
        outputs = RUNTIME_EMBED_MODEL(**inputs)
        last_hidden_state = outputs.last_hidden_state
        # 使用 mean pooling
        embedding = last_hidden_state.mean(dim=1).squeeze()
    
    return embedding.cpu().numpy()

# 测试用
'''
def get_embedding_for_data(data_item):
    """
    (辅助函数)
    为传入的单个材料数据（新数据）生成embedding。
    *** 已更新，以匹配 v3 离线脚本的逻辑 ***
    """
    if not RUNTIME_EMBED_MODEL or not RUNTIME_EMBED_TOKENIZER:
        print("❌ 运行时Embedding模型未加载。")
        return None
    props_copy = data_item.copy()
    material_name = props_copy.pop("name", "未知牌号") 
    props_copy.pop("id", None)     
    props_copy.pop("来源", None)   
    props_copy.pop("data", None)   
    props_str = ", ".join([f"{k}: {v}" for k, v in props_copy.items() if v is not None])

    text = f"材料名称: {material_name}, 材料属性: {props_str}"
    
    # 生成 Embedding
    with torch.no_grad():
        inputs = RUNTIME_EMBED_TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(RUNTIME_EMBED_DEVICE)
        outputs = RUNTIME_EMBED_MODEL(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embedding = last_hidden_state.mean(dim=1).squeeze()
    
    return embedding.cpu().numpy()
'''

def recall_top5_materials(current_node_id, data_item):
    """
    (工具函数实现)
    在当前节点下，根据材料数据，通过向量召回Top-5最相似的Material节点。
    current_node_id: 当前节点ID
    data_item: 完整的材料数据 (用于生成embedding)
    """
    if MATERIAL_EMBEDDINGS is None or not MATERIAL_METADATA or RUNTIME_EMBED_MODEL is None:
        return "向量召回功能不可用（模型或向量库未加载）。请使用 get_include_outbound 或 get_isbelongto_inbound。"

    # 1. 为新数据生成向量
    try:
        query_embedding = get_embedding_for_data(data_item)
        if query_embedding is None:
            return "为新数据生成向量失败。请使用其他工具。"
    except Exception as e:
        print(f"❌ 为新数据生成向量时出错: {e}")
        return f"为新数据生成向量时出错: {e}。请使用其他工具。"

    # 2. 计算余弦相似度
    query_embedding = query_embedding.flatten()
    query_norm = norm(query_embedding)
    if query_norm == 0:
        print("⚠️ 警告: 查询向量的范数为0。")
        return "查询向量生成失败（范数为0）。请使用其他工具。"
        
    db_norms = norm(MATERIAL_EMBEDDINGS, axis=1)
    db_norms[db_norms == 0] = 1e-6 
    
    similarities = np.dot(MATERIAL_EMBEDDINGS, query_embedding) / (db_norms * query_norm)
    
    # 3. 获取Top-5的索引
    top_5_indices = np.argsort(similarities)[-5:][::-1]
    
    # 4. 格式化结果
    result_list = []
    for i, idx in enumerate(top_5_indices):
        if idx < len(MATERIAL_METADATA):
            meta_item = MATERIAL_METADATA[idx] 
            sim_score = similarities[idx]
            
            node_id = meta_item.get("id")
            node_name = meta_item.get("name")
            node_label = meta_item.get("label")
            
            info = f"- 选项{chr(ord('A')+i)} (相似度: {sim_score:.4f}): {node_name}, id为{node_id}, 节点属性为{node_label}"
            info += " (该节点为向量召回结果)"
            
            result_list.append(info)

    if not result_list:
        return "向量召回未找到任何结果。请使用其他工具。"
        
    return "\n".join(result_list)

def mount_data(id, data, f_out):
    """
    (模拟挂载) 将结果写入文件，而不是数据库。
    id: 目标节点ID (来自 func_args["id"])
    data: 完整的材料数据 (包含 _id)
    f_out: 打开的文件句柄
    """
    try:
        data_id = data.get('_id', 'UNKNOWN_ID') # 原代码
        # data_id = data.get('id', 'UNKNOWN_ID')  # 测试用
        target_node_id = str(id)
        relation_name = "isBelongTo"

        result_record = {
            "data_id": data_id,
            "target_node_id": target_node_id,
            "relation_type": relation_name
        }

        f_out.write(json.dumps(result_record, ensure_ascii=False) + '\n')
        
        return f"成功记录到文件：data {data_id} -> [{relation_name}] -> node {target_node_id}"

    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        return f"写入文件失败：{e}"

os.makedirs(RESULT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_filename = f"{RESULT_FILE_PREFIX}_{timestamp}.jsonl" 
result_filepath = os.path.join(RESULT_DIR, result_filename)

print(f"✅ 结果将保存到: {result_filepath}")
# 获取所有function
tools = []
with open('/home/thl/2025Fall/LLM_Mount_KG/tools.json', 'r', encoding='utf-8') as f:
    tools = json.load(f)

client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)

with open(result_filepath, 'w', encoding='utf-8') as f_out:
    with open(DATA_FILE_PATH, "r") as f:  # 原代码
    # with open("/home/thl/2025Fall/LLM_Mount_KG/test/data/test.json", "r") as f: # 测试用
        data_list = json.load(f)
        for data in data_list:
            #初始节点
            curr_node = "材料"
            curr_id = 0
            curr_label = "Class"
            extra_info = ""

            for i in range(7):
                # 最多6轮
                print("="*20 + f"第{str(i+1)}轮" + "="*20)
                prompt = f"""
# 任务说明
你正在材料知识图谱中导航，目标是将材料数据挂载到正确材料牌号下。

# 当前状态
- 当前节点ID: {curr_id}，名称：{curr_node}，标签：{curr_label}
- 材料数据：{str(data)}

# 可用工具
1. get_include_outbound - 查看当前节点的下级分类
2. get_isbelongto_inbound - 查看当前节点的包含的材料material种类
3. recall_top5_materials - (向量召回) 当 get_isbelongto_inbound 结果过多时，系统会自动调用此工具。
4. mount_data - 挂载材料数据

# 导航规则
1.  **如果当前节点标签是 "Material"**: 
    - **必须** 调用 `mount_data`。
2.  **如果当前节点标签是 "Class"**:
    - **必须** 优先调用 `get_include_outbound` 寻找下级分类。
    - **(仅当 get_include_outbound 返回为空时)**: 你才应该调用 `get_isbelongto_inbound` 寻找挂载的材料实例。
{extra_info}
"""
    # todo prompt这里因为暂时向量召回未实现，要求其到Material节点强行执行挂载
                print(prompt)
                # 用户查询
                messages = [{"role": "user", "content": prompt}]

                # 第一次调用：模型决定是否调用函数
                response = client.chat.completions.create(
                    # model="deepseek-chat",
                    model="ep-20251027162913-hvhdm",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message

                # 检查模型是否要求调用函数
                if response_message.tool_calls:
                    print("-"*30)
                    print("function call")
                    # 提取函数调用信息
                    tool_call = response_message.tool_calls[0]
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    print(f"function name: {func_name}")
                    print(f"function args: {func_args}")
                    print("-"*30)
                    
                    # 执行对应的函数
                    if func_name == "get_include_outbound":
                        tool_result = get_include_outbound(func_args["id"])
                        if len(tool_result) == 0:
                            extra_info = "上一次调用get_include_outbound没有任何结果，应该使用其他工具函数。"
                            continue

                    elif func_name == "get_isbelongto_inbound":
                        # 1. 先获取 *所有* 原始数据
                        inbound_ids, inbound_names, inbound_labels = neo4j.get_inbound_by_id(func_args["id"], "isBelongTo")
                        
                        total_count = len(inbound_ids)
                        
                        if total_count > 5:
                            # 2. 数量过多，自动触发向量召回
                            print(f"--- 节点 {curr_id} 实例过多 ({total_count}个)，自动触发向量召回 ---")
                            tool_result = recall_top5_materials(func_args["id"], data)
                            if not tool_result or "不可用" in tool_result or "未找到" in tool_result or "失败" in tool_result:
                                print(f"❌ 向量召回失败: {tool_result}")
                                extra_info = "向量召回失败，终止当前数据。"
                                break # 终止当前数据的处理
                        
                        elif 0 < total_count <= 5:
                            # 3. 数量可控 (<=5)，调用 *新* 的格式化函数
                            print(f"--- 节点 {curr_id} 实例数量可控 ({total_count}个)，使用 get_isbelongto_inbound ---")
                            # 我们重用已经获取的数据，而不是再次查询
                            tool_result = get_isbelongto_inbound(inbound_ids, inbound_names, inbound_labels)
                        
                        else: # total_count == 0
                            # 4. 没有任何实例
                            tool_result = "" # 将触发下面的 len(tool_result) == 0
                        
                        if len(tool_result) == 0:
                            extra_info = "上一次调用get_isbelongto_inbound没有任何结果。这可能是一个空的叶子节点，无法挂载。"
                            print(extra_info)
                            break # 终止当前数据的处理
                        # <--- 新逻辑结束 ---

                    elif func_name == "recall_top5_materials":
                        # LLM 仍然可能直接调用它 (例如在promptTuning失败时)，我们保留这个路径
                        print("--- LLM 主动调用 recall_top5_materials ---")
                        tool_result = recall_top5_materials(func_args["id"], data)
                        if not tool_result or "不可用" in tool_result or "未找到" in tool_result or "失败" in tool_result:
                            print(f"❌ 向量召回失败: {tool_result}")
                            extra_info = "向量召回失败，终止当前数据。"
                            break
                        
                    elif func_name == "mount_data":
                        tool_result = mount_data(func_args["id"], data, f_out)
                        print(tool_result)
                        break

                    extra_info = ""                     
                    prompt = f"""
# 任务说明
你正在材料知识图P中导航，你需要抉择当前节点的走向。

# 当前状态
- 当前节点ID: {curr_id}，名称：{curr_node}，标签：{curr_label}
- 材料数据：{str(data)}

# 节点去向
{tool_result}

请根据节点去向信息（包括名称和下游节点相关例子），选择一个与当前材料数据最接近的类型的材料去向输出。
请严格按照以下格式输出ID，名称，标签，不要添加任何额外文字：
[节点ID] [节点名称] [节点标签]

示例：
0 材料 Class
"""
                    print(prompt)
                    messages = [{"role": "user", "content": prompt}]
                    
                    # 第二次调用：模型基于函数结果决定去向
                    second_response = client.chat.completions.create(
                        # model="deepseek-chat",
                        model="ep-20251027162913-hvhdm",
                        messages=messages
                    )
                    final_answer = second_response.choices[0].message.content
                    print("-"*30)
                    print(final_answer)
                    print("-"*30)
                    if final_answer == "完毕":
                        break
                    else:
                        try:
                            # <--- 修复：重写解析器以处理带空格的名称 ---
                            clean_answer = final_answer.replace('[', '').replace(']', '').strip()
                            parts = clean_answer.split()
                            
                            if len(parts) >= 3:
                                curr_id = parts[0]
                                # 标签始终是最后一个词
                                curr_label = parts[-1]
                                # 名称是 ID 和 Label 之间的所有内容
                                curr_node = " ".join(parts[1:-1])
                            else:
                                # 至少需要 ID, Name, Label
                                print(f"❌ AI返回格式错误 (部件太少): '{final_answer}'")
                                raise IndexError
                            
                        except IndexError:
                            print(f"❌ AI返回格式错误: '{final_answer}'， 终止当前数据处理。")
                            break
                        
                else:
                    # 模型未调用函数，直接返回回答
                    print(response_message.content)

print(f"✅ 任务完成，所有数据已处理并保存到: {result_filepath}")