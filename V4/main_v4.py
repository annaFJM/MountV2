import json
import os
import sys
import logging
import numpy as np
from datetime import datetime
from openai import OpenAI
sys.path.append(os.getcwd())

# 导入配置和自定义类
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    RESULT_DIR, RESULT_FILE_PREFIX
)
from neo4j_connector import Neo4jConnector
from function_pool import FunctionPool

# --- 1. 基础环境与日志配置 ---
log_dir = "/home/thl/2025Fall/LLM_Mount_KG/V4/log"
os.makedirs(log_dir, exist_ok=True)
log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_filepath = os.path.join(log_dir, f"{log_timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- 2. 资源加载与初始化 ---

# 初始化数据库连接
neo4j_conn = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# 初始化 LLM 客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 加载向量库数据
EMBEDDINGS_DB_PATH = "/home/thl/2025Fall/LLM_Mount_KG/test/data/material_embeddings_qwenAPI.npy"
EMBEDDINGS_METADATA_PATH = "/home/thl/2025Fall/LLM_Mount_KG/test/data/material_metadata_qwenAPI.json"
MATERIAL_EMBEDDINGS = None
MATERIAL_METADATA = []

try:
    MATERIAL_EMBEDDINGS = np.load(EMBEDDINGS_DB_PATH)
    with open(EMBEDDINGS_METADATA_PATH, 'r', encoding='utf-8') as f:
        MATERIAL_METADATA = json.load(f)
    logging.info(f"✅ 成功加载 {len(MATERIAL_METADATA)} 条预计算的Material向量。")
except Exception as e:
    logging.info(f"⚠️ 警告：未能加载向量库: {e}。recall_top5_materials 功能将不可用。")

# --- 3. 实例化 FunctionPool ---
pool = FunctionPool(neo4j_conn, client, MATERIAL_EMBEDDINGS, MATERIAL_METADATA)

# 结果文件准备
os.makedirs(RESULT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_filepath = os.path.join(RESULT_DIR, f"{RESULT_FILE_PREFIX}_{timestamp}.jsonl")
logging.info(f"✅ 结果将保存到: {result_filepath}")

# 加载 Tools 定义
with open('/home/thl/2025Fall/LLM_Mount_KG/V4/tool.json', 'r', encoding='utf-8') as f:
    tools = json.load(f)

# --- 4. 主执行循环 ---
if __name__ == "__main__":
    with open(result_filepath, 'w', encoding='utf-8') as f_out:
        # 加载测试数据
        test_data_path = "/home/thl/2025Fall/LLM_Mount_KG/test/data/test.json"
        with open(test_data_path, "r") as f:
            # test_data_list = json.load(f)
            data_list = json.load(f)
            test_data_list = data_list[:2]
            logging.info(f"--- 成功加载数据，共 {len(test_data_list)} 条 ---")

        for data in test_data_list:
            # 调用 Pool 获取补充信息
            supplementary_info = pool.get_supplementary_info_from_llm(data)
            
            curr_node = "材料"
            curr_id = 0
            curr_label = "Class"
            extra_info = ""
            mount_succeeded = False

            # 每一条数据最多执行 7 轮对话
            for i in range(7):
                logging.info("="*20 + f"第{str(i+1)}轮" + "="*20)
                
                prompt = f"""
# 任务说明
你正在材料知识图谱中导航，目标是将材料数据挂载到正确材料牌号下。

# 当前状态
- 当前节点ID: {curr_id}，名称：{curr_node}，标签：{curr_label}
- 材料数据：{str(data)}
- **补充信息**: {supplementary_info}

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
                logging.info(prompt)
                messages = [{"role": "user", "content": prompt}]

                # --- 第一次调用 LLM：决策 ---
                response = client.chat.completions.create(
                    model="qwen3-max",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message

                if response_message.tool_calls:
                    logging.info("-" * 30 + "\nfunction call")
                    tool_call = response_message.tool_calls[0]
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    logging.info(f"function name: {func_name}")
                    logging.info(f"function args: {func_args}")
                    logging.info("-" * 30)

                    tool_result = ""

                    # --- 分发工具调用到 FunctionPool ---
                    if func_name == "get_include_outbound":
                        tool_result = pool.get_include_outbound(func_args["id"])
                        if len(tool_result) == 0:
                            extra_info = "上一次调用get_include_outbound没有任何结果，应该使用其他工具函数。"
                            continue

                    elif func_name == "get_isbelongto_inbound":
                        # 特殊逻辑保留在 Main：判断数量并决定是否回退到向量召回
                        inbound_ids, inbound_names, inbound_labels = neo4j_conn.get_inbound_by_id(func_args["id"], "isBelongTo")
                        total_count = len(inbound_ids)

                        if total_count > 5:
                            logging.info(f"--- 节点 {curr_id} 实例过多 ({total_count}个)，自动触发向量召回 ---")
                            tool_result = pool.recall_top5_materials(func_args["id"], data)
                            if not tool_result or "不可用" in tool_result or "失败" in tool_result:
                                extra_info = "向量召回失败，终止当前数据。"
                                break
                        elif 0 < total_count <= 5:
                            logging.info(f"--- 节点 {curr_id} 实例数量可控 ({total_count}个)，使用 get_isbelongto_inbound ---")
                            # 调用 Pool 中的格式化方法
                            tool_result = pool.format_isbelongto_inbound(inbound_ids, inbound_names, inbound_labels)
                        else:
                            tool_result = "" # 触发空结果逻辑

                        if len(tool_result) == 0:
                            extra_info = "上一次调用get_isbelongto_inbound没有任何结果。这可能是一个空的叶子节点，无法挂载。"
                            logging.info(extra_info)
                            break

                    elif func_name == "recall_top5_materials":
                        logging.info("--- LLM 主动调用 recall_top5_materials ---")
                        tool_result = pool.recall_top5_materials(func_args["id"], data)
                        if not tool_result or "不可用" in tool_result or "失败" in tool_result:
                            extra_info = "向量召回失败，终止当前数据。"
                            break

                    elif func_name == "mount_data":
                        # 传递文件句柄
                        tool_result = pool.mount_data(func_args["id"], data, f_out)
                        logging.info(tool_result)
                        mount_succeeded = True
                        break
                    
                    # 清空 extra_info
                    extra_info = ""
                    # --- 第二次调用 LLM：基于工具结果选择去向 ---
                    prompt = f"""
# 任务说明
你正在材料知识图谱中导航，你需要抉择当前节点的走向。

# 当前状态
- 当前节点ID: {curr_id}，名称：{curr_node}，标签：{curr_label}
- 材料数据：{str(data)}
- **补充信息**: {supplementary_info}

# 节点去向
{tool_result}

请根据节点去向信息（包括名称和下游节点相关例子），选择一个与当前材料数据最接近的类型的材料去向输出。
请严格按照以下格式输出ID，名称，标签，不要添加任何额外文字：
[节点ID] [节点名称] [节点标签]

示例：
0 材料 Class
"""
                    logging.info(prompt)
                    messages = [{"role": "user", "content": prompt}]
                    
                    second_response = client.chat.completions.create(
                        model="qwen3-max",
                        messages=messages
                    )
                    final_answer = second_response.choices[0].message.content
                    logging.info("-" * 30)
                    logging.info(final_answer)
                    logging.info("-" * 30)

                    if final_answer == "完毕":
                        break
                    else:
                        try:
                            clean_answer = final_answer.replace('[', '').replace(']', '').strip()
                            parts = clean_answer.split()
                            if len(parts) >= 3:
                                curr_id = parts[0]
                                curr_label = parts[-1]
                                curr_node = " ".join(parts[1:-1])
                            else:
                                logging.info(f"❌ AI返回格式错误 (部件太少): '{final_answer}'")
                                raise IndexError
                        except IndexError:
                            logging.info(f"❌ AI返回格式错误: '{final_answer}'， 终止当前数据处理。")
                            break
                else:
                    logging.info(response_message.content)

            # 记录挂载失败
            if not mount_succeeded:
                try:
                    data_id = data.get('identity', 'UNKNOWN_ID')
                    logging.info(f"--- 数据 {data_id} 未能成功挂载，写入空值记录 ---")
                    failure_record = {
                        "data_id": data_id,
                        "target_node_id": None,
                        "relation_type": None
                    }
                    f_out.write(json.dumps(failure_record, ensure_ascii=False) + '\n')
                except Exception as e:
                    logging.info(f"❌ 写入失败记录时发生错误: {e}")

    logging.info(f"✅ 任务完成，所有数据已处理并保存到: {result_filepath}")