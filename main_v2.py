import requests
import json
import os
import sys
from openai import OpenAI
sys.path.append(os.getcwd())
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    DATA_FILE_PATH, DEEPSEEK_API_KEY , DEEPSEEK_BASE_URL
)
from neo4j_connector import Neo4jConnector

# 数据库接口初始化
neo4j = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

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
            for i in range(len(extra_out_names)):
                if count == 3:
                    break
                info += f"{extra_out_names[i]},"
                count += 1
        else:
            info += "该下游节点没有include出边。"
        if len(extra_in_names) > 0:
            info += "该下游节点代表性的isBelongTo入边有："
            count = 0
            for i in range(len(extra_in_names)):
                if count == 3:
                    break
                info += f"{extra_in_names[i]},"
                count += 1
        else:
            info += "该下游节点没有isBelongTo入边。"
        result_list.append(info)
        # print(info)

    return "\n".join(result_list)

def get_isbelongto_inbound(id):
    # 获取所有isBelongTo入边
    inbound_ids, inbound_names, inbound_labels = neo4j.get_inbound_by_id(id, "isBelongTo")
    result_list = []
    # 补充include出边节点的信息
    for i in range(len(inbound_ids)):
        _, extra_out_names, __ = neo4j.get_outbound_by_id(inbound_ids[i],"include")
        _, extra_in_names, __ = neo4j.get_outbound_by_id(inbound_ids[i],"isBelongTo")
        info = f"- 选项{chr(ord('A')+i)}: {inbound_names[i]},id为{inbound_ids[i]},节点属性为{inbound_labels[i]}"
        if len(extra_out_names) > 0:
            info += "该下游节点代表性的include出边有："
            count = 0
            for i in range(len(extra_out_names)):
                if count == 3:
                    break
                info += f"{extra_out_names[i]},"
        else:
            info += "该下游节点没有include出边。"
        if len(extra_in_names) > 0:
            info += "该下游节点代表性的isBelongTo入边有："
            count = 0
            for i in range(len(extra_in_names)):
                if count == 3:
                    break
                info += f"{extra_in_names[i]},"
        else:
            info += "该下游节点没有isBelongTo入边。"
        result_list.append(info)
        # print(info)

    return "\n".join(result_list)

def mount_data(id, data):
    try:
        new_id = neo4j.create_entity(data)
        neo4j.create_relation(new_id, id)
    except Exception as e:
        return f"挂载失败：{e}"

    return f"成功挂载，data位于{str(new_id)}"


# 获取所有function
tools = []
with open('/home/thl/2025Fall/LLM_Mount_KG/tools.json', 'r', encoding='utf-8') as f:
    tools = json.load(f)

# LLM调用接口
client = OpenAI(
    api_key=DEEPSEEK_API_KEY, #已脱敏
    base_url=DEEPSEEK_BASE_URL,
)

with open(DATA_FILE_PATH, "r") as f:
    data_list = json.load(f)
    for data in data_list:
        #初始节点
        curr_node = "材料"
        curr_id = 0
        curr_label = "Class"
        extra_info = ""

        for i in range(6):
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
2. get_isbelongto_inbound - 查看当前节点的包含的材料牌号实例  
3. mount_data - 挂载材料数据

标签类型为“Material”时，已经到达正确的位置，**严格**调用mount_data工具进行挂载
{extra_info}
"""
# todo prompt这里因为暂时向量召回未实现，要求其到Material节点强行执行挂载
            print(prompt)
            # 用户查询
            messages = [{"role": "user", "content": prompt}]

            # 第一次调用：模型决定是否调用函数
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                }
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
                    tool_result = get_isbelongto_inbound(func_args["id"])
                    if len(tool_result) == 0:
                        extra_info = "上一次调用get_include_outbound没有任何结果，应该使用其他工具函数。"
                        continue
                elif func_name == "mount_data":
                    tool_result = mount_data(func_args["id"], data)
                    print(tool_result)
                    break

                extra_info = ""                    
                prompt = f"""
# 任务说明
你正在材料知识图谱中导航，你需要抉择当前节点的走向。

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
                    model="deepseek-chat",
                    messages=messages,
                    extra_body={
                        "thinking": {
                            "type": "disabled"
                        }
                    }
                )
                final_answer = second_response.choices[0].message.content
                print("-"*30)
                print(final_answer)
                print("-"*30)
                if final_answer == "完毕":
                    break
                else:
                    curr_id = final_answer.split()[0]
                    curr_node = final_answer.split()[1]
                    curr_label = final_answer.split()[2]
                    
            else:
                # 模型未调用函数，直接返回回答
                print(response_message.content)
        break

