# test_function_call.py
import os
from volcenginesdkarkruntime import Ark
import json

# 初始化Ark客户端
client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)

# 定义测试用的 function (工具)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_material_info",
            "description": "获取材料信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_id": {
                        "type": "string",
                        "description": "材料ID"
                    },
                    "property_name": {
                        "type": "string",
                        "description": "要查询的属性名称"
                    }
                },
                "required": ["material_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_materials",
            "description": "搜索符合条件的材料",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["keyword"]
            }
        }
    }
]

# 测试 Function Call
print("="*50)
print("测试 Function Call 功能")
print("="*50)

try:
    response = client.chat.completions.create(
        model="ep-20251027162913-hvhdm",
        messages=[
            {"role": "system", "content": "你是一个材料知识图谱助手。"},
            {"role": "user", "content": "帮我查询ID为12345的材料信息"}
        ],
        tools=tools,
        tool_choice="auto",  # 让模型自动决定是否调用函数
        extra_headers={'x-is-encrypted': 'true'},
    )
    
    message = response.choices[0].message
    
    # 检查模型是否调用了函数
    if message.tool_calls:
        print("\n✅ 模型支持 Function Call！")
        print("\n模型决定调用以下函数：")
        for tool_call in message.tool_calls:
            print(f"\n函数名: {tool_call.function.name}")
            print(f"函数参数: {tool_call.function.arguments}")
            
            # 解析参数
            args = json.loads(tool_call.function.arguments)
            print(f"解析后的参数: {json.dumps(args, indent=2, ensure_ascii=False)}")
    else:
        print("\n⚠️  模型没有调用函数，直接返回了回答：")
        print(message.content)
        print("\n这可能意味着：")
        print("1. 模型不支持 function call")
        print("2. 或者模型认为不需要调用函数")
        
except Exception as e:
    print(f"\n❌ 错误: {e}")
    print("\n可能的原因：")
    print("1. 该模型不支持 function call 功能")
    print("2. API Key 配置有误")
    print("3. Endpoint 配置有误")

print("\n" + "="*50)

# 再测试一个更明确需要调用函数的例子
print("\n测试第二个场景（更明确的函数调用需求）")
print("="*50)

try:
    response2 = client.chat.completions.create(
        model="ep-20251027162913-hvhdm",
        messages=[
            {"role": "user", "content": "搜索包含'高熵合金'的材料"}
        ],
        tools=tools,
        tool_choice="auto",
        extra_headers={'x-is-encrypted': 'true'},
    )
    
    message2 = response2.choices[0].message
    
    if message2.tool_calls:
        print("\n✅ 第二个测试也成功！")
        for tool_call in message2.tool_calls:
            print(f"函数名: {tool_call.function.name}")
            print(f"参数: {tool_call.function.arguments}")
    else:
        print("\n⚠️  第二个测试：模型未调用函数")
        print(message2.content)
        
except Exception as e:
    print(f"\n❌ 第二个测试错误: {e}")

print("\n测试完成！")