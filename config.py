"""
配置文件 - 存放所有配置信息
"""
import os

# Neo4j 数据库配置
NEO4J_URI = "neo4j://10.77.50.200:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "thl123!@#"

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 数据文件路径
DATA_FILE_PATH = "/home/thl/2025Fall/Mount-Data-to-KG/project/data/high_entropy_alloy.json"

# 根节点配置
ROOT_ELEMENT_ID = "4:e8330145-fd8c-484c-a38a-f3900a6199de:0"
ROOT_NAME = "材料"

# ===== 新增配置 =====
ENTITY_SIMILARITY_THRESHOLD = 20
MAX_CONVERSATION_ROUNDS = 20
# 特殊节点列表（需要特殊分类的节点）
SPECIAL_NODES = ["高熵合金"]  # 后续可扩展

# 分类最大深度（防止死循环）
MAX_CLASSIFICATION_DEPTH = 20

# 日志配置
LOG_DIR = "logs"
LOG_FILE_PREFIX = "mount_log"  # 格式: mount_log_20250113_143025.log

# 结果文件配置
RESULT_DIR = "results"
RESULT_FILE_PREFIX = "mount_result"  # 格式: mount_result_20250113_143025.json