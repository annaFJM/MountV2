import re
# 负责解析化学式
# 1-118号元素全集
ALL_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}

# 常见干扰词
NOISE_WORDS = [
    "High", "Entropy", "Alloy", "Coating", "Series", "Based", "Composite", 
    "Signage", "The", "And", "For", "With", "In", "As", "At", "Of", "On", 
    "To", "By", "Am", "Is", "Or", "Be", "Film", "Layer", "Method", "Preparation",
    "Wt", "Vol", "At", "Pct", "Percent" # 增加成分描述里的干扰词
]

def extract_elements_from_text(text):
    """
    从任意文本（名称或成分描述）中提取化学元素集合
    """
    if not isinstance(text, str) or not text:
        return set()
    
    # 1. 去除中文
    clean_text = re.sub(r'[\u4e00-\u9fa5]', '', text)
    # 2. 替换非字母字符为空格 (保留字母，用于分割单词)
    clean_text = re.sub(r'[^a-zA-Z]', ' ', clean_text)
    
    tokens = clean_text.split()
    valid_tokens = []
    
    for token in tokens:
        # 忽略干扰词 (Case-insensitive)
        if token.capitalize() in NOISE_WORDS:
            # 特殊处理：如果单词在干扰词表中，但也在元素表中（如 In, As, Be）
            # 简单策略：如果在这种上下文中出现，通常是单词而非元素，这里选择忽略
            continue
        valid_tokens.append(token)
        
    search_str = " ".join(valid_tokens)
    
    # 3. 正则提取 (大写开头，可选小写结尾)
    potential_elements = re.findall(r'([A-Z][a-z]?)', search_str)
    
    final_elements = set()
    for el in potential_elements:
        if el in ALL_ELEMENTS:
            final_elements.add(el)
            
    return final_elements

def extract_elements_from_data_item(data):
    """
    针对传入的数据字典，综合从 Name 和 成分 中提取
    """
    elements = set()
    
    # 1. 优先从 Name 提取
    name = data.get("name", "")
    elements.update(extract_elements_from_text(name))
    
    # 2. 如果 Name 没提取到，或者想补充，从成分提取
    # 注意：你的数据成分可能是字典也可能是字符串
    composition = data.get("成分", "")
    if isinstance(composition, dict):
        # 如果是字典 {"Ti": 1.0, ...}
        for key in composition.keys():
            elements.update(extract_elements_from_text(key))
    elif isinstance(composition, str):
        # 如果是字符串 "Al: 9.85 wt%\nCu: 16.52 wt%..."
        elements.update(extract_elements_from_text(composition))
        
    return elements

def calculate_jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union