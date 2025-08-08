import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import re
import numpy as np
import time
import os
import unicodedata  # 新增导入
from functools import lru_cache
from tqdm import tqdm

# 全局模型变量
tokenizer = None
pubmedbert_model = None
sbert_model = None

# 添加缓存装饰器以避免重复计算相同的文本对
@lru_cache(maxsize=1024)
def is_empty_value_cached(value):
    """检查值是否为空值（包括各种表示空的形式）- 带缓存版本"""
    # 快速路径 - 先检查常见情况
    if value is None:
        return True
        
    if pd.isna(value):
        return True
        
    if isinstance(value, str):
        value_str = value.lower().strip()
        # 检查各种常见的空值表示
        empty_values = ['', 'nan', 'na', 'n/a', 'none', 'null', '-', '/', 'unknown']
        return value_str in empty_values
    
    if isinstance(value, (int, float)):
        return np.isnan(value) if np.issubdtype(type(value), np.floating) else False
        
    # 其他情况，转为字符串
    value_str = str(value).lower().strip()
    empty_values = ['', 'nan', 'na', 'n/a', 'none', 'null', '-', '/', 'unknown']
    return value_str in empty_values

def is_empty_value(value):
    """为非哈希类型值提供兼容的空值检查"""
    # 尝试使用缓存版本
    try:
        return is_empty_value_cached(value)
    except:
        # 如果值不可哈希，则回退到非缓存版本
        if pd.isna(value):
            return True
            
        if isinstance(value, (int, float)):
            return np.isnan(value) if np.issubdtype(type(value), np.floating) else False
            
        value_str = str(value).lower().strip()
        empty_values = ['', 'nan', 'na', 'n/a', 'none', 'null', '-', '/', 'unknown']
        return value_str in empty_values

@lru_cache(maxsize=512)
def rule_based_similarity_cached(text1, text2):
    """规则匹配相似度计算 - 带缓存版本（增强版）"""
    # 处理空值
    if is_empty_value(text1) or is_empty_value(text2):
        return None
        
    text1 = str(text1).strip()
    text2 = str(text2).strip()
    
    # 规则1：直接包含关系
    if text2 in text1 or text1 in text2:
        return 1.0
    
    # 规则2：百分比值提取与比较
    def extract_percentage(text):
        # 提取百分比数值 (考虑 "ca." 等前缀)
        percentage_match = re.search(r'(?:ca\.|approximately|~)?\s*(\d+(?:\.\d+)?)\s*%', text)
        if percentage_match:
            return float(percentage_match.group(1))
        
        # 检查是否为小数形式 (如 0.35 表示 35%)
        decimal_match = re.search(r'(?:ca\.|approximately|~)?\s*0\.(\d+)\b', text)
        if decimal_match:
            return float(decimal_match.group(1))
            
        # 检查是否为小数形式 (如 0.8 表示 80%)
        simple_decimal = re.search(r'^0\.(\d+)$', text.strip())
        if simple_decimal:
            return float(simple_decimal.group(1))
        
        return None
    
    percent1 = extract_percentage(text1)
    percent2 = extract_percentage(text2)
    
    if percent1 is not None and percent2 is not None:
        # 比较提取出的百分比值
        if abs(percent1 - percent2) < 0.1:  # 允许小误差
            return 1.0
    
    # 规则3：括号内缩写匹配
    def extract_abbreviations(text):
        # 从括号中提取缩写
        abbrevs = re.findall(r'\((.*?)\)', text)
        return [abb.strip() for abb in abbrevs]
    
    # 从文本1提取缩写，检查是否匹配文本2
    abbrevs1 = extract_abbreviations(text1)
    if abbrevs1 and any(abb == text2 for abb in abbrevs1):
        return 1.0
    
    # 从文本2提取缩写，检查是否匹配文本1
    abbrevs2 = extract_abbreviations(text2)
    if abbrevs2 and any(abb == text1 for abb in abbrevs2):
        return 1.0
    
    # 规则4：化学式模式匹配
    if re.search(r'[A-Z][0-9a-z]*(?:-[A-Z0-9]+)?', text2):
        chemical_patterns = re.findall(r'[A-Z][0-9a-z]*(?:-[A-Z0-9]+)?', text1)
        if text2 in chemical_patterns:
            return 1.0
    
    # 规则5：规范化比较 - 处理特殊字符和空格差异
    def normalize_text(text):
        # 1. 移除所有空格
        text = re.sub(r'\s+', '', text)
        # 2. 统一Unicode变体字符 (例如不同版本的字母)
        text = unicodedata.normalize('NFKC', text)
        # 3. 转换为小写
        text = text.lower()
        return text
    
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    if norm_text1 == norm_text2:
        return 1.0
            
    # 规则6：产率描述匹配
    # 匹配 "X% yield" 和 "X%" 这样的模式
    yield_pattern1 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:yield|产率)', text1)
    yield_pattern2 = re.search(r'(\d+(?:\.\d+)?)\s*%', text2)
    
    if yield_pattern1 and yield_pattern2:
        if yield_pattern1.group(1) == yield_pattern2.group(1):
            return 1.0
            
    # 反向检查 - 文本2中的产率描述与文本1中的百分比
    yield_pattern1 = re.search(r'(\d+(?:\.\d+)?)\s*%', text1)
    yield_pattern2 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:yield|产率)', text2)
    
    if yield_pattern1 and yield_pattern2:
        if yield_pattern1.group(1) == yield_pattern2.group(1):
            return 1.0
    
    # 规则7：设备特殊处理 - 主要处理Teflon等类型的设备描述
    # 首先检查是否为设备描述
    equipment_keywords = [
        'autoclave', 'reactor', 'lined', 'teflon', 'teﬂon', 'steel', 'stainless', 
        'vessel', 'container', 'pot', 'bomb', 'flask', 'vial'
    ]
    
    # 如果两个文本都包含设备关键词，进行设备特定匹配
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    if any(keyword in text1_lower for keyword in equipment_keywords) and \
       any(keyword in text2_lower for keyword in equipment_keywords):
        
        # 提取关键部分进行比较
        # 针对Teflon lined autoclave类型的表述
        def extract_key_equipment(text):
            # 提取主要材料和设备类型
            text = text.lower()
            # 规范化常见变体
            text = text.replace('teﬂon', 'teflon')
            text = text.replace('steel', 'stainless')
            
            # 提取关键材料和设备类型
            key_parts = []
            for kw in ['teflon', 'stainless', 'autoclave', 'reactor', 'pot', 'bomb', 'vessel', 'container']:
                if kw in text:
                    key_parts.append(kw)
            
            # 如果同时有'teflon'和'lined'，确保它们被视为一个单元
            if 'teflon' in key_parts and 'lined' in text:
                key_parts.append('lined')
                
            return set(key_parts)
        
        key_parts1 = extract_key_equipment(text1_lower)
        key_parts2 = extract_key_equipment(text2_lower)
        
        # 计算关键部分的匹配程度
        common_parts = key_parts1.intersection(key_parts2)
        
        # 检查是否包含关键设备类型
        equipment_types = {'autoclave', 'reactor', 'pot', 'bomb', 'vessel', 'container'}
        has_common_equipment = bool(common_parts.intersection(equipment_types))
        
        # 检查是否都包含teflon/lined组合
        both_teflon_lined = ('teflon' in common_parts and 'lined' in common_parts)
        
        # 如果有共同的设备类型和材料描述，认为是匹配的
        if has_common_equipment and (both_teflon_lined or len(common_parts) >= 2):
            return 1.0
    
    # 规则8：化学品名称与缩写匹配
    # 常见化学品名称与缩写对照表
    chemical_abbr_map = {
        'benzenetricarboxylate': 'btc',
        'btc': 'benzenetricarboxylate',
        '1,3,5-benzenetricarboxylate': 'btc',
        'h3btc': 'benzenetricarboxylate',
        'dimethylformamide': 'dmf',
        'n,n-dimethylformamide': 'dmf',
        'dmf': 'dimethylformamide',
        'terephthalic acid': 'bdc',
        'benzenedicarboxylic acid': 'bdc',
        'h2bdc': 'benzenedicarboxylic acid',
        'trimesic acid': 'btc',
        'benzenetricarboxylic acid': 'btc'
    }
    
    # 检查两个文本是否为化学名称-缩写关系
    text1_lower = text1.lower().replace('-', '').replace(' ', '')
    text2_lower = text2.lower().replace('-', '').replace(' ', '')
    
    # 直接在映射表中查找
    if text1_lower in chemical_abbr_map and chemical_abbr_map[text1_lower] == text2_lower:
        return 1.0
    if text2_lower in chemical_abbr_map and chemical_abbr_map[text2_lower] == text1_lower:
        return 1.0
    
    # 针对H3BTC, H2BDC等形式与全名的匹配
    for key, value in chemical_abbr_map.items():
        if (text1_lower == key and text2_lower in value) or (text1_lower in value and text2_lower == key):
            return 1.0
    
    # 规则9：物质量和质量描述匹配
    # 匹配形如 "0.25mmol, 0.061g" 和 "0.061 g (0.25 mmol)" 的模式
    def extract_amount_info(text):
        # 提取所有的数量信息对
        # 格式1: "0.25mmol, 0.061g"
        # 格式2: "0.061 g (0.25 mmol)"
        # 格式3: "0.25mmol, 0.0225g;0.13mmol, 0.023g"
        
        # 尝试匹配括号格式
        paren_pattern = re.findall(r'(\d+(?:\.\d+)?)\s*g\s*\(\s*(\d+(?:\.\d+)?)\s*mmol\s*\)', text)
        if paren_pattern:
            return [(float(mmol), float(gram)) for gram, mmol in paren_pattern]
        
        # 尝试匹配逗号格式
        comma_pattern = re.findall(r'(\d+(?:\.\d+)?)\s*mmol\s*,\s*(\d+(?:\.\d+)?)\s*g', text)
        if comma_pattern:
            return [(float(mmol), float(gram)) for mmol, gram in comma_pattern]
        
        return []
    
    # 提取文本中的数量信息
    amounts1 = extract_amount_info(text1)
    amounts2 = extract_amount_info(text2)
    
    # 如果两者都有提取出的量信息
    if amounts1 and amounts2:
        # 检查每一对量信息是否匹配
        matches = 0
        total_pairs = max(len(amounts1), len(amounts2))
        
        for i in range(min(len(amounts1), len(amounts2))):
            mmol1, gram1 = amounts1[i]
            mmol2, gram2 = amounts2[i]
            
            # 允许小数点后两位的误差
            if abs(mmol1 - mmol2) < 0.01 and abs(gram1 - gram2) < 0.01:
                matches += 1
        
        # 如果所有量信息都匹配，则视为匹配
        if matches == total_pairs:
            return 1.0
    
    # 规则10：溶剂体积匹配 - 处理例如 "15 mL;2mL" 和 "17 mL"
    def extract_volumes(text):
        # 提取所有体积数值
        volumes = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ml|mL)', text)
        return [float(vol) for vol in volumes]
    
    volumes1 = extract_volumes(text1)
    volumes2 = extract_volumes(text2)
    
    # 如果提取到体积信息
    if volumes1 and volumes2:
        # 计算总体积
        total1 = sum(volumes1)
        total2 = sum(volumes2)
        
        # 如果总体积相等，则视为匹配
        if abs(total1 - total2) < 0.1:
            return 1.0
    
    return None

def rule_based_similarity(text1, text2):
    """规则匹配相似度计算，支持非字符串输入（增强版）"""
    try:
        return rule_based_similarity_cached(str(text1), str(text2))
    except:
        # 如果缓存失败，回退到非缓存版本
        if is_empty_value(text1) or is_empty_value(text2):
            return None
            
        text1 = str(text1).strip()
        text2 = str(text2).strip()
        
        # 规则1：直接包含关系
        if text2 in text1 or text1 in text2:
            return 1.0
        
        # 规则2：百分比值提取与比较
        def extract_percentage(text):
            # 提取百分比数值 (考虑 "ca." 等前缀)
            percentage_match = re.search(r'(?:ca\.|approximately|~)?\s*(\d+(?:\.\d+)?)\s*%', text)
            if percentage_match:
                return float(percentage_match.group(1))
            
            # 检查是否为小数形式 (如 0.35 表示 35%)
            decimal_match = re.search(r'(?:ca\.|approximately|~)?\s*0\.(\d+)\b', text)
            if decimal_match:
                return float(decimal_match.group(1))
                
            # 检查是否为小数形式 (如 0.8 表示 80%)
            simple_decimal = re.search(r'^0\.(\d+)$', text.strip())
            if simple_decimal:
                return float(simple_decimal.group(1))
            
            return None
        
        percent1 = extract_percentage(text1)
        percent2 = extract_percentage(text2)
        
        if percent1 is not None and percent2 is not None:
            # 比较提取出的百分比值
            if abs(percent1 - percent2) < 0.1:  # 允许小误差
                return 1.0
        
        # 规则3：括号内缩写匹配
        def extract_abbreviations(text):
            # 从括号中提取缩写
            abbrevs = re.findall(r'\((.*?)\)', text)
            return [abb.strip() for abb in abbrevs]
        
        # 从文本1提取缩写，检查是否匹配文本2
        abbrevs1 = extract_abbreviations(text1)
        if abbrevs1 and any(abb == text2 for abb in abbrevs1):
            return 1.0
        
        # 从文本2提取缩写，检查是否匹配文本1
        abbrevs2 = extract_abbreviations(text2)
        if abbrevs2 and any(abb == text1 for abb in abbrevs2):
            return 1.0
        
        # 规则4：化学式模式匹配
        if re.search(r'[A-Z][0-9a-z]*(?:-[A-Z0-9]+)?', text2):
            chemical_patterns = re.findall(r'[A-Z][0-9a-z]*(?:-[A-Z0-9]+)?', text1)
            if text2 in chemical_patterns:
                return 1.0
        
        # 规则5：规范化比较 - 处理特殊字符和空格差异
        def normalize_text(text):
            text = re.sub(r'\s+', '', text)
            text = unicodedata.normalize('NFKC', text)
            text = text.lower()
            return text
        
        norm_text1 = normalize_text(text1)
        norm_text2 = normalize_text(text2)
        
        if norm_text1 == norm_text2:
            return 1.0
            
        # 规则6：产率描述匹配
        yield_pattern1 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:yield|产率)', text1)
        yield_pattern2 = re.search(r'(\d+(?:\.\d+)?)\s*%', text2)
        
        if yield_pattern1 and yield_pattern2:
            if yield_pattern1.group(1) == yield_pattern2.group(1):
                return 1.0
        
        # 反向检查
        yield_pattern1 = re.search(r'(\d+(?:\.\d+)?)\s*%', text1)
        yield_pattern2 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:yield|产率)', text2)
        
        if yield_pattern1 and yield_pattern2:
            if yield_pattern1.group(1) == yield_pattern2.group(1):
                return 1.0
        
        # 规则7：设备特殊处理 - 主要处理Teflon等类型的设备描述
        # 首先检查是否为设备描述
        equipment_keywords = [
            'autoclave', 'reactor', 'lined', 'teflon', 'teﬂon', 'steel', 'stainless', 
            'vessel', 'container', 'pot', 'bomb', 'flask', 'vial'
        ]
        
        # 如果两个文本都包含设备关键词，进行设备特定匹配
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        if any(keyword in text1_lower for keyword in equipment_keywords) and \
           any(keyword in text2_lower for keyword in equipment_keywords):
            
            # 提取关键部分进行比较
            # 针对Teflon lined autoclave类型的表述
            def extract_key_equipment(text):
                # 提取主要材料和设备类型
                text = text.lower()
                # 规范化常见变体
                text = text.replace('teﬂon', 'teflon')
                text = text.replace('steel', 'stainless')
                
                # 提取关键材料和设备类型
                key_parts = []
                for kw in ['teflon', 'stainless', 'autoclave', 'reactor', 'pot', 'bomb', 'vessel', 'container']:
                    if kw in text:
                        key_parts.append(kw)
                
                # 如果同时有'teflon'和'lined'，确保它们被视为一个单元
                if 'teflon' in key_parts and 'lined' in text:
                    key_parts.append('lined')
                    
                return set(key_parts)
            
            key_parts1 = extract_key_equipment(text1_lower)
            key_parts2 = extract_key_equipment(text2_lower)
            
            # 计算关键部分的匹配程度
            common_parts = key_parts1.intersection(key_parts2)
            
            # 检查是否包含关键设备类型
            equipment_types = {'autoclave', 'reactor', 'pot', 'bomb', 'vessel', 'container'}
            has_common_equipment = bool(common_parts.intersection(equipment_types))
            
            # 检查是否都包含teflon/lined组合
            both_teflon_lined = ('teflon' in common_parts and 'lined' in common_parts)
            
            # 如果有共同的设备类型和材料描述，认为是匹配的
            if has_common_equipment and (both_teflon_lined or len(common_parts) >= 2):
                return 1.0
        
        # 规则8：化学品名称与缩写匹配
        # 常见化学品名称与缩写对照表
        chemical_abbr_map = {
            'benzenetricarboxylate': 'btc',
            'btc': 'benzenetricarboxylate',
            '1,3,5-benzenetricarboxylate': 'btc',
            'h3btc': 'benzenetricarboxylate',
            'dimethylformamide': 'dmf',
            'n,n-dimethylformamide': 'dmf',
            'dmf': 'dimethylformamide',
            'terephthalic acid': 'bdc',
            'benzenedicarboxylic acid': 'bdc',
            'h2bdc': 'benzenedicarboxylic acid',
            'trimesic acid': 'btc',
            'benzenetricarboxylic acid': 'btc'
        }
        
        # 检查两个文本是否为化学名称-缩写关系
        text1_lower = text1.lower().replace('-', '').replace(' ', '')
        text2_lower = text2.lower().replace('-', '').replace(' ', '')
        
        # 直接在映射表中查找
        if text1_lower in chemical_abbr_map and chemical_abbr_map[text1_lower] == text2_lower:
            return 1.0
        if text2_lower in chemical_abbr_map and chemical_abbr_map[text2_lower] == text1_lower:
            return 1.0
        
        # 针对H3BTC, H2BDC等形式与全名的匹配
        for key, value in chemical_abbr_map.items():
            if (text1_lower == key and text2_lower in value) or (text1_lower in value and text2_lower == key):
                return 1.0
        
        # 规则9：物质量和质量描述匹配
        # 匹配形如 "0.25mmol, 0.061g" 和 "0.061 g (0.25 mmol)" 的模式
        def extract_amount_info(text):
            # 提取所有的数量信息对
            # 格式1: "0.25mmol, 0.061g"
            # 格式2: "0.061 g (0.25 mmol)"
            # 格式3: "0.25mmol, 0.0225g;0.13mmol, 0.023g"
            
            # 尝试匹配括号格式
            paren_pattern = re.findall(r'(\d+(?:\.\d+)?)\s*g\s*\(\s*(\d+(?:\.\d+)?)\s*mmol\s*\)', text)
            if paren_pattern:
                return [(float(mmol), float(gram)) for gram, mmol in paren_pattern]
            
            # 尝试匹配逗号格式
            comma_pattern = re.findall(r'(\d+(?:\.\d+)?)\s*mmol\s*,\s*(\d+(?:\.\d+)?)\s*g', text)
            if comma_pattern:
                return [(float(mmol), float(gram)) for mmol, gram in comma_pattern]
            
            return []
        
        # 提取文本中的数量信息
        amounts1 = extract_amount_info(text1)
        amounts2 = extract_amount_info(text2)
        
        # 如果两者都有提取出的量信息
        if amounts1 and amounts2:
            # 检查每一对量信息是否匹配
            matches = 0
            total_pairs = max(len(amounts1), len(amounts2))
            
            for i in range(min(len(amounts1), len(amounts2))):
                mmol1, gram1 = amounts1[i]
                mmol2, gram2 = amounts2[i]
                
                # 允许小数点后两位的误差
                if abs(mmol1 - mmol2) < 0.01 and abs(gram1 - gram2) < 0.01:
                    matches += 1
            
            # 如果所有量信息都匹配，则视为匹配
            if matches == total_pairs:
                return 1.0
        
        # 规则10：溶剂体积匹配 - 处理例如 "15 mL;2mL" 和 "17 mL"
        def extract_volumes(text):
            # 提取所有体积数值
            volumes = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ml|mL)', text)
            return [float(vol) for vol in volumes]
        
        volumes1 = extract_volumes(text1)
        volumes2 = extract_volumes(text2)
        
        # 如果提取到体积信息
        if volumes1 and volumes2:
            # 计算总体积
            total1 = sum(volumes1)
            total2 = sum(volumes2)
            
            # 如果总体积相等，则视为匹配
            if abs(total1 - total2) < 0.1:
                return 1.0
        
        return None
    
# 使用批处理提高效率
def batch_encode_texts(texts, tokenizer, batch_size=32):
    """批量编码文本以提高效率"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs = pubmedbert_model(**inputs)
        
        # 平均池化
        token_embeddings = outputs[0]
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 规范化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings)
    
    return torch.cat(all_embeddings, dim=0)

# 计算批量相似度
def batch_calculate_similarities(texts1, texts2, method='pubmedbert', batch_size=32):
    """批量计算相似度以提高效率"""
    global tokenizer, pubmedbert_model, sbert_model
    
    # 过滤掉空值
    valid_indices = [i for i, (t1, t2) in enumerate(zip(texts1, texts2)) 
                  if not (is_empty_value(t1) or is_empty_value(t2))]
    
    if not valid_indices:
        return [0.0] * len(texts1)
    
    valid_texts1 = [str(texts1[i]) for i in valid_indices]
    valid_texts2 = [str(texts2[i]) for i in valid_indices]
    
    results = [0.0] * len(texts1)
    
    if method == 'pubmedbert':
        # 批量编码文本
        embeddings1 = batch_encode_texts(valid_texts1, tokenizer, batch_size)
        embeddings2 = batch_encode_texts(valid_texts2, tokenizer, batch_size)
        
        # 计算相似度
        similarities = torch.cosine_similarity(embeddings1, embeddings2)
        
        # 填充结果
        for idx, sim in zip(valid_indices, similarities):
            results[idx] = sim.item()
    
    elif method == 'sbert':
        # 对于SentenceTransformer，使用其内置批处理
        embeddings1 = sbert_model.encode(valid_texts1, batch_size=batch_size, convert_to_tensor=True)
        embeddings2 = sbert_model.encode(valid_texts2, batch_size=batch_size, convert_to_tensor=True)
        
        # 计算相似度
        similarities = torch.cosine_similarity(embeddings1, embeddings2)
        
        # 填充结果
        for idx, sim in zip(valid_indices, similarities):
            results[idx] = sim.item()
    
    return results

def calculate_pubmedbert_similarity(text1, text2, tokenizer, model):
    """使用 PubMedBERT 计算两个文本之间的相似度"""
    # 处理空值
    if is_empty_value(text1) or is_empty_value(text2):
        return 0.0
        
    text1 = str(text1).strip()
    text2 = str(text2).strip()
        
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    inputs1 = tokenizer(text1, padding=True, truncation=True, return_tensors='pt')
    inputs2 = tokenizer(text2, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    sentence_embeddings1 = mean_pooling(outputs1, inputs1['attention_mask'])
    sentence_embeddings2 = mean_pooling(outputs2, inputs2['attention_mask'])
    
    sentence_embeddings1 = torch.nn.functional.normalize(sentence_embeddings1, p=2, dim=1)
    sentence_embeddings2 = torch.nn.functional.normalize(sentence_embeddings2, p=2, dim=1)
    
    similarity = torch.cosine_similarity(sentence_embeddings1, sentence_embeddings2)
    return similarity.item()

def calculate_sbert_similarity(text1, text2, sbert_model):
    """使用 Sentence Transformer 计算两个文本之间的相似度"""
    # 处理空值
    if is_empty_value(text1) or is_empty_value(text2):
        return 0.0
        
    text1 = str(text1).strip()
    text2 = str(text2).strip()
    
    embedding1 = sbert_model.encode(text1, convert_to_tensor=True)
    embedding2 = sbert_model.encode(text2, convert_to_tensor=True)
    
    similarity = torch.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def calculate_metrics(tp, fp, fn, tn):
    """计算并返回评估指标（包含TN）"""
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

def load_models():
    """加载所有模型，确保只加载一次"""
    global tokenizer, pubmedbert_model, sbert_model
    
    start_time = time.time()
    print("加载模型中...")
    
    if tokenizer is None or pubmedbert_model is None or sbert_model is None:
        try:
            # PubMedBERT
            model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"  
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pubmedbert_model = AutoModel.from_pretrained(model_name)
            
            # Sentence Transformer
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print(f"模型加载完成，耗时 {time.time() - start_time:.2f} 秒")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise

def compare_xlsx(file1, file2, output_file, threshold=0.95, batch_size=32, use_batch=True, verbose=True, only_common=False):
    """
    比较两个Excel文件中的文本数据，计算相似度，并将结果写入新的Excel文件
    
    参数:
    file1: 第一个Excel文件路径
    file2: 第二个Excel文件路径
    output_file: 输出结果文件路径
    threshold: 相似度阈值，高于此值视为匹配
    batch_size: 批处理大小
    use_batch: 是否使用批处理
    verbose: 是否输出详细信息
    only_common: 是否只比较两个文件中共同存在的记录（不考虑缺失项）
    
    返回:
    results: 包含评估指标的DataFrame
    detailed_comparison: 包含详细比较结果的列表
    """
    # 加载必要的模型
    load_models()
    
    # 定义使用规则+PubMedBERT的列
    pubmedbert_columns = [
        'Metal Source', 'Organic Linkers Source', 
        'Modulator Source', 'Solvent Source'
    ]
    
    # 定义使用Sentence Transformer的列
    sbert_columns = [
        'Quantity of Metal', 'Quantity of Organic Linkers',
        'Quantity of Modulator', 'Quantity of Solvent',
        'Mol Ratio of Proportion of Metals, Organic linkers',
        'Synthesis Temperature', 'Synthesis Time',
        'Crystal Morphology', 'Yield', 'Equipment'
    ]
    
    # 读取Excel文件
    print("读取Excel文件...")
    try:
        start_time = time.time()
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)
        print(f"Excel读取完成，耗时 {time.time() - start_time:.2f} 秒")
    except Exception as e:
        print(f"读取Excel文件失败: {str(e)}")
        raise
    
    # 创建一个空的 DataFrame 来存储结果
    results = pd.DataFrame(columns=["Metric"] + df1.columns[1:].tolist())
    
    # 创建一个DataFrame存储每次比较的详细信息
    detailed_comparison = []
    
    # 获取位置列的名称
    position_col = df1.columns[0]
    
    # 开始时间
    total_start_time = time.time()
    
    # 统计各文件有效条目数和共同条目数
    df1_positions = set(df1[position_col].values)
    df2_positions = set(df2[position_col].values)
    common_positions = df1_positions.intersection(df2_positions)
    
    # 记录数据条目统计信息
    count_stats = {
        "文件1条目数": len(df1_positions),
        "文件2条目数": len(df2_positions),
        "共同条目数": len(common_positions),
        "文件1独有条目数": len(df1_positions - df2_positions),
        "文件2独有条目数": len(df2_positions - df1_positions),
    }
    
    # 打印条目统计信息
    print("\n数据条目统计:")
    for key, value in count_stats.items():
        print(f"{key}: {value}")
    
    # 遍历每一列进行对比
    for col in df1.columns[1:]:  # 跳过第一列（定位列）
        col_start_time = time.time()
        print(f"\n开始分析列: {col}")
        tp, fp, fn, tn = 0, 0, 0, 0
        
        # 计算第一个文件中存在但第二个文件中不存在的位置 - 这些将被视为FN
        # 如果只关注共同记录，则不计算缺失项
        if not only_common:
            missing_positions = df1_positions - df2_positions
            fn += len(missing_positions)
        else:
            missing_positions = set()  # 如果只关注共同记录，则创建一个空集合
        
        # 为批处理准备数据
        if use_batch:
            # 创建位置到行索引的映射，加速查找
            df1_position_index = {pos: idx for idx, pos in enumerate(df1[position_col])}
            df2_position_index = {pos: idx for idx, pos in enumerate(df2[position_col])}
            
            # 准备批处理数据
            batch_positions = list(common_positions)
            batch_text1 = [df1.iloc[df1_position_index[pos]][col] for pos in batch_positions]
            batch_text2 = [df2.iloc[df2_position_index[pos]][col] for pos in batch_positions]
            
            # 检查空值
            batch_text1_empty = [is_empty_value(t) for t in batch_text1]
            batch_text2_empty = [is_empty_value(t) for t in batch_text2]
            
            # 两者都为空的情况直接判定为TP
            both_empty = [(pos, i) for i, (pos, t1_empty, t2_empty) in enumerate(zip(batch_positions, batch_text1_empty, batch_text2_empty)) if t1_empty and t2_empty]
            for pos, idx in both_empty:
                tp += 1
                # 详细输出信息
                if verbose:
                    print(f"比较定位 {pos}: 文本1: {batch_text1[idx]} 文本2: {batch_text2[idx]}")
                    print(f"判定: 匹配 (TP) - 两者都为空")
                
                detailed_comparison.append({
                    "列名": col,
                    "定位": pos,
                    "文本1": batch_text1[idx],
                    "文本2": batch_text2[idx],
                    "相似度方法": "空值匹配",
                    "相似度分数": 1.0,
                    "判定结果": "匹配 (TP) - 两者都为空",
                    "文本1是否为空": True,
                    "文本2是否为空": True
                })
            
            # 过滤掉两者都为空的情况，准备进行相似度计算
            valid_indices = [i for i, (t1_empty, t2_empty) in enumerate(zip(batch_text1_empty, batch_text2_empty)) if not (t1_empty and t2_empty)]
            
            if valid_indices:
                valid_positions = [batch_positions[i] for i in valid_indices]
                valid_text1 = [batch_text1[i] for i in valid_indices]
                valid_text2 = [batch_text2[i] for i in valid_indices]
                valid_text1_empty = [batch_text1_empty[i] for i in valid_indices]
                valid_text2_empty = [batch_text2_empty[i] for i in valid_indices]
                
                # 先进行规则匹配
                rule_results = []
                rule_methods = []
                for t1, t2 in zip(valid_text1, valid_text2):
                    # 详细输出原始文本
                    if verbose:
                        print(f"规则匹配: 文本1: {t1} 文本2: {t2}")
                    
                    rule_result = rule_based_similarity(t1, t2)
                    if rule_result is not None:
                        if verbose:
                            print(f"规则匹配结果: {rule_result:.4f}")
                        rule_methods.append("规则匹配")
                    else:
                        rule_methods.append(None)
                    rule_results.append(rule_result)
                
                # 对于规则匹配失败的情况，使用模型计算
                need_model_indices = [i for i, res in enumerate(rule_results) if res is None]
                
                if need_model_indices:
                    need_model_positions = [valid_positions[i] for i in need_model_indices]
                    need_model_text1 = [valid_text1[i] for i in need_model_indices]
                    need_model_text2 = [valid_text2[i] for i in need_model_indices]
                    
                    # 根据列类型选择相似度计算方法
                    if col in pubmedbert_columns:
                        model_method = 'pubmedbert'
                        method_name = "PubMedBERT"
                    else:
                        model_method = 'sbert'
                        method_name = "Sentence Transformer"
                    
                    if verbose:
                        print(f"使用{method_name}计算相似度...")
                    
                    # 批量计算相似度
                    similarities = batch_calculate_similarities(
                        need_model_text1, need_model_text2, 
                        method=model_method, batch_size=batch_size
                    )
                    
                    # 将模型计算结果合并回规则结果
                    for i, sim in zip(need_model_indices, similarities):
                        rule_results[i] = sim
                        rule_methods[i] = method_name
                
                # 处理结果
                for i, (pos, t1, t2, t1_empty, t2_empty, sim, method_used) in enumerate(zip(
                    valid_positions, valid_text1, valid_text2, 
                    valid_text1_empty, valid_text2_empty, rule_results, rule_methods
                )):
                    # 确定相似度方法
                    if t1_empty or t2_empty:
                        method = "空值检测"
                    elif method_used is not None:
                        method = method_used
                    elif col in pubmedbert_columns:
                        method = "PubMedBERT"
                    else:
                        method = "Sentence Transformer"
                    
                    # 输出详细比较信息
                    if verbose:
                        print(f"比较定位 {pos}: 文本1: {t1} 文本2: {t2} {method} 相似度: {sim:.4f}")
                    
                    # 根据相似度判断匹配情况
                    if sim >= threshold:
                        tp += 1
                        status = "匹配 (TP)"
                        if verbose:
                            print(f"判定: {status}")
                    else:
                        # 一方为空的情况
                        if (t1_empty and not t2_empty) or (not t1_empty and t2_empty):
                            fp += 1
                            status = "不匹配 (FP) - 一方为空"
                            if verbose:
                                print(f"判定: {status}")
                        # 两者都不为空但相似度低
                        elif not t1_empty and not t2_empty:
                            fp += 1
                            status = "不匹配 (FP) - 相似度低"
                            if verbose:
                                print(f"判定: {status}")
                        # 其他情况
                        else:
                            tn += 1
                            status = "正确拒绝 (TN)"
                            if verbose:
                                print(f"判定: {status}")
                    
                    # 记录详细比较结果
                    detailed_comparison.append({
                        "列名": col,
                        "定位": pos,
                        "文本1": t1,
                        "文本2": t2,
                        "相似度方法": method,
                        "相似度分数": sim,
                        "判定结果": status,
                        "文本1是否为空": t1_empty,
                        "文本2是否为空": t2_empty
                    })
        else:
            # 不使用批处理，逐个比较
            for position in tqdm(common_positions, desc=f"处理列 {col}"):
                row1 = df1[df1[position_col] == position].iloc[0]
                row2 = df2[df2[position_col] == position].iloc[0]
                
                text1 = row1[col]
                text2 = row2[col]
                
                # 详细输出
                if verbose:
                    print(f"比较定位 {position}: 文本1: {text1} 文本2: {text2}")
                
                comparison_result = {
                    "列名": col,
                    "定位": position,
                    "文本1": text1,
                    "文本2": text2
                }
                
                # 检查是否都是空值
                text1_empty = is_empty_value(text1)
                text2_empty = is_empty_value(text2)
                
                # 如果两个文本都为空，视为匹配
                if text1_empty and text2_empty:
                    tp += 1
                    status = "匹配 (TP) - 两者都为空"
                    similarity_score = 1.0
                    method = "空值匹配"
                    if verbose:
                        print(f"判定: {status}")
                else:
                    # 根据列类型选择相似度计算方法
                    if col in pubmedbert_columns:
                        # 首先尝试规则匹配
                        rule_similarity = rule_based_similarity(text1, text2)
                        if rule_similarity is not None:
                            similarity_score = rule_similarity
                            method = "规则匹配"
                            if verbose:
                                print(f"{method} 相似度: {similarity_score:.4f}")
                        else:
                            # 使用 PubMedBERT 计算相似度
                            similarity_score = calculate_pubmedbert_similarity(text1, text2, tokenizer, pubmedbert_model)
                            method = "PubMedBERT"
                            if verbose:
                                print(f"{method} 相似度: {similarity_score:.4f}")
                    
                    elif col in sbert_columns:
                        # 使用 Sentence Transformer 计算相似度
                        similarity_score = calculate_sbert_similarity(text1, text2, sbert_model)
                        method = "Sentence Transformer"
                        if verbose:
                            print(f"{method} 相似度: {similarity_score:.4f}")
                    else:
                        # 默认使用 Sentence Transformer
                        similarity_score = calculate_sbert_similarity(text1, text2, sbert_model)
                        method = "Sentence Transformer (默认)"
                        if verbose:
                            print(f"{method} 相似度: {similarity_score:.4f}")
                    
                    # 根据相似度判断匹配情况
                    if similarity_score >= threshold:
                        tp += 1
                        status = "匹配 (TP)"
                        if verbose:
                            print(f"判定: {status}")
                    else:
                        # 1. 如果一个文本为空而另一个不为空，视为实际不同但需要匹配
                        if (text1_empty and not text2_empty) or (not text1_empty and t2_empty):
                            fp += 1
                            status = "不匹配 (FP) - 一方为空"
                            if verbose:
                                print(f"判定: {status}")
                        # 2. 如果两个文本都不为空但相似度低，视为FP
                        elif not text1_empty and not text2_empty:
                            fp += 1
                            status = "不匹配 (FP) - 相似度低"
                            if verbose:
                                print(f"判定: {status}")
                        # 3. 其他情况视为TN
                        else:
                            tn += 1
                            status = "正确拒绝 (TN)"
                            if verbose:
                                print(f"判定: {status}")
                
                comparison_result["相似度方法"] = method
                comparison_result["相似度分数"] = similarity_score
                comparison_result["判定结果"] = status
                comparison_result["文本1是否为空"] = text1_empty
                comparison_result["文本2是否为空"] = text2_empty
                detailed_comparison.append(comparison_result)
        
        # 仅当不选择只比较共同项时记录缺失位置的详细信息
        if not only_common:
            for position in missing_positions:
                row1 = df1[df1[position_col] == position].iloc[0]
                text1 = row1[col]
                text1_empty = is_empty_value(text1)
                
                if verbose:
                    print(f"比较定位 {position}: 文本1: {text1} 文本2: 缺失")
                    print(f"判定: 缺失 (FN)")
                
                detailed_comparison.append({
                    "列名": col,
                    "定位": position,
                    "文本1": text1,
                    "文本2": "缺失",
                    "相似度方法": "N/A",
                    "相似度分数": 0.0,
                    "判定结果": "缺失 (FN)",
                    "文本1是否为空": text1_empty,
                    "文本2是否为空": True
                })
            
        # 计算评估指标
        accuracy, precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)
        
        # 记录当前列的有效比较数
        if only_common:
            valid_comparisons = len(common_positions)
        else:
            valid_comparisons = len(common_positions) + len(missing_positions)
        
        # 输出列的统计结果
        print(f"\n列 {col} 的统计结果:")
        print(f"参与比较的有效条目数: {valid_comparisons}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确度: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1 分数: {f1:.4f}")
        print(f"列 {col} 分析耗时: {time.time() - col_start_time:.2f} 秒")
        
        # 将结果添加到 DataFrame
        results.loc["有效条目数", col] = valid_comparisons
        results.loc["TP", col] = tp
        results.loc["FP", col] = fp
        results.loc["FN", col] = fn
        results.loc["TN", col] = tn
        results.loc["Accuracy", col] = accuracy
        results.loc["Precision", col] = precision
        results.loc["Recall", col] = recall
        results.loc["F1", col] = f1
    
    # 保存结果到 Excel 文件
    print(f"\n保存结果到 {output_file}...")
    try:
        with pd.ExcelWriter(output_file) as writer:
            # 保存数据条目统计信息
            count_df = pd.DataFrame({
                "指标": list(count_stats.keys()),
                "数值": list(count_stats.values())
            })
            count_df.to_excel(writer, sheet_name='数据条目统计', index=False)
            
            # 保存汇总指标
            results.to_excel(writer, sheet_name='汇总指标', index=True)
            
            # 保存详细比较结果
            pd.DataFrame(detailed_comparison).to_excel(writer, sheet_name='详细比较结果', index=False)
            
            # 保存空值统计
            empty_stats = pd.DataFrame(detailed_comparison).groupby(['列名', '文本1是否为空', '文本2是否为空']).size().reset_index(name='数量')
            empty_stats.to_excel(writer, sheet_name='空值统计', index=False)
        
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")
        raise
    
    total_time = time.time() - total_start_time
    print(f"\n总计分析耗时: {total_time:.2f} 秒")
    
    return results, detailed_comparison

def search_specific_positions(detailed_comparison, positions=None, columns=None, status=None):

    results = pd.DataFrame(detailed_comparison)
    
    # 过滤条件
    if positions:
        results = results[results['定位'].isin(positions)]
    
    if columns:
        results = results[results['列名'].isin(columns)]
    
    if status:
        if isinstance(status, str):
            results = results[results['判定结果'].str.contains(status)]
        else:  # 假设是列表
            mask = results['判定结果'].str.contains('|'.join(status))
            results = results[mask]
    
    return results



if __name__ == "__main__":
    # 文件路径
    file1 = "./dataset/structure/compermeta1.xlsx"
    file2 = "./dataset/structure/compermodel1.xlsx"
    output_file = "./comparison_results1.xlsx"
    
    # 设置相似度阈值
    threshold = 0.9
    
    # 性能调优参数
    batch_size = 32      # 批处理大小
    use_batch = True     # 是否使用批处理
    verbose = True       # 是否输出详细信息
    only_common = True   # 是否只比较两边都有的内容
    
    # 运行比较
    results, detailed_comparison = compare_xlsx(
        file1, file2, output_file, 
        threshold, batch_size, use_batch, verbose, only_common
    )