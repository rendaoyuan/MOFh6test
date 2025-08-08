import pandas as pd
import numpy as np
import re
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from difflib import ndiff
import logging
import warnings

# Suppress the specific message about creating a new model
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

def preprocess_text(text):
    """
    预处理文本，应用特定规则以提高比较准确性。
    
    Parameters:
        text: 需要预处理的文本字符串
        
    Returns:
        str: 预处理后的文本
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 新规则0：移除合成标题，如"Synthesis of CPM-23:"
    text = re.sub(r'^Synthesis of [\w-]+:', '', text).strip()
    
    # 规则1: 移除元素分析、IR等分析数据部分
    patterns = [
        r'Elemental analyses.*?Found:.*?\.', 
        r'Anal\.\s+found.*?%\.', 
        r'IR\s+\(KBr,\s*cm-?1\).*?\.', 
        r'IR\s+\(KBr.*?\).*?\.', 
        r'\d+\(\w+\),\s*\d+\(\w+\).*?\.',  # IR数据模式
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 规则2: 处理化合物缩写 - 替换为统一占位符
    text = re.sub(r'\[\w+.*?\]·\d+\w+', 'COMPOUND_PLACEHOLDER', text)
    text = re.sub(r'\d+·\d+\w+', 'COMPOUND_PLACEHOLDER', text)
    
    # 规则3: 处理带括号的化合物缩写
    text = re.sub(r'\(\d+-\w+·\d+\w+\)', 'ABBREVIATED_COMPOUND', text)
    text = re.sub(r'\d+-\w+\s+\d+\s+\d+\w+', 'ABBREVIATED_COMPOUND', text)
    
    # 规则3.1: 处理H3L和完整化合物名称的对应关系
    ligand_abbr_pattern = re.compile(r'H\d+L')
    ligand_matches = ligand_abbr_pattern.findall(text)
    if ligand_matches:
        for match in ligand_matches:
            text = text.replace(match, "LIGAND_PLACEHOLDER")
    
    # 新规则4: 处理化学试剂的全称和缩写
    abbreviations = re.findall(r'([^()\d]+)\s*\(([A-Z]+)\)', text)
    for full_name, abbr in abbreviations:
        text = text.replace(f"{full_name.strip()} ({abbr.strip()})", f"{abbr.strip()}")
    
    common_abbr = {
        "DMA": "N,N′-dimethylacetamide",
        "DMF": "N,N-dimethylformamide",
        "THF": "tetrahydrofuran",
        "DMSO": "dimethyl sulfoxide",
        "MeOH": "methanol",
        "EtOH": "ethanol"
    }
    # 根据需要可以进一步扩展或使用 common_abbr 做替换
    
    # 新规则5: 统一温度和时间表示
    text = re.sub(r'(\d+)\s*[°℃o\*]\s*C', r'\1C', text)
    text = re.sub(r'(\d+)\s*C\b', r'\1C', text)
    text = re.sub(r'(\d+)\s*(?:hr|hrs|hour|hours)\b', r'\1h', text)
    text = re.sub(r'(\d+)\s*(?:day|days)\b', r'\1d', text)
    
    # 新规则6: 移除多余空格、连字符等
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-(\w)', r'\1\2', text)
    
    # 新规则7: 统一温度冷却表达
    text = re.sub(r'cool(?:ed|ing)?\s+(?:down)?\s+to\s+room[\s-]*temp(?:erature)?',
                  'cooled to room temperature', text, flags=re.IGNORECASE)
    
    # 规则8: 处理产率信息中不同化合物的情况
    text = re.sub(r'\(yield:.*?based on .*?\)', 'YIELD_PLACEHOLDER', text, flags=re.IGNORECASE)
    
    # 规则9: 处理化合物缩写或完整名称的区别
    full_compound_pattern = r'tris\[\(.*?\).*?\](?:amine|acid)'
    text = re.sub(full_compound_pattern, 'COMPOUND_NAME_PLACEHOLDER', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def compare_excel_files_with_embeddings(file_a, file_b):
    """
    使用文本嵌入模型比较两个 Excel 文件中匹配行的特定列的相似度，
    应用预处理规则处理文本，先比较预处理后的文本是否完全一致，
    若不一致则使用句子嵌入和余弦相似度计算相似度。
    
    Parameters:
        file_a: 第一个 Excel 文件路径（包含 CCDC_Code, Method 等列）
        file_b: 第二个 Excel 文件路径（包含 ccdc, method 等列）
        
    Returns:
        DataFrame: 包含匹配结果和相似度得分的结果数据框
    """
    print("加载嵌入模型: PubMedBERT")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')  
    
    print(f"加载文件A: {file_a}")
    df_a = pd.read_excel(file_a)
    print(f"加载文件B: {file_b}")
    df_b = pd.read_excel(file_b)
    
    # 规范列名（小写去空格）
    df_a.columns = [col.strip().lower() for col in df_a.columns]
    df_b.columns = [col.strip().lower() for col in df_b.columns]
    
    # 识别 ID 和 Method 列
    col_a_id = [col for col in df_a.columns if 'ccdc' in col][0]
    col_a_method = [col for col in df_a.columns if 'method' in col][0]
    col_b_id = [col for col in df_b.columns if 'ccdc' in col][0]
    col_b_method = [col for col in df_b.columns if 'method' in col][0]
    
    print(f"比较: A.{col_a_method} 与 B.{col_b_method} 当 A.{col_a_id} 匹配 B.{col_b_id}")
    
    # 转换 ID 列为字符串
    df_a[col_a_id] = df_a[col_a_id].astype(str)
    df_b[col_b_id] = df_b[col_b_id].astype(str)
    
    results = []
    batch_size = 32  # 根据内存可调整
    
    # 用于批量处理匹配对
    comparison_pairs = []
    ids = []
    methods_a = []
    methods_b = []
    
    for _, row_a in df_a.iterrows():
        id_a = row_a[col_a_id]
        method_a = row_a[col_a_method]
        if pd.isna(id_a) or id_a.strip() == '':
            continue
        matches = df_b[df_b[col_b_id] == id_a]
        if not matches.empty:
            for _, row_b in matches.iterrows():
                method_b = row_b[col_b_method]
                ids.append(id_a)
                methods_a.append(method_a)
                methods_b.append(method_b)
                comparison_pairs.append((method_a, method_b))
    
    if comparison_pairs:
        print(f"处理 {len(comparison_pairs)} 个匹配的行...")
        for i in range(0, len(comparison_pairs), batch_size):
            batch = comparison_pairs[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_methods_a = methods_a[i:i+batch_size]
            batch_methods_b = methods_b[i:i+batch_size]
            
            all_texts = []
            preprocessed_texts_a = []
            preprocessed_texts_b = []
            for method_a, method_b in batch:
                processed_a = preprocess_text(method_a)
                processed_b = preprocess_text(method_b)
                preprocessed_texts_a.append(processed_a)
                preprocessed_texts_b.append(processed_b)
                all_texts.append(processed_a)
                all_texts.append(processed_b)
            
            embeddings = model.encode(all_texts, convert_to_tensor=True)
            
            for j in range(len(batch)):
                idx1 = j * 2
                idx2 = j * 2 + 1
                if pd.isna(batch_methods_a[j]) or pd.isna(batch_methods_b[j]):
                    sim_score = np.nan
                else:
                    if preprocessed_texts_a[j] == preprocessed_texts_b[j] and preprocessed_texts_a[j] != "":
                        sim_score = 1.0
                    else:
                        emb1 = embeddings[idx1].cpu().numpy().reshape(1, -1)
                        emb2 = embeddings[idx2].cpu().numpy().reshape(1, -1)
                        sim_score = cosine_similarity(emb1, emb2)[0][0]
                
                results.append({
                    'ID': batch_ids[j],
                    'Method_A': batch_methods_a[j],
                    'Method_B': batch_methods_b[j],
                    'Preprocessed_A': preprocessed_texts_a[j],
                    'Preprocessed_B': preprocessed_texts_b[j],
                    'Embedding_Similarity': sim_score,
                    'Exact_Match_After_Rules': preprocessed_texts_a[j] == preprocessed_texts_b[j] and preprocessed_texts_a[j] != ""
                })
            print(f"已处理 {min(i+batch_size, len(comparison_pairs))}/{len(comparison_pairs)} 行")
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("在两个文件之间未找到匹配的 ID。")
        return None
    
    # 计算平均得分
    avg_similarity = results_df['Embedding_Similarity'].mean()
    print(f"平均相似度得分: {avg_similarity:.4f}")
    
    # 计算完全匹配的百分比
    #exact_match_percent = results_df['Exact_Match_After_Rules'].mean() * 100
    #print(f"预处理后完全匹配的百分比: {exact_match_percent:.2f}%")
    
    # 计算高相似度(>0.9)的百分比
    #high_similarity = (results_df['Embedding_Similarity'] > 0.9).mean() * 100
    #print(f"相似度>0.9的百分比: {high_similarity:.2f}%")
    
    results_df = results_df.sort_values('Embedding_Similarity', ascending=False)
    print(f"找到 {len(results_df)} 个匹配的行。")
    return results_df

def run_preprocessing_test(text1, text2):
    """
    通过传入两个文本，测试预处理规则的效果和差异。
    
    Parameters:
        text1: 第一个文本字符串
        text2: 第二个文本字符串
    """
    print("原始文本 1:")
    print(text1)
    print("\n原始文本 2:")
    print(text2)
    
    processed_1 = preprocess_text(text1)
    processed_2 = preprocess_text(text2)
    
    print("\n预处理后文本 1:")
    print(processed_1)
    print("\n预处理后文本 2:")
    print(processed_2)
    
    is_exact_match = processed_1 == processed_2 and processed_1 != ""
    print("\n预处理后是否完全匹配:", is_exact_match)
    
    if not is_exact_match:
        print("\n差异部分:")
        for diff in ndiff(processed_1.split(), processed_2.split()):
            if diff[0] != ' ':
                print(diff)

def main():
    # 如有需要，可调用 run_preprocessing_test 来验证预处理效果：
    # 示例调用（你可以根据需要取消注释或修改输入文本）
    # run_preprocessing_test("原始文本示例 1", "原始文本示例 2")
    
    # 如果通过命令行参数提供文件路径，则使用命令行参数
    if len(sys.argv) > 2:
        file_a = sys.argv[1]
        file_b = sys.argv[2]
        if len(sys.argv) > 3:
            output_file = sys.argv[3]
        else:
            output_file = 'embedding_similarity_results.xlsx'
    else:
        # 否则使用默认路径
        file_a = './paragraphmeta.xlsx'
        file_b = './paragraphmeta.xlsx'
        output_file = './198shotresults1.xlsx'
    
    # 确保文件存在
    if not os.path.isfile(file_a):
        print(f"错误: 文件不存在 - {file_a}")
        return
    if not os.path.isfile(file_b):
        print(f"错误: 文件不存在 - {file_b}")
        return
    
    # 执行比较
    results = compare_excel_files_with_embeddings(file_a, file_b)
    
    if results is not None:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存结果
        results.to_excel(output_file, index=False)
        print(f"\n结果已保存到 {output_file}")

if __name__ == "__main__":
    main()