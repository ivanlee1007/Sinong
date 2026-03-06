import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# 引入 RAG 引擎
from rag_engine import RAGEngine

# 尝试引入进度条库
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

def process_batch(
    input_file: str, 
    output_file: str, 
    db_path: str = "agri_knowledge",
    workers: int = 4,
    question_key: str = "question"
):
    """
    批量问答（含参考来源版）
    """
    
    # 1. 加载数据
    print(f"📂 正在加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 共加载 {len(data)} 条测试数据")

    # 2. 初始化引擎
    print(f"🚀 初始化 RAG 引擎 (DB: {db_path})...")
    engine = RAGEngine(
        db_path=db_path,
        llm_model="qwen-plus", 
        embedding_model="bge-m3"
    )

    start_time = time.time()

    # --- 单个任务处理函数 ---
    def process_one(index: int, item: Dict):
        query = item.get(question_key)
        result_item = item.copy()
        
        # 移除原数据可能存在的 id (防止冲突)
        if "id" in result_item:
            del result_item["id"]

        if not query:
            return index, result_item
        
        try:
            # 调用 RAG，获取 回答(answer) 和 引用(refs)
            answer, refs = engine.chat(query, top_k=5)
            
            result_item["rag_answer"] = answer
            
            # 【关键修改】保留参考内容
            result_item["rag_references"] = []
            for r in refs:
                result_item["rag_references"].append({
                    "source": r["source"],         # 文件名
                    "score": float(r["score"]),    # 相似度 (转float以兼容JSON)
                    "content": r["content"][:500]  # 截取前500字，防止文件过大
                })
            
        except Exception as e:
            print(f"❌ 处理出错: {e}")
            result_item["rag_answer"] = f"Error: {str(e)}"
            result_item["rag_references"] = []
            
        return index, result_item

    # 3. 多线程并发执行
    print(f"⚡ 开始并发处理 (线程数: {workers})...")
    
    results_with_index = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, i, item) for i, item in enumerate(data)]
        
        for future in tqdm(as_completed(futures), total=len(futures), unit="条"):
            try:
                idx, res = future.result()
                results_with_index.append((idx, res))
            except Exception as e:
                print(f"线程异常: {e}")

    # 4. 排序与保存
    results_with_index.sort(key=lambda x: x[0])
    final_results = [res for idx, res in results_with_index]

    print(f"💾 正在保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    engine.close()
    
    duration = time.time() - start_time
    print(f"✅ 全部完成！耗时: {duration:.2f}秒")


if __name__ == "__main__":
    # 配置区
    INPUT_JSON = "test_data.json"
    OUTPUT_JSON = "rag_results_with_refs.json"
    DB_PATH = "agri_knowledge"  # 你的数据库文件夹名
    
    # 自动生成测试数据(如果不存在)
    if not os.path.exists(INPUT_JSON):
        sample_data = [
            {"question": "湿地松在种植过程中有哪些需要注意的事项，在哪些环境下更易于种植？", "ground_truth": "待填充"}
        ]
        with open(INPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

    # 运行
    process_batch(
        input_file=INPUT_JSON,
        output_file=OUTPUT_JSON,
        db_path=DB_PATH,
        workers=8
    )