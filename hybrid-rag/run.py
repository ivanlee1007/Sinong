import os
import sys
import argparse
from typing import List, Dict, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from rag_engine import RAGEngine
from clients import LLMClient, EmbeddingClient
from cleaner import TextCleaner
from faiss_database import VectorDatabase
from chunker import SmartChunker

class DocumentStore:
    def __init__(self, db_path="agri_knowledge", embedding_model="bge-m3", 
                 llm_model="qwen3-next", index_type="flat", use_llm=True, 
                 clean_text=True, percentile: int = 15, margin: float = 0.1):
        
        print("=" * 60 + "\n初始化文档存储\n" + "=" * 60)
        self.embedder = EmbeddingClient(embedding_model)
        self.llm = LLMClient(llm_model) if use_llm else None
        
        self.cleaner = TextCleaner(remove_frontmatter=True, remove_toc=True, clean_pdf_artifacts=True) if clean_text else None
        
        self.chunker = SmartChunker(
            embedding_client=self.embedder, llm_client=self.llm,
            breakpoint_percentile_threshold=percentile, arbiter_margin=margin,
            merge_broken_paragraphs=True
        )
        
        self.database = VectorDatabase(
            embedding_client=self.embedder, index_type=index_type, 
            keyword_method="hybrid", db_path=db_path
        )
        print("初始化完成!\n")
    
    def diagnose(self, text: str = None, file_path: str = None):
        target_text = ""
        if text: target_text = text
        elif file_path:
            with open(file_path, 'r', encoding='utf-8') as f: target_text = f.read()
        if self.cleaner: target_text = self.cleaner.clean(target_text, show_progress=False).text
        return self.chunker.diagnose(target_text)
    
    def add_document(self, text: str, source: str = "", show_progress: bool = False) -> Tuple[int, Counter]:
        # 1. 清洗
        extracted_title = ""
        if self.cleaner: 
            clean_result = self.cleaner.clean(text, source=source, show_progress=False)
            text = clean_result.text
            extracted_title = clean_result.title

        # 2. 构建文档级关键词档案 (两阶段策略的第一阶段)
        doc_profile = None
        if self.database.keyword_extractor:
            doc_profile = self.database.keyword_extractor.build_doc_profile(
                text, max_candidates=200, top_n_stopwords=50
            )

        # 3. 智能分块
        final_title = extracted_title if extracted_title else source
        chunks = self.chunker.chunk(
            text, source=source, title=final_title, 
            remove_frontmatter=False, show_progress=False
        )
        
        # 4. 统计策略
        method_stats = Counter([c['metadata'].get('method', 'unknown') for c in chunks])

        # 5. 入库 (传入 doc_profile，让 chunk 级关键词从文档级候选池中筛选)
        embed_batch_size = 5 if self.embedder.dimension >= 2048 else 32
        self.database.add_documents(
            chunks, show_progress=show_progress, 
            embed_batch_size=embed_batch_size, 
            doc_profile=doc_profile
        )
        
        return len(chunks), method_stats
    
    def add_file(self, file_path: str, **kwargs) -> Tuple[int, Counter]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
            source = kwargs.pop('source', os.path.basename(file_path))
            return self.add_document(text, source=source, **kwargs)
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return 0, Counter()
    
    def add_directory(self, dir_path: str, max_workers: int = 4, **kwargs) -> int:
        files = [os.path.join(r, f) for r, _, fs in os.walk(dir_path) for f in fs if f.endswith(('.md', '.txt'))]
        if not files: return 0

        # ============================================================
        # 断点续传：跳过已入库的文件
        # ============================================================
        ingested_sources = self.database.get_ingested_sources()
        if ingested_sources:
            before = len(files)
            files = [f for f in files if os.path.basename(f) not in ingested_sources]
            skipped = before - len(files)
            if skipped > 0:
                print(f"⏩ 断点续传：跳过已入库的 {skipped} 个文件，剩余 {len(files)} 个待处理")
            if not files:
                print("✅ 所有文件已入库，无需重复处理")
                return 0

        # ============================================================
        # 第一轮：语料库级 IDF
        # - 断点续传（目录文件数 == IDF记录的文档数）→ 直接复用
        # - 新增文档（目录文件数 > IDF记录的文档数）→ 增量更新（只扫描新文件）
        # - 无 IDF 文件 → 全量构建
        # ============================================================
        idf_path = f"{self.database.db_path}_idf.json"
        extractor = self.database.keyword_extractor
        
        all_files_in_dir = [os.path.join(r, f) for r, _, fs in os.walk(dir_path) 
                            for f in fs if f.endswith(('.md', '.txt'))]
        total_file_count = len(all_files_in_dir)
        
        if extractor.load_idf(idf_path):
            if extractor.corpus_doc_count >= total_file_count:
                # 断点续传：IDF 文档数 >= 目录文件数，直接复用
                print(f"📊 IDF 表匹配 (IDF基于{extractor.corpus_doc_count}篇, "
                      f"目录{total_file_count}篇)，直接复用")
            else:
                # 新增文档：增量更新，只读取新文件
                new_file_count = total_file_count - extractor.corpus_doc_count
                print(f"📊 检测到新增约 {new_file_count} 篇文档，增量更新 IDF...")
                
                # 新文件 = 待入库文件（已在断点续传阶段过滤出来了）
                new_texts = []
                for f in tqdm(files, desc="扫描新文档", unit="file"):
                    try:
                        with open(f, 'r', encoding='utf-8') as fh:
                            text = fh.read()
                        if self.cleaner:
                            try:
                                clean_result = self.cleaner.clean(
                                    text, source=os.path.basename(f), show_progress=False)
                                text = clean_result.text
                            except Exception:
                                pass
                        new_texts.append(text)
                    except Exception:
                        pass
                
                if new_texts:
                    extractor.update_idf(new_texts, debug=True)
                    extractor.save_idf(idf_path)
                    del new_texts
        else:
            # 无 IDF 文件：全量构建（扫描整个目录）
            print(f"\n📊 第一轮：扫描 {total_file_count} 个文件构建 IDF 表...")
            all_texts = []
            for f in tqdm(all_files_in_dir, desc="扫描文档", unit="file"):
                try:
                    with open(f, 'r', encoding='utf-8') as fh:
                        text = fh.read()
                    if self.cleaner:
                        try:
                            clean_result = self.cleaner.clean(
                                text, source=os.path.basename(f), show_progress=False)
                            text = clean_result.text
                        except Exception:
                            pass
                    all_texts.append(text)
                except Exception:
                    pass
            
            if all_texts:
                extractor.build_corpus_idf(all_texts, debug=True)
                extractor.save_idf(idf_path)
                del all_texts

        # ============================================================
        # 第二轮：入库（带增量保存）
        # ============================================================
        print(f"\n📥 第二轮：入库 {len(files)} 个文件，启动 {max_workers} 个线程...")
        total_chunks = 0
        success_count = 0
        fail_count = 0
        global_method_stats = Counter()
        save_interval = max(50, len(files) // 20)  # 每处理 5% 或 50 个文件保存一次
        since_last_save = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.add_file, f, show_progress=False, **kwargs): f for f in files}
            with tqdm(total=len(files), unit="file", desc="入库总进度") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        chunks_added, file_stats = future.result()
                        total_chunks += chunks_added
                        global_method_stats.update(file_stats)
                        success_count += 1
                        since_last_save += 1
                    except Exception as e:
                        print(f"\n❌ 处理出错: {os.path.basename(file_path)} - {str(e)}")
                        fail_count += 1
                    finally:
                        pbar.update(1)
                    
                    # 增量保存 FAISS 索引（SQLite 是 autocommit，但 FAISS 需手动保存）
                    if since_last_save >= save_interval:
                        try:
                            self.database.save()
                            since_last_save = 0
                        except Exception as e:
                            print(f"\n⚠️ 增量保存失败: {e}")
        
        # 最终保存
        self.database.save()
        
        print("\n" + "="*60)
        print(f"📊 任务完成统计报告")
        print("="*60)
        print(f"✅ 处理文件: {success_count} 成功 / {fail_count} 失败")
        print(f"📚 总计生成: {total_chunks} 个 Chunk")
        print("-" * 60)
        print(f"🧠 分块策略分布:")
        method_cn_map = {
            "adaptive_merge": "自适应合并 (Semantic Merge)",
            "arbiter_merge":  "LLM仲裁-合并 (Arbiter Merge)",
            "arbiter_split":  "LLM仲裁-切分 (Arbiter Split)",
            "paragraph":      "基础段落 (Paragraph)",
            "model_limit":    "物理极限切分 (Safety Split)",
            "title_glue":     "标题强制粘合 (Title Glue)"
        }
        if not global_method_stats: print("   (无数据)")
        else:
            for method, count in global_method_stats.most_common():
                cn_name = method_cn_map.get(method, method)
                ratio = (count / total_chunks * 100) if total_chunks > 0 else 0
                print(f"   • {cn_name:<30}: {count:>5} ({ratio:.1f}%)")
        print("="*60 + "\n")
        return total_chunks

    def search(self, query: str, top_k: int = 5, hybrid: bool = True):
        return self.database.search_hybrid(query, top_k) if hybrid else self.database.search(query, top_k)
    def save(self, path: str): self.database.save(path)
    def print_statistics(self):
        stats = self.database.get_statistics()
        print("\nDB Stats:", stats)


# ================================================================
# 命令行接口
# ================================================================

def cmd_add(args):
    """添加文档（增量）"""
    store = DocumentStore(
        db_path=args.db, embedding_model=args.embedding, llm_model=args.llm, 
        index_type=args.index, use_llm=not args.no_llm, 
        percentile=args.percentile, margin=args.margin
    )
    if args.file: store.add_file(args.file)
    elif args.directory: store.add_directory(args.directory, max_workers=args.workers)
    store.save(args.db)
    if args.file: print("\n(单文件入库完成)")

def cmd_new(args):
    """新建数据库（覆盖旧的）"""
    db_path = args.db
    
    # 1. 暴力清理旧文件
    files_to_remove = [
        f"{db_path}.db", 
        f"{db_path}.faiss", 
        f"{db_path}.faiss.pkl",  # 防止有些 FAISS 版本产生 pickle
        f"{db_path}_idf.json",   # 语料库 IDF 表
    ]
    
    print(f"\n⚠️  正在创建新数据库: {db_path} (将覆盖旧文件)")
    cleaned = False
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"   🗑️  已删除旧文件: {f}")
                cleaned = True
            except Exception as e:
                print(f"   ❌ 删除失败 {f}: {e}")
    
    if not cleaned:
        print("   (未发现旧库，将直接创建)")

    # 2. 初始化（会自动创建空库）
    store = DocumentStore(
        db_path=args.db, embedding_model=args.embedding, llm_model=args.llm, 
        index_type=args.index, use_llm=not args.no_llm, 
        percentile=args.percentile, margin=args.margin
    )
    
    # 3. 如果指定了文件，顺便入库（复用 add 逻辑）
    if args.file: 
        print(f"   📥 立即导入文件: {args.file}")
        store.add_file(args.file)
    elif args.directory: 
        print(f"   📥 立即导入目录: {args.directory}")
        store.add_directory(args.directory, max_workers=args.workers)
    
    store.save(args.db)
    print(f"\n✅ 新数据库已就绪: {args.db}")


def cmd_diagnose(args):
    store = DocumentStore(
        db_path=args.db, embedding_model=args.embedding, use_llm=True, 
        percentile=args.percentile, margin=args.margin
    )
    store.diagnose(file_path=args.file)

def cmd_info(args):
    db = VectorDatabase(db_path=args.db)
    stats = db.get_statistics()
    print("\n" + "="*60 + "\n数据库信息\n" + "="*60)
    for k, v in stats.items():
        if k == 'top_keywords': continue
        print(f"  {k}: {v}")

def cmd_search(args):
    store = DocumentStore(db_path=args.db, embedding_model=args.embedding, use_llm=False)
    results = store.search(args.query, top_k=args.top_k, hybrid=not args.vector_only)
    print(f"\n搜索结果 (query: {args.query}):\n")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] 相似度: {score:.3f} | 标题: {doc.metadata.get('title', 'N/A')}")
        print(f"    {doc.content[:150]}...")
        print()

def cmd_chat(args):
    try:
        engine = RAGEngine(db_path=args.db, llm_model=args.llm, embedding_model=args.embedding)
        print("\n🤖 RAG 智能问答系统 (输入 'exit' 退出)")
        history = []
        while True:
            try:
                query = input("\n🙋 请提问: ").strip()
                if query.lower() in ['exit', 'quit', 'q']: break
                if not query: continue
                answer, refs = engine.chat(query, top_k=args.top_k, history=history)
                print("\n🤖 回答:\n" + "-" * 40 + "\n" + answer + "\n" + "-" * 40)
                if args.show_source and refs:
                    print("\n📖 参考资料:")
                    for ref in refs:
                        print(f"  [{ref['id']}] {ref['source']} (Score: {ref['score']:.2f})")
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": answer})
            except KeyboardInterrupt: break
            except Exception as e: print(f"\n❌ 出错: {e}")
    finally:
        if 'engine' in locals(): engine.close()

def cmd_export(args):
    import json
    db = VectorDatabase(db_path=args.db)
    stats = db.get_statistics()
    if stats["total_documents"] == 0: return
    all_docs = []
    for doc in db.iter_documents(): all_docs.append(doc.to_dict())
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 成功导出 {len(all_docs)} 条文档到: {args.output}")
    db.close()


def cmd_predict(args):
    """批量问答预测"""
    import json
    import time
    
    input_file = args.input
    output_file = args.output
    question_key = args.question_key
    
    # 1. 加载数据
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    print(f"📂 正在加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"📊 共加载 {len(data)} 条测试数据")
    
    # 2. 初始化引擎
    print(f"🚀 初始化 RAG 引擎 (DB: {args.db}, LLM: {args.llm})...")
    engine = RAGEngine(
        db_path=args.db,
        llm_model=args.llm,
        embedding_model=args.embedding
    )
    
    start_time = time.time()
    
    # 3. 单条处理函数
    def process_one(index, item):
        query = item.get(question_key)
        result_item = item.copy()
        if "id" in result_item:
            del result_item["id"]
        
        if not query:
            return index, result_item
        
        try:
            answer, refs = engine.chat(query, top_k=args.top_k)
            result_item["rag_answer"] = answer
            result_item["rag_references"] = [
                {
                    "source": r["source"],
                    "score": float(r["score"]),
                    "content": r["content"][:500]
                }
                for r in refs
            ]
        except Exception as e:
            print(f"❌ 处理第{index}条出错: {e}")
            result_item["rag_answer"] = f"Error: {str(e)}"
            result_item["rag_references"] = []
        
        return index, result_item
    
    # 4. 并发执行
    workers = args.workers
    print(f"⚡ 开始并发处理 (线程数: {workers})...")
    
    results_with_index = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, i, item) for i, item in enumerate(data)]
        for future in tqdm(as_completed(futures), total=len(futures), unit="条", desc="批量预测"):
            try:
                idx, res = future.result()
                results_with_index.append((idx, res))
            except Exception as e:
                print(f"线程异常: {e}")
    
    # 5. 排序保存
    results_with_index.sort(key=lambda x: x[0])
    final_results = [res for _, res in results_with_index]
    
    print(f"💾 正在保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    engine.close()
    
    duration = time.time() - start_time
    total = len(final_results)
    errors = sum(1 for r in final_results if r.get("rag_answer", "").startswith("Error:"))
    print(f"\n✅ 批量预测完成！共 {total} 条，成功 {total - errors}，失败 {errors}，耗时 {duration:.1f}秒")


def main():
    parser = argparse.ArgumentParser(description="RAG知识库管理工具")
    parser.add_argument('--db', default='agri_knowledge', help='数据库路径')
    
    subparsers = parser.add_subparsers(dest='command')
    
    # 1. Add (增量添加)
    add = subparsers.add_parser('add', help='【增量】添加文档')
    add.add_argument('--file', help='文件路径')
    add.add_argument('--directory', help='文件夹路径')
    add.add_argument('--embedding', default='bge-m3')
    add.add_argument('--llm', default='qwen3-next')
    add.add_argument('--index', default='flat')
    add.add_argument('--workers', type=int, default=4)
    add.add_argument('--no-llm', action='store_true', help='禁用LLM裁判')
    add.add_argument('--percentile', type=int, default=15)
    add.add_argument('--margin', type=float, default=0.1)
    
    # 2. New (新建/覆盖) - 继承 add 的参数以便立即入库
    new = subparsers.add_parser('new', help='【新建】清空旧库并新建(可同时导入)')
    # 复用 add 的参数定义
    new.add_argument('--file', help='文件路径(可选)')
    new.add_argument('--directory', help='文件夹路径(可选)')
    new.add_argument('--embedding', default='bge-m3')
    new.add_argument('--llm', default='qwen3-next')
    new.add_argument('--index', default='flat')
    new.add_argument('--workers', type=int, default=4)
    new.add_argument('--no-llm', action='store_true')
    new.add_argument('--percentile', type=int, default=15)
    new.add_argument('--margin', type=float, default=0.1)
    
    # Diagnose
    dia = subparsers.add_parser('diagnose', help='诊断分块效果')
    dia.add_argument('--file', required=True, help='文件路径')
    dia.add_argument('--embedding', default='bge-m3')
    dia.add_argument('--percentile', type=int, default=15)
    dia.add_argument('--margin', type=float, default=0.1)

    # Info, Search, Chat, Export
    subparsers.add_parser('info', help='数据库信息')
    
    search = subparsers.add_parser('search', help='搜索')
    search.add_argument('query')
    search.add_argument('--top-k', type=int, default=5)
    search.add_argument('--embedding', default='bge-m3')
    search.add_argument('--vector-only', action='store_true')
    
    chat = subparsers.add_parser('chat', help='问答')
    chat.add_argument('--top-k', type=int, default=5)
    chat.add_argument('--llm', default='qwen-plus')
    chat.add_argument('--embedding', default='bge-m3')
    chat.add_argument('--show-source', action='store_true', default=True)

    export = subparsers.add_parser('export', help='导出')
    export.add_argument('--output', default='export.json')

    # Predict (批量问答)
    predict = subparsers.add_parser('predict', help='批量问答预测')
    predict.add_argument('--input', required=True, help='输入JSON文件路径')
    predict.add_argument('--output', default='rag_results.json', help='输出JSON文件路径')
    predict.add_argument('--llm', default='qwen-plus', help='生成模型')
    predict.add_argument('--embedding', default='bge-m3')
    predict.add_argument('--top-k', type=int, default=5, help='检索数量')
    predict.add_argument('--workers', type=int, default=4, help='并发线程数')
    predict.add_argument('--question-key', default='question', help='JSON中问题字段名')

    args = parser.parse_args()
    if not args.command: parser.print_help(); return

    commands = {
        'add': cmd_add, 'new': cmd_new, 
        'diagnose': cmd_diagnose, 'info': cmd_info,
        'search': cmd_search, 'chat': cmd_chat, 'export': cmd_export,
        'predict': cmd_predict
    }
    commands[args.command](args)

if __name__ == "__main__":
    main()