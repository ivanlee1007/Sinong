"""
FAISS & SQLite 对应关系诊断脚本
===================================

检测三个层次的问题：
  Level 1 - 计数一致性：FAISS 向量数 == SQLite 文档数？
  Level 2 - ID 连续性：SQLite id 是否从0开始连续无空洞？
  Level 3 - 语义一致性：随机抽样向量，重新 embed 对应文档，验证余弦相似度

用法：
  python diagnose_db.py --db agri_knowledge --model bge-m3 --samples 20
"""

import argparse
import json
import os
import sqlite3
import numpy as np

# ============================================================
#  工具函数
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def color(text, code):
    """ANSI 颜色输出"""
    return f"\033[{code}m{text}\033[0m"

def ok(msg):   return color(f"  ✅  {msg}", "32")
def warn(msg): return color(f"  ⚠️  {msg}", "33")
def err(msg):  return color(f"  ❌  {msg}", "31")
def info(msg): return color(f"  ℹ️  {msg}", "36")

# ============================================================
#  Level 1：计数一致性
# ============================================================

def check_count(faiss_index, sqlite_conn):
    print("\n" + "="*55)
    print("Level 1  计数一致性")
    print("="*55)

    faiss_total = faiss_index.ntotal
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    sqlite_total = cursor.fetchone()[0]

    print(info(f"FAISS  向量数: {faiss_total}"))
    print(info(f"SQLite 文档数: {sqlite_total}"))

    if faiss_total == sqlite_total:
        print(ok("数量一致 ✓"))
        return True
    else:
        diff = faiss_total - sqlite_total
        print(err(f"数量不一致！差值 = {diff}"))
        if diff > 0:
            print(err("  → FAISS 向量比 SQLite 多，可能存在重复入库或 SQLite 数据被删除"))
        else:
            print(err("  → SQLite 文档比 FAISS 多，可能 FAISS 索引未保存或已损坏"))
        return False

# ============================================================
#  Level 2：ID 连续性
# ============================================================

def check_id_continuity(sqlite_conn, faiss_total):
    print("\n" + "="*55)
    print("Level 2  SQLite ID 连续性")
    print("="*55)

    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id FROM documents ORDER BY id ASC")
    rows = cursor.fetchall()
    ids = [r[0] for r in rows]

    if not ids:
        print(err("SQLite 中无文档！"))
        return False

    min_id, max_id = ids[0], ids[-1]
    print(info(f"SQLite id 范围: [{min_id}, {max_id}]，共 {len(ids)} 条"))

    # FAISS 内部序号是 0-based 连续整数
    # 所以要求 SQLite id 必须恰好是 0, 1, 2, ..., N-1
    issues = []

    if min_id != 0:
        issues.append(f"最小 id 为 {min_id}（期望为 0）")

    # 检测空洞
    expected = set(range(min_id, max_id + 1))
    actual = set(ids)
    holes = sorted(expected - actual)
    if holes:
        display = holes[:10]
        suffix = f"... 共 {len(holes)} 处" if len(holes) > 10 else ""
        issues.append(f"发现 id 空洞: {display}{suffix}")

    # 检测重复
    if len(ids) != len(actual):
        issues.append(f"发现重复 id（{len(ids) - len(actual)} 个）")

    # 检测范围是否与 FAISS 大小一致
    # FAISS 序号 = 0 ~ faiss_total-1，对应 SQLite id 应为 0 ~ faiss_total-1
    if max_id != faiss_total - 1:
        issues.append(
            f"SQLite 最大 id={max_id}，但 FAISS 大小={faiss_total}（期望最大 id={faiss_total-1}）"
        )

    if issues:
        for iss in issues:
            print(err(iss))
        print(err("ID 不连续 → FAISS 序号与 SQLite id 存在错位！"))
        print(warn("  建议：重新建库，或维护单独的 faiss_pos -> sqlite_id 映射表"))
        return False
    else:
        print(ok("ID 从 0 开始连续无空洞，与 FAISS 大小一致 ✓"))
        return True

# ============================================================
#  Level 3：语义一致性（核心验证）
# ============================================================

def check_semantic_alignment(faiss_index, sqlite_conn, embedder, n_samples=20, threshold=0.90):
    print("\n" + "="*55)
    print(f"Level 3  语义一致性（随机抽样 {n_samples} 条）")
    print("="*55)

    faiss_total = faiss_index.ntotal
    if faiss_total == 0:
        print(err("FAISS 索引为空，跳过"))
        return

    # 随机采样位置
    sample_positions = np.random.choice(
        faiss_total,
        size=min(n_samples, faiss_total),
        replace=False
    ).tolist()

    cursor = sqlite_conn.cursor()

    results = []
    print(f"\n  {'FAISS位置':>10}  {'SQLite id':>10}  {'相似度':>8}  {'状态':>6}  来源文件")
    print("  " + "-"*70)

    for pos in sorted(sample_positions):
        # 从 FAISS 取出该位置的向量（使用 reconstruct）
        try:
            stored_vec = faiss_index.reconstruct(int(pos))
        except Exception as e:
            print(warn(f"  pos={pos}: 无法 reconstruct 向量（{e}），跳过"))
            continue

        # 按 FAISS 位置去 SQLite 查对应文档（假设 id == pos）
        cursor.execute(
            "SELECT id, content, source FROM documents WHERE id = ?", (pos,)
        )
        row = cursor.fetchone()

        if row is None:
            print(f"  {pos:>10}  {'NOT FOUND':>10}  {'—':>8}  {'❌':>6}  SQLite 中无此 id")
            results.append(("missing", pos))
            continue

        doc_id, content, source = row

        # 重新 embed 文档内容（用 doc instruction）
        try:
            recomputed_vec = embedder.embed_documents([content], show_progress=False)[0]
        except Exception as e:
            print(warn(f"  pos={pos}: embed 失败（{e}），跳过"))
            continue

        sim = cosine_similarity(stored_vec, recomputed_vec)
        status = "✅" if sim >= threshold else "❌"
        source_short = os.path.basename(source)[:30] if source else "—"

        print(f"  {pos:>10}  {doc_id:>10}  {sim:>8.4f}  {status:>6}  {source_short}")
        results.append((sim, pos))

    # 汇总
    valid = [r[0] for r in results if isinstance(r[0], float)]
    missing = [r for r in results if r[0] == "missing"]

    print("\n  " + "-"*70)
    if valid:
        avg_sim = np.mean(valid)
        low_sim = [v for v in valid if v < threshold]
        print(info(f"平均相似度: {avg_sim:.4f}（阈值: {threshold}）"))

        if missing:
            print(err(f"有 {len(missing)} 个 FAISS 位置在 SQLite 中找不到对应文档 → ID 严重错位"))

        if not low_sim and not missing:
            print(ok("全部样本语义一致，FAISS 与 SQLite 对应正确 ✓"))
        elif low_sim:
            print(err(f"有 {len(low_sim)}/{len(valid)} 个样本相似度低于阈值"))
            print(warn("  可能原因：embedding 模型与入库时不同，或向量对应了错误的文档"))
        if avg_sim < 0.5:
            print(err("  ⛔ 平均相似度极低（< 0.5），FAISS 与 SQLite 几乎确认已错位！"))
            print(err("  建议：清空数据库，重新入库"))
    else:
        print(err("无有效样本，请检查数据库状态"))

# ============================================================
#  Level 4：搜索结果验证（端到端）
# ============================================================

def check_search_sanity(faiss_index, sqlite_conn, embedder, test_queries=None):
    print("\n" + "="*55)
    print("Level 4  端到端搜索验证（搜索结果是否有意义）")
    print("="*55)

    if test_queries is None:
        # 从数据库里随机取几条文档的前50字作为"查询"——应该能检索到自身
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT id, content, source FROM documents ORDER BY RANDOM() LIMIT 3")
        rows = cursor.fetchall()
        test_queries = [(row[1][:80], row[0], row[2]) for row in rows]  # (query_text, expected_id, source)
        print(info("未指定测试查询，自动从库中随机取3条文档片段作为查询（期望命中自身）"))
    
    for query_text, expected_id, source in test_queries:
        query_vec = embedder.embed_query(query_text).reshape(1, -1).astype(np.float32)
        scores, indices = faiss_index.search(query_vec, 5)
        
        top_ids = [int(i) for i in indices[0] if i >= 0]
        top_scores = [float(s) for s in scores[0] if s > 0]
        
        hit = expected_id in top_ids
        rank = top_ids.index(expected_id) + 1 if hit else -1
        
        print(f"\n  查询: 「{query_text[:50]}...」")
        print(f"  期望 id: {expected_id}（来源: {os.path.basename(source)}）")
        print(f"  Top-5 id: {top_ids}，分数: {[f'{s:.3f}' for s in top_scores]}")
        
        if hit and rank == 1:
            print(ok(f"命中！排名第 {rank} ✓"))
        elif hit:
            print(warn(f"命中但排名第 {rank}，不是第一名（可能存在更相似的文档，或向量漂移）"))
        else:
            print(err("未命中！文档查不到自身 → 向量对应严重异常"))

# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FAISS & SQLite 对应关系诊断")
    parser.add_argument("--db",      default="agri_knowledge",  help="数据库路径前缀（不含扩展名）")
    parser.add_argument("--model",   default="bge-m3",           help="Embedding 模型名称（与入库时一致）")
    parser.add_argument("--samples", type=int, default=20,       help="语义一致性抽样数量")
    parser.add_argument("--threshold", type=float, default=0.90, help="相似度合格阈值（默认0.90）")
    parser.add_argument("--skip-embed", action="store_true",     help="跳过 Level 3/4（不加载 embedding 模型，速度快）")
    args = parser.parse_args()

    faiss_path  = f"{args.db}.faiss"
    sqlite_path = f"{args.db}.db"
    import os
    # ---- 检查文件存在 ----
    print(color("\n🔍 FAISS & SQLite 对应关系诊断工具", "1;36"))
    print(f"  DB 前缀: {args.db}")
    print(f"  FAISS : {faiss_path}  {'✅' if os.path.exists(faiss_path) else '❌ 不存在'}")
    print(f"  SQLite: {sqlite_path}  {'✅' if os.path.exists(sqlite_path) else '❌ 不存在'}")

    if not os.path.exists(faiss_path) or not os.path.exists(sqlite_path):
        print(err("文件不存在，请检查路径"))
        return

    # ---- 加载 FAISS ----
    try:
        import faiss
        faiss_index = faiss.read_index(faiss_path)
        print(info(f"FAISS 加载成功，维度={faiss_index.d}，向量数={faiss_index.ntotal}"))
    except Exception as e:
        print(err(f"FAISS 加载失败: {e}"))
        return

    # ---- 连接 SQLite ----
    try:
        conn = sqlite3.connect(sqlite_path)
        print(info("SQLite 连接成功"))
    except Exception as e:
        print(err(f"SQLite 连接失败: {e}"))
        return

    # ---- Level 1 ----
    count_ok = check_count(faiss_index, conn)

    # ---- Level 2 ----
    id_ok = check_id_continuity(conn, faiss_index.ntotal)

    # ---- Level 3 & 4（需要加载 embedding 模型）----
    if not args.skip_embed:
        # 检查 FAISS 是否支持 reconstruct（IVF 类型默认不支持）
        can_reconstruct = hasattr(faiss_index, 'reconstruct')
        if not can_reconstruct:
            print(warn("当前 FAISS 索引类型不支持 reconstruct，跳过 Level 3"))
        else:
            try:
                # 测试是否真的能 reconstruct
                faiss_index.reconstruct(0)
            except Exception:
                can_reconstruct = False
                print(warn("reconstruct 调用失败（可能是 IVF 索引），跳过 Level 3"))

        print("\n正在加载 Embedding 模型（用于语义验证）...")
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from clients import EmbeddingClient
            embedder = EmbeddingClient(args.model)

            if can_reconstruct:
                check_semantic_alignment(
                    faiss_index, conn, embedder,
                    n_samples=args.samples,
                    threshold=args.threshold
                )

            check_search_sanity(faiss_index, conn, embedder)

        except Exception as e:
            print(err(f"Embedding 模型加载失败: {e}"))
            print(warn("可使用 --skip-embed 跳过语义验证，只做计数和 ID 检查"))
    else:
        print(warn("\n已跳过 Level 3/4（--skip-embed）"))

    # ---- 总结 ----
    print("\n" + "="*55)
    print("诊断总结")
    print("="*55)
    if count_ok and id_ok:
        print(ok("计数和 ID 检查均通过，如仍有检索问题请关注 Level 3/4 语义验证结果"))
    else:
        print(err("存在结构性问题，检索结果可能完全不可信"))
        print()
        print(color("  修复建议：", "1;33"))
        if not count_ok:
            print(warn("  1. 清空 .faiss 和 .db 文件，重新从原始文档入库"))
        if not id_ok:
            print(warn("  2. 在 VectorDatabase 中增加 faiss_pos → sqlite_id 的映射表"))
            print(warn("     例如：self._faiss_to_sqlite = []  入库时 append(current_id)"))
            print(warn("     检索时：sqlite_id = self._faiss_to_sqlite[faiss_pos]"))

    conn.close()

if __name__ == "__main__":
    main()