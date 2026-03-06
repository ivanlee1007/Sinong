"""
RAG 引擎 - 检索增强生成

查询流程：
  1a. 从原始 query 提取加权关键词（不经过 HyDE，避免幻觉污染）
  1b. LLM 生成简短假设性回答文档 (HyDE)，仅用于向量检索扩展
  2.  三路向量检索：query embedding(主) + HyDE embedding(补) + 关键词(辅)
  3.  融合排序 → 构建 context → LLM 生成最终回答
"""
from typing import List, Dict, Tuple
import json
import re
import numpy as np
from clients import LLMClient
from faiss_database import VectorDatabase


class RAGEngine:
    def __init__(self,
                 db_path: str = "knowledge_base",
                 llm_model: str = "qwen-plus",
                 embedding_model: str = "bge-m3"):

        print(f"正在初始化 RAG 引擎 (DB: {db_path})...")

        # 1. 初始化向量数据库
        from clients import EmbeddingClient
        self.embedder = EmbeddingClient(embedding_model)
        self.db = VectorDatabase(
            embedding_client=self.embedder,
            db_path=db_path
        )

        # 2. 初始化生成模型
        self.llm = LLMClient(llm_model)

        # 系统提示词
        self.system_prompt = """你是一个专业的知识库问答助手。请基于提供的【参考资料】回答用户的问题。
规则：
1. 必须基于提供的资料回答，不要编造信息。
2. 如果资料中没有答案，请明确说明"知识库中未找到相关信息"。
3. 回答要逻辑清晰，结构化。
4. 引用资料时，请注明来源文件名。"""

    # ================================================================
    #  阶段 1：查询增强 — 关键词提取 + HyDE 生成（两个独立步骤）
    # ================================================================

    def _extract_keywords(self, query: str) -> List[Tuple[str, float]]:
        """
        从【原始 query】提取带权重的关键词。
        
        与旧版的区别：关键词直接从用户问题提取，而非从 HyDE 假设文档提取。
        这样可以避免 HyDE 幻觉内容（如错误术语、虚构专有名词）污染关键词，
        保证关键词路径的精确性。
        """
        prompt = f"""请从以下用户问题中提取 3-6 个核心搜索关键词或短语。

要求：
- 直接从问题原文中抽取，不要补充或推断问题中没有的内容
- 保留完整的专有名词（物种名、地名、技术术语等）
- 按重要性赋予权重（0.5-1.0），核心词权重高
- 只输出 JSON，不要有任何其他内容

问题："{query}"

JSON格式：
{{"keywords": [{{"keyword": "词语", "weight": 0.9}}, ...]}}"""

        try:
            response = self.llm.complete(prompt, max_tokens=200, temperature=0.1).strip()
            if "```" in response:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
                if match:
                    response = match.group(1).strip()
            data = json.loads(response)
            keywords = []
            for item in data.get("keywords", []):
                kw = item.get("keyword", "").strip()
                wt = float(item.get("weight", 0.5))
                if kw:
                    keywords.append((kw, wt))
            return keywords
        except Exception as e:
            print(f"⚠️ 关键词提取失败: {e}，跳过关键词路径")
            return []

    def _generate_hyde(self, query: str) -> str:
        """
        生成简短的假设性回答文档（HyDE），仅用于向量检索扩展。

        与旧版的区别：
        - 字数限制从 100-200 字缩短到 50-80 字，减少幻觉空间
        - 明确要求"只写你有把握的内容"，不确定的留空
        - 关键词已在 _extract_keywords 中单独提取，HyDE 只负责语义扩展
        """
        prompt = f"""请用 50-80 字写一段关于以下问题的专业描述，用于文档检索。

要求：
- 只写你有把握的内容，不要编造具体数据或专有名词
- 使用专业术语，风格像教科书或技术手册
- 不确定的内容宁可省略，也不要猜测

问题："{query}"

描述（直接输出文字，不要 JSON 或标题）："""

        try:
            hypo_doc = self.llm.complete(prompt, max_tokens=150, temperature=0.2).strip()
            # 防御：如果模型还是输出了 JSON 格式，降级到原始 query
            if hypo_doc.startswith('{') or len(hypo_doc) < 10:
                return query
            return hypo_doc
        except Exception as e:
            print(f"⚠️ HyDE 生成失败: {e}，使用原始 query")
            return query

    def _enhance_query(self, query: str) -> Tuple[str, List[Tuple[str, float]]]:
        """
        查询增强入口：并发执行关键词提取和 HyDE 生成。
        两个步骤独立，互不干扰，失败互不影响。
        返回: (hypothetical_doc, weighted_keywords)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        hypo_doc = query
        weighted_keywords = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_kw   = executor.submit(self._extract_keywords, query)
            fut_hyde = executor.submit(self._generate_hyde, query)

            try:
                weighted_keywords = fut_kw.result(timeout=15)
            except Exception as e:
                print(f"⚠️ 关键词提取超时/失败: {e}")

            try:
                hypo_doc = fut_hyde.result(timeout=15)
            except Exception as e:
                print(f"⚠️ HyDE 生成超时/失败: {e}")

        return hypo_doc, weighted_keywords

    # ================================================================
    #  阶段 2：检索 — HyDE embedding + 关键词双路召回
    # ================================================================

    def _retrieve(self, query: str, hypo_doc: str,
                  weighted_keywords: List[Tuple[str, float]],
                  top_k: int) -> List[Tuple]:
        """
        三路检索:
          路径A1: 原始 query 的 embed_query() 向量（带 query instruction，是主力）
          路径A2: HyDE 假设文档的 embed_documents() 向量（文档空间，召回互补）
          路径B:  关键词 SQLite 独立召回（精确补充）
          融合:   A1/A2 合并去重后由 search_hybrid 完成关键词融合
        """
        # ── 路径 A1：原始 query 向量（embed_query 带 instruction，主力路径）──
        query_embedding = None
        try:
            query_embedding = self.embedder.embed_query(query)  # shape: (D,)
        except Exception as e:
            print(f"⚠️ query embedding 失败: {e}")

        # ── 路径 A2：HyDE 假设文档向量（embed_documents，文档空间）──
        # 关键修复：必须使用 embed_documents 而非 embed(instruction="")
        # 原因：入库时用的是 doc_instruction，查询用 query_instruction；
        #       HyDE 假设文档在语义上是"文档"而非"查询"，应与入库向量在同一空间
        hyde_embedding = None
        if hypo_doc and hypo_doc != query:
            try:
                hyde_embedding = self.embedder.embed_documents(
                    [hypo_doc], show_progress=False
                )[0]  # shape: (D,)
            except Exception as e:
                print(f"⚠️ HyDE embedding 失败: {e}")

        # ── 合并 A1 + A2，送入混合检索 ──
        # search_hybrid 的 query_embedding 参数作为主向量路径
        # 当两路向量都存在时，额外检索 HyDE 路径并补充新文档到候选集
        if query_embedding is not None and hyde_embedding is not None:
            # 两路向量分别检索
            results_query = self.db.search(top_k=top_k * 2, query_embedding=query_embedding)
            results_hyde  = self.db.search(top_k=top_k * 2, query_embedding=hyde_embedding)

            # 合并：取两路最高分，避免互相覆盖
            merged: dict = {}
            for doc, score in results_query:
                merged[doc.id] = (doc, score)
            for doc, score in results_hyde:
                if doc.id not in merged:
                    merged[doc.id] = (doc, score)
                else:
                    merged[doc.id] = (doc, max(merged[doc.id][1], score))

            # 按合并分数预排序（search_hybrid 会在此基础上叠加关键词分数）
            pre_sorted = sorted(merged.values(), key=lambda x: -x[1])
            print(f"  向量候选: query路径={len(results_query)}, HyDE路径={len(results_hyde)}, "
                  f"合并去重后={len(pre_sorted)}")

        # 最终混合检索：以 query_embedding 为主向量路径，关键词辅助
        search_results = self.db.search_hybrid(
            query,
            top_k=top_k,
            weighted_keywords=weighted_keywords if weighted_keywords else None,
            query_embedding=query_embedding,   # 主路径：query 语义向量
        )

        # 若 HyDE 带来了额外候选但 search_hybrid 未收录，手动补充到末尾
        if hyde_embedding is not None and query_embedding is not None:
            existing_ids = {doc.id for doc, _ in search_results}
            for doc, score in pre_sorted:
                if len(search_results) >= top_k:
                    break
                if doc.id not in existing_ids:
                    search_results.append((doc, score * 0.9))  # HyDE补充结果轻微降权
                    existing_ids.add(doc.id)

        return search_results

    # ================================================================
    #  阶段 3：生成回答
    # ================================================================

    def chat(self, query: str, top_k: int = 5,
             history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        """
        完整 RAG 流程: 查询增强 → 检索 → 生成
        """
        # 1. 查询增强 (HyDE + 关键词)
        hypo_doc, weighted_keywords = self._enhance_query(query)

        if weighted_keywords:
            print(f"🧠 关键词: {[(kw, f'{wt:.1f}') for kw, wt in weighted_keywords]}")
        if hypo_doc != query:
            print(f"📝 HyDE 假设文档: {hypo_doc[:100]}...")

        # 2. 检索
        search_results = self._retrieve(query, hypo_doc, weighted_keywords, top_k)

        if not search_results:
            return "抱歉，知识库中没有找到相关信息。", []

        # 3. 构建上下文
        context_str = ""
        references = []

        for i, (doc, score) in enumerate(search_results):
            ref_info = {
                "id": i + 1,
                "source": doc.source,
                "score": score,
                "content": doc.content
            }
            references.append(ref_info)
            context_str += f"\n【资料 {i+1}】(来源: {doc.source})\n{doc.content}\n" + "-" * 30

        # 4. 构建最终 Prompt
        user_prompt = f"""请根据以下参考资料回答问题：

问题：{query}

---
{context_str}
---

请回答："""

        # 5. 调用 LLM 生成
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            # 限制历史长度，且截断过长的单条消息
            trimmed_history = []
            for msg in history[-4:]:
                trimmed = msg.copy()
                if len(trimmed.get("content", "")) > 2000:
                    trimmed["content"] = trimmed["content"][:2000] + "..."
                trimmed_history.append(trimmed)
            messages.extend(trimmed_history)

        messages.append({"role": "user", "content": user_prompt})

        print("  正在思考...")
        response = self.llm.chat(messages, temperature=0.3)

        return response, references

    def close(self):
        if self.db:
            self.db.close()