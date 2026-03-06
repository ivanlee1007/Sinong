"""
客户端模块 - LLM和Embedding

LLM客户端:
- 支持阿里云百炼(DashScope)
- 支持OpenAI兼容接口

Embedding客户端:
- 支持本地sentence-transformers模型
- 支持自定义模型路径
"""

import os
import numpy as np
from typing import List, Dict, Union
from openai import OpenAI
import torch

class LLMClient:
    """
    LLM客户端
    
    支持多种模型配置，用于：
    - 语义完整性判断（快速模型）
    - 仲裁重新切分（高质量模型）
    """
    
    # 预设模型
    MODELS = {
        "qwen-flash": {
            "base_url": "",
            "model": "qwen-flash",
            "description": "极速、超低成本"
        },
        # 阿里云百炼 - 高质量
        "qwen-plus": {
            "base_url": "",
            "model": "qwen-plus",
            "description": "平衡性能"
        },
        "qwen-max": {
            "base_url": "",
            "model": "qwen-max",
            "description": "最强性能"
        },
        # DeepSeek
        "deepseek-v3.2-exp": {
            "base_url": "",
            "model": "deepseek-v3.2-exp",
            "description": "DeepSeek对话"
        },
        "qwen3-next":{
            "base_url": "",
            "model": "",
            "description": "开源优越"
        },
        "qwen3-32b":{
            "base_url": "",
            "model": "",
            "description": "开源高效"
        }
    }
    
    def __init__(self,
                 model: str = "qwen3-next",
                 api_key: str = None,
                 base_url: str = None):
        """
        初始化
        
        Args:
            model: 模型名称（预设名或自定义模型ID）
            api_key: API密钥（默认从环境变量读取）
            base_url: API地址
        """
        # 解析预设
        if model in self.MODELS:
            preset = self.MODELS[model]
            self.model = preset["model"]
            self.base_url = base_url or preset["base_url"]
            env_key = preset.get("env_key", "DASHSCOPE_API_KEY")
            self.api_key = api_key or os.getenv(env_key, "")
            self.description = preset["description"]
        else:
            self.model = model
            self.base_url = base_url or ""
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
            self.description = "自定义模型"
        
        if not self.api_key:
            print(f"警告: 未设置API密钥，LLM功能将不可用")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def chat(self, messages: List[Dict], 
             max_tokens: int = 4096,
             temperature: float = 0.7) -> str:
        """发送聊天请求"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化（缺少API密钥）")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def complete(self, prompt: str, **kwargs) -> str:
        """简单补全"""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
    
    def __repr__(self):
        return f"LLMClient(model={self.model}, {self.description})"


class EmbeddingClient:
    """
    Embedding客户端 (优化版)
    
    特性:
    - 支持 FP16 半精度加速
    - 支持 Instruction (指令前缀)
    - 自动 GPU/CPU 检测
    """
    
    MODELS = {
        "Qwen3-Embeding": {
            "name": "", # 请确保路径正确
            "dimension": 4096,
            "description": "Qwen3-Embedding-8B 双语嵌入",
            "query_instruction": "Represent this query for retrieving relevant documents: ",
            "doc_instruction": ""
        },
        "bge-m3": {
            "name": "",
            "dimension": 1024,
            "description": "BGE-M3 多语言（100+语言）",
            "query_instruction": "", # BGE-M3 通常不需要特定指令，或根据具体任务设定
            "doc_instruction": ""
        },
        "bge-large-zh-v1.5": {
            "name": "",
            "dimension": 1024,
            "description": "BGE 中文专用",
            "query_instruction": "为这个句子生成表示以用于检索相关文章：",
            "doc_instruction": ""
        }
    }
    
    def __init__(self, model: str = "Qwen3-Embeding", use_fp16: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and (self.device == "cuda")
        
        if model in self.MODELS:
            info = self.MODELS[model]
            self.model_name = info["name"]
            self._dimension = info["dimension"]
            self.description = info["description"]
            # 保存默认指令
            self.default_query_instruction = info.get("query_instruction", "")
            self.default_doc_instruction = info.get("doc_instruction", "")
        else:
            self.model_name = model
            self._dimension = None
            self.description = "自定义模型"
            self.default_query_instruction = ""
            self.default_doc_instruction = ""
        
        print(f"Loading Embedding Model: {self.model_name} on {self.device} (FP16={self.use_fp16})...")
        
        try:
            from sentence_transformers import SentenceTransformer
            # 加载模型
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)
            
            # 开启半精度 (FP16)
            if self.use_fp16:
                self.model.half()
                
            if self._dimension is None:
                self._dimension = self.model.get_sentence_embedding_dimension()
                
        except Exception as e:
            raise RuntimeError(f"Embedding模型加载失败: {e}")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, texts: Union[str, List[str]], 
              batch_size: int = None,
              show_progress: bool = True,
              normalize: bool = True,
              max_length: int = 20480,
              instruction: str = "") -> np.ndarray:
        """
        生成文本向量
        
        Args:
            instruction: 文本前的指令前缀（例如 "Represent this query: "）
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 1. 拼接指令 (如果提供了 instruction)
        if instruction:
            texts = [f"{instruction}{t}" for t in texts]
        
        # 2. 截断 (防止 Token 溢出)
        # 注意：这里是字符级粗略截断，模型内部还有 Token 级截断
        texts = [t[:max_length] if len(t) > max_length else t for t in texts]
        
        # 3. 智能 Batch Size 选择
        if batch_size is None:
            if self.device == "cpu":
                batch_size = 4
            elif self._dimension >= 4096: # 超大模型
                batch_size = 4 if not self.use_fp16 else 16
            elif self._dimension >= 1024: # BGE-M3 等
                batch_size = 16 if not self.use_fp16 else 64
            else:
                batch_size = 64
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress and len(texts) > 10,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return np.array(embeddings, dtype=np.float32)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠️ GPU显存不足，自动降级为逐条处理...")
                torch.cuda.empty_cache()
                embeddings = self.model.encode(
                    texts,
                    batch_size=1,
                    show_progress_bar=show_progress,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True
                )
                return np.array(embeddings, dtype=np.float32)
            raise
    
    def embed_query(self, text: str) -> np.ndarray:
        """专门用于查询的 Embedding (自动带指令)"""
        return self.embed([text], show_progress=False, instruction=self.default_query_instruction)[0]
    
    def embed_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """专门用于文档入库的 Embedding"""
        return self.embed(texts, instruction=self.default_doc_instruction, **kwargs)

    def similarity(self, text1: str, text2: str) -> float:
        """计算相似度"""
        # 相似度计算通常不需要 instruction，或者两边都用 doc instruction
        emb1 = self.embed([text1], show_progress=False)[0]
        emb2 = self.embed([text2], show_progress=False)[0]
        return float(np.dot(emb1, emb2))


# 便捷函数
def get_llm(model: str = "qwen3-next") -> LLMClient:
    """获取LLM客户端"""
    return LLMClient(model)


def get_embedder(model: str = "bge-m3") -> EmbeddingClient:
    """获取Embedding客户端"""
    return EmbeddingClient(model)