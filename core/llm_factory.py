import torch
from transformers import AutoTokenizer, AutoConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from config.settings import LLM_PATH, EMBEDDING_PATH
from utils.logger import get_logger

logger = get_logger("ModelFactory")

class ModelFactory:
    @staticmethod
    def init_models():
        """初始化 Embedding 和 LLM 模型并注入 Settings"""
        
        # 1. Embedding
        logger.info(f"加载 Embedding 模型: {EMBEDDING_PATH}")
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_PATH)
        Settings.embed_model = embed_model

        # 2. Tokenizer & LLM
        logger.info(f"加载 LLM 模型: {LLM_PATH}")
        try:
            hf_config = AutoConfig.from_pretrained(LLM_PATH, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                LLM_PATH,
                trust_remote_code=True,
                use_fast=False,
                config=hf_config,
            )
            
            llm = HuggingFaceLLM(
                model_name=LLM_PATH,
                tokenizer=tokenizer,
                context_window=8192,
                max_new_tokens=768,
                generate_kwargs={
                    "temperature": 0.1,
                    "do_sample": True,
                    "repetition_penalty": 1.1,
                    # Qwen 特殊停止符
                    "eos_token_id": [151645, 151643, tokenizer.eos_token_id],
                },
                # Qwen 专用 prompt 转换
                messages_to_prompt=ModelFactory._qwen_messages_to_prompt,
                completion_to_prompt=ModelFactory._qwen_completion_to_prompt,
                device_map="auto",
            )
            Settings.llm = llm
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

    @staticmethod
    def _qwen_completion_to_prompt(completion):
        return f"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n{completion}\n<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def _qwen_messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            prompt += f"<|im_start|>{message.role}\n{message.content}\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt