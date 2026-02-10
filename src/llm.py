import time
from typing import Optional
from llama_cpp import Llama
from openai import OpenAI
from loguru import logger

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, 
                model: str = "bartowski/Llama-3.2-3B-Instruct-GGUF", model_path: str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf", 
                n_ctx: int = 15000, n_threads: int = 4, verbose: bool = False,
                lang:str = "English"):
        self.model = model
        self.lang = lang
        
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id=model,
                filename=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=verbose,
            )

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(3)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]


def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, model_path: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, model_path=model_path, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM


def destroy_global_llm() -> None:
    """
    销毁全局LLM实例，释放内存空间
    """
    global GLOBAL_LLM
    if GLOBAL_LLM is not None:
        logger.info(f"Destroying LLM instance: {GLOBAL_LLM.model}")
        # 释放Llama模型资源
        if hasattr(GLOBAL_LLM, 'llm'):
            if isinstance(GLOBAL_LLM.llm, Llama):
                # 对于Llama本地模型，尝试调用释放资源的方法
                try:
                    # llama_cpp的Llama实例没有显式的close方法，但可以通过删除实例来释放资源
                    del GLOBAL_LLM.llm
                    logger.info("Llama model resources released")
                except Exception as e:
                    logger.error(f"Failed to release Llama model resources: {e}")
            # 对于OpenAI API客户端，不需要显式释放资源
        
        # 删除全局LLM实例
        del GLOBAL_LLM
        GLOBAL_LLM = None
        logger.info("Global LLM instance destroyed")
    else:
        logger.info("No global LLM instance to destroy")

