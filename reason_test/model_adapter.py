"""
统一的模型接入层
支持多种模型类型：
- DeepSeek API
- Gemini-2.5-flash API
- bbl-lite (本地模型或 API)
- vLLM 本地服务（保持兼容）
"""
import os
import json
import itertools
import threading
from typing import Optional, List, Dict, Any
from openai import OpenAI
from verl.utils import hf_tokenizer

# 禁用代理（只在本模块有效）
for key in ["http_proxy", "https_proxy", "all_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(key, None)

root_path = '/root/autodl-tmp'

# 加载 API keys
def load_api_keys():
    """从 ragen/env/api-keys.json 加载 API keys"""
    api_keys_path = 'ragen/env/api-keys.json'
    if os.path.exists(api_keys_path):
        with open(api_keys_path, 'r') as f:
            keys = json.load(f)
        return keys
    else:
        # 如果文件不存在，尝试从环境变量读取
        keys = {}
        if os.environ.get('DEEPSEEK_API_KEY'):
            keys['deepseek'] = [os.environ.get('DEEPSEEK_API_KEY')]
        if os.environ.get('GEMINI_API_KEY'):
            keys['gemini'] = [os.environ.get('GEMINI_API_KEY')]
        return keys

# 线程安全的循环迭代器（用于多 key 轮换）
class ThreadSafeCycle:
    def __init__(self, iterable):
        self._lock = threading.Lock()
        self._iterator = itertools.cycle(iterable if iterable else [None])

    def __next__(self):
        with self._lock:
            return next(self._iterator)

# 加载 API keys
api_keys = load_api_keys()
deepseek_keys = ThreadSafeCycle(api_keys.get('deepseek', []))
gemini_keys = ThreadSafeCycle(api_keys.get('gemini', []))


class ModelAdapter:
    """统一的模型适配器，支持多种模型类型"""
    
    def __init__(self, model_type: str, model_name: str = None, 
                 model_path: str = None, port: str = "2100",
                 tokenizer_path: str = None):
        """
        初始化模型适配器
        
        Args:
            model_type: 模型类型，可选值：
                - 'deepseek': DeepSeek API
                - 'gemini': Gemini-2.5-flash API
                - 'bbl-lite': bbl-lite 模型（本地或 API）
                - 'vllm': vLLM 本地服务
            model_name: 模型名称（用于 vLLM 和 bbl-lite）
            model_path: 模型路径（相对于 root_path，用于 vLLM 和 bbl-lite）
            port: vLLM 服务端口（仅用于 vLLM 类型）
            tokenizer_path: tokenizer 路径（可选，用于格式化 prompt）
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model_path = model_path
        self.port = port
        
        # 初始化 tokenizer（如果需要）
        self.tokenizer = None
        if tokenizer_path:
            try:
                self.tokenizer = hf_tokenizer(tokenizer_path)
            except Exception as e:
                print(f"⚠️  tokenizer 初始化失败: {e}，将不使用 chat template")
        elif model_type in ['vllm', 'bbl-lite'] and model_path and model_name:
            # 尝试自动初始化 tokenizer
            try:
                tokenizer_path = f"{root_path}/{model_path}/{model_name}"
                if os.path.exists(tokenizer_path):
                    self.tokenizer = hf_tokenizer(tokenizer_path)
            except Exception as e:
                print(f"⚠️  自动初始化 tokenizer 失败: {e}")
        
        # 初始化客户端
        self.client = None
        if model_type == 'deepseek':
            api_key = next(deepseek_keys)
            if not api_key:
                raise ValueError("DeepSeek API key 未配置，请在 ragen/env/api-keys.json 中配置或设置 DEEPSEEK_API_KEY 环境变量")
            self.client = OpenAI(
                api_key=api_key,
                base_url='https://api.deepseek.com'
            )
        elif model_type == 'gemini':
            api_key = next(gemini_keys)
            if not api_key:
                raise ValueError("Gemini API key 未配置，请在 ragen/env/api-keys.json 中配置或设置 GEMINI_API_KEY 环境变量")
            # Gemini 通过 dmxapi.cn 代理服务调用
            self.client = OpenAI(
                api_key=api_key,
                base_url='https://www.dmxapi.cn/v1'
            )
        elif model_type == 'vllm':
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1"
            )
        elif model_type == 'bbl-lite':
            # bbl-lite 可能是本地模型，使用 vLLM 服务
            # 或者可能是 API 调用，需要根据实际情况调整
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1"
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def generate(self, prompt: str, max_tokens: int = 600, 
                 temperature: float = 0.5, use_chat_template: bool = True) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入 prompt（对于 API 模型，应该是格式化后的 prompt 字符串；
                    对于本地模型，可以是原始 prompt，会根据 use_chat_template 决定是否应用 chat template）
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            use_chat_template: 是否使用 chat template（仅对支持 tokenizer 的模型类型）
        
        Returns:
            生成的文本
        """
        if self.model_type == 'deepseek':
            # DeepSeek API 调用使用 chat completions
            # prompt 应该是格式化后的字符串，直接作为 user message
            messages = [
                {"role": "system", "content": "You're a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name or "deepseek-chat",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"DeepSeek API 调用失败：{str(e)}")
        
        elif self.model_type == 'gemini':
            # Gemini API 调用
            # prompt 应该是格式化后的字符串，直接作为 user message
            messages = [
                {"role": "system", "content": "You're a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            try:
                # 使用 chat completions（参考 common.py 的调用方式）
                response = self.client.chat.completions.create(
                    model=self.model_name or "gemini-2.5-flash-nothinking",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                # 如果失败，尝试使用 Google AI Studio API 格式
                # 注意：这可能需要使用 google-generativeai 库
                raise RuntimeError(f"Gemini API 调用失败：{str(e)}。可能需要使用 google-generativeai 库或配置正确的代理服务")
        
        elif self.model_type in ['vllm', 'bbl-lite']:
            # vLLM 服务使用 completions
            if use_chat_template and self.tokenizer:
                message = [
                    {"role": "system", "content": "You're a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    message, add_generation_prompt=True, tokenize=False
                )
            else:
                formatted_prompt = prompt
            
            # 构建模型路径
            if self.model_path:
                model_full_path = f"{root_path}/{self.model_path}/{self.model_name}"
            else:
                model_full_path = f"{root_path}/{self.model_name}"
            
            try:
                response = self.client.completions.create(
                    model=model_full_path,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].text
            except Exception as e:
                raise RuntimeError(f"vLLM API 调用失败：{str(e)}")
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")


def create_model_adapter(model_type: str, model_name: str = None,
                        model_path: str = None, port: str = "2100") -> ModelAdapter:
    """
    便捷函数：创建模型适配器
    
    Args:
        model_type: 模型类型 ('deepseek', 'gemini', 'bbl-lite', 'vllm')
        model_name: 模型名称
        model_path: 模型路径
        port: vLLM 服务端口
    
    Returns:
        ModelAdapter 实例
    """
    # 确定 tokenizer 路径
    tokenizer_path = None
    if model_path and model_name:
        tokenizer_path = f"{root_path}/{model_path}/{model_name}"
        if not os.path.exists(tokenizer_path):
            tokenizer_path = None
    
    return ModelAdapter(
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        port=port,
        tokenizer_path=tokenizer_path
    )

