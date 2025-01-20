import os
from dotenv import load_dotenv
from core import ChatOpenAI as OriginalChatOpenAI
from typing import Dict, List, Any
from pydantic import SecretStr

# 加载 .env 文件中的变量
load_dotenv()

class LLM(OriginalChatOpenAI):
    def __init__(self, *args, **kwargs):
        # 从环境变量中加载配置
        base = SecretStr(os.getenv("API_BASE"))
        key = SecretStr(os.getenv("API_KEY"))
        mod = SecretStr(os.getenv("API_MODEL"))
        
        kwargs["base"] = base.get_secret_value()
        kwargs["key"] = key.get_secret_value()
        kwargs["model"] = mod.get_secret_value()
        super().__init__(*args, **kwargs)

    @property
    def lc_secrets(self) -> Dict[str, SecretStr]:
        return {"key": self.key}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "chat_models", "openai"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.base:
            attributes["base"] = self.base
        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True
