import warnings
from abc import ABC
from typing import Any, Dict, Optional, Tuple

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import Field

from langchain.memory.utils import get_prompt_input_key

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.pydantic_v1 import root_validator

from langchain.memory.chat_memory import BaseChatMemory, BaseMemory
from langchain.memory.utils import get_prompt_input_key

class LimitedHistoryMemory(BaseChatMemory):
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    max_history_length: int = 10
    log_file: str = "conversation_log.txt"

    def __init__(self, **data):
        super().__init__(**data)
        with open(self.log_file, 'a') as f:
            pass

    @property
    def buffer(self) -> Any:
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    async def abuffer(self) -> Any:
        return await self.abuffer_as_messages() if self.return_messages else await self.abuffer_as_str()

    def _buffer_as_str(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            f"{self.human_prefix}: {msg.content}" if isinstance(msg, HumanMessage) else f"{self.ai_prefix}: {msg.content}"
            for msg in messages
        )

    @property
    def buffer_as_str(self) -> str:
        return self._buffer_as_str(self.chat_memory.messages[-self.max_history_length:])

    async def abuffer_as_str(self) -> str:
        messages = await self.chat_memory.aget_messages()
        return self._buffer_as_str(messages[-self.max_history_length:])

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        return self.chat_memory.messages[-self.max_history_length:]

    async def abuffer_as_messages(self) -> List[BaseMessage]:
        messages = await self.chat_memory.aget_messages()
        return messages[-self.max_history_length:]

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {self.memory_key: self.buffer}

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        buffer = await self.abuffer()
        return {self.memory_key: buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        super().save_context(inputs, outputs)
        self._truncate_memory()
        self._log_conversation(inputs, outputs)

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        await super().asave_context(inputs, outputs)
        self._truncate_memory()
        self._log_conversation(inputs, outputs)

    def _truncate_memory(self) -> None:
        if len(self.chat_memory.messages) > self.max_history_length:
            self.chat_memory.messages = self.chat_memory.messages[-self.max_history_length:]

    def _log_conversation(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        with open(self.log_file, 'a') as f:
            f.write(f"{self.human_prefix}: {input_str}\n")
            f.write(f"{self.ai_prefix}: {output_str}\n")


