# soda/openai/__init__.py
from .text import completion_model, instruct_chat_model, completion_model_batched

__all__ = [
    'completion_model',
    'instruct_chat_model',
    'completion_model_batched'
]