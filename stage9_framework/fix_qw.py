import ipdb
import langchain
import langchain_openai
from langchain_openai.chat_models.base import (
    _convert_dict_to_message as old_convert_dict_to_message,
)


# def fixed_convert_dict_to_message(_dict):
#     """Fixed version of _convert_dict_to_message that handles None content."""
#     if _dict.get("content") is None:
#         _dict["content"] = _dict.get("reasoning", "")
#     return old_convert_dict_to_message(_dict)


# langchain_openai.chat_models.base._convert_dict_to_message = (
#     fixed_convert_dict_to_message
# )
