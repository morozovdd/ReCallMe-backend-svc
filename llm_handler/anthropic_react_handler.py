import os
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


llm = Anthropic(temperature=0.0, model='claude-3-opus-20240229')
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


# Using Anthropic LLM API for LLM
os.environ['ANTHROPIC_API_KEY'] = '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'

# from IPython.display import display, HTML

from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool


Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is 20+(2*4)? Calculate step by step ")
