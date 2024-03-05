from langchain_community.chat_models import ChatOpenAI
from typing import List
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser
import openai
from dotenv import load_dotenv, find_dotenv
import os
import sys
from langchain.prompts import ChatPromptTemplate

sys.path.append("../..")


_ = load_dotenv(find_dotenv())  # read local .env file

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

openai.api_key = os.environ["OPENAI_API_KEY"]
MODEL = os.environ["OPENAI_MODEL"]


class Sentences(BaseModel):
    sentences: List[str]


def get_propositions(text):
    prompt = ChatPromptTemplate.from_template(
        """
Here are texts I crawled from a company webpage, please break it into some chunks of relevant information about company such as About Us, Mission,  Services and Products, Address, Contact, Careers, FAQ, and more.
Ensuring they are interpretable out of context. Here are some more guidelines to help you:
1. You can modify the text as you see fit, comprehensively and accurately.
2. Present the results as a list of strings, formatted in JSON, example:
["About Us: <Company's name> is a financial company that does X and Y.", "Mission: <Company's name> mission is to do Z.", ...]

Content:
{input}
"""
    )
    llm = ChatOpenAI(
        model_name=MODEL,
        temperature=0,
    )
    chain = prompt | llm | JsonOutputParser(pydantic_object=Sentences)
    return chain.invoke({"input": text})
