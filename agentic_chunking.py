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
Decompose the following content into clear and simple propositions, ensuring they are interpretable out of
context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. You can remove some information if it is not necessary to understand the meaning of the context.
5. Present the results as a list of strings, formatted in JSON.

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
    # return propositions
