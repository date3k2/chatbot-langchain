from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())  # read local .env file
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
gemma_id = "google/gemma-2b-it"
# hf = HuggingFacePipeline.from_model_id(
#     model_id=gemma_id,
#     task="text-generation",
#     max_new_tokens=512,
#     top_k=10,
#     top_p=0.95,
#     typical_p=0.95,
#     temperature=0.01,
#     repetition_penalty=1.03,
#     callbacks=[StreamingStdOutCallbackHandler()],
#     streaming=True,
# )
gemma = HuggingFaceEndpoint(
    repo_id=gemma_id,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
)
template = """<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""

# We create a prompt from the template so we can use it with Langchain
prompt_template = PromptTemplate(template=template, input_variables=["question"])
question = "What is the capital of France?"
end = "<end_of_turn>\n"
prompt = """<start_of_turn>system
You are very helpful assistant that can answer user question with some short sentences with creativity,
if you don't know the answer, just say you don't know.<end_of_turn>\n"""
while True:
    question = input("User: ")
    if question == "exit":
        break
    prompt += prompt_template.format(question=question)
    print("Assistant: ", end="")
    prompt += gemma.invoke(prompt)[:-5] + end
    print()
print(prompt)
