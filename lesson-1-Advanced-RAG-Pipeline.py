# Credit:
# Much code from this section came from these sources, check them out too
# https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/3/rag-triad-of-metrics
# https://www.thequalityduck.co.uk/testing-ai-how-to-assess-llm-quality-with-trulens/

from llama_index.llms.ollama import Ollama

### Configs ###
# Load model configurations from .env file
# Nice to keep configs in one place to ensure model stays same across files. 
# Changing model takes a long time for first load
from dotenv import load_dotenv
import os
import logging
import sys

# Use load dot env to load the .env file and overwrite the os.environ variables
load_dotenv()

# Configure logging for detailed output. Adjust or disable for less verbosity.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

model=os.getenv("MODEL")
base_url=os.getenv("BASE_OLLAMA_URL")
context_window_size=os.getenv("CONTEXT_WINDOW_SIZE")
################

print(f"Model: {model}")
print(f"base_url: {base_url}")
print(f"context_window_size: {context_window_size}")

####################################################################################
# Initialize Ollama Model
####################################################################################
# Setup Ollama model with specific configurations for use in chat interactions.
print("Configuring Ollama...")
from llama_index.core import Settings
Settings.llm = Ollama(
    model=model,
    base_url=base_url,
    request_timeout=120  # Adjust timeout for model responsiveness.
)
Settings.context_window = int(context_window_size)

# Set embeddings model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
print("Configuring Local Embeddings Models...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

####################################################################################
### Prepare Data ###
####################################################################################

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
print("Loading data...")

documents = SimpleDirectoryReader(
    input_files=["./data/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

print(f"Loaded {len(documents)} documents")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

####################################################################################
# Query the Model with naive rag
####################################################################################


# print("\n#######\nQuerying the model...\n")
# query = "What are steps to take when finding projects to build your experience?"
# print(f"Query: {query}")
# response = query_engine.query(query)
# print(f"Response: {response}")
# print("\n#######\n")


####################################################################################
### Sentence Window ###
####################################################################################

from llama_index.core import Document

document = Document(text="\n\n".\
                    join([doc.text for doc in documents]))




####################################################################################
# Evals using Trulens using ollama as the LLM provider #
####################################################################################

from trulens_eval import Tru
from trulens_eval import (
    Feedback,
    TruLlama
)
from trulens_eval.feedback import GroundTruthAgreement
from trulens_eval.feedback.provider.litellm import LiteLLM
import numpy as np
import pandas as pd

print("Evaluating the model...\n\n#######\n")
tru = Tru()
tru.reset_database() # This is needed in order to resolve some bug with setting the right ollama url

# NOTE: Cannot get the LiteLLM to work with the ollama model, simply doesn't give any feedback
from trulens_eval.feedback.provider.endpoint import LiteLLMEndpoint
# provider = LiteLLM()
# provider = LiteLLM(
#     model_engine="ollama/llama3:8b-instruct-fp16", 
#     endpoint="http://localhost:11435"
# )
# endpoint = LiteLLMEndpoint(litellm_provider="ollama")
# provider = LiteLLM(
#     # model_engine=f"ollama/{model}", llama3:8b-instruct-fp16
#     # model_engine=f"ollama/llama3:8b-instruct-fp16",
#     model_engine="ollama/llama3",
#     # api_endpoint=base_url
# )
# provider = LiteLLM()

# provider.set_verbose=True

from trulens_eval import OpenAI as fOpenAI
provider = fOpenAI(
)

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

context_selection = TruLlama.select_source_nodes().node.text

f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

# grounded = GroundTruthAgreement(groundedness_provider=provider)

# f_groundedness = (
#     Feedback(grounded.groundedness_measure_with_cot_reasons,
#              name="ground_truth"
#             )
#     .on(context_selection)
#     .on_output()
#     .aggregate(grounded.grounded_statements_aggregator)
# )

golden_set = [
    {"query": "who invented the lightbulb?", "response": "Thomas Edison"},
    {"query": "¿quien invento la bombilla?", "response": "Thomas Edison"}
]

f_groundtruth = Feedback(
    GroundTruthAgreement( golden_set, provider ).agreement_measure, 
    name = "Ground Truth"
).on_input_output()

tru_recorder = TruLlama(
    query_engine,
    app_id="App_1",
        feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundtruth
    ]
)

eval_questions = []
# with open('eval_questions.txt', 'r') as file:
#     for line in file:
#         # Remove newline character and convert to integer
#         item = line.strip()
#         eval_questions.append(item)

eval_questions.append("How can I be successful in AI?")

print(eval_questions)

print("\n#######\nRunning evals")
with tru_recorder as recording:
    for question in eval_questions:
        print(f"Eval Query: {question}")
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()

print("\n#######\nFinished Running evals")

pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]

tru.get_leaderboard(app_ids=[])

tru.run_dashboard()

#### Old Stuff ###############################################################

# eval_questions = []
# with open('eval_questions.txt', 'r') as file:
#     for line in file:
#         # Remove newline character and convert to integer
#         item = line.strip()
#         print(item)
#         eval_questions.append(item)

# tru = Tru()
# tru.reset_database() # This is needed in order to resolve some bug with setting the right ollama url

# litellm_provider = LiteLLM()
# provider = LiteLLM(
#     model_engine=f"ollama/{model}", 
#     endpoint=base_url
# )

# qa_relevance = (
#     Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
#     .on_input_output()
# )

# qs_relevance = (
#     Feedback(provider.relevance_with_cot_reasons, name = "Context Relevance")
#     .on_input()
#     .on(TruLlama.select_source_nodes().node.text)
#     .aggregate(np.mean)
# )

# # ToDo: Get Groundedness working
# from trulens_eval.feedback import Groundedness
# grounded = Groundedness(groundedness_provider=provider)

# groundedness = (
#     Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
#         .on(TruLlama.select_source_nodes().node.text)
#         .on_output()
#         .aggregate(grounded.grounded_statements_aggregator)
# )

# f_groundedness = (
#     Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
#     .on(TruLlama.select_source_nodes().node.text)
#     .on_output()
#     .aggregate(grounded.grounded_statements_aggregator)
# )

# f_groundedness = (
#     Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
#     .on(Select.RecordCalls.retrieve.rets.collect())
#     .on_output()
# )

# huggingface_provider = Huggingface()
# groundedness_hug = Groundedness(groundedness_provider=huggingface_provider)
# f_groundedness_hug = Feedback(groundedness_hug.groundedness_measure, name = "Groundedness Huggingface").on_input().on_output().aggregate(groundedness_hug.grounded_statements_aggregator)
# def wrapped_groundedness_hug(input, output):
#     return np.mean(list(f_groundedness_hug(input, output)[0].values()))

# # feedbacks = [qa_relevance, qs_relevance, groundedness]
# feedbacks = [qa_relevance, qs_relevance]
# feedbacks = [qa_relevance]

# def get_prebuilt_trulens_recorder(query_engine, app_id):
#     tru_recorder = TruLlama(
#         query_engine,
#         app_id=app_id,
#         feedbacks=feedbacks
#         )
#     return tru_recorder

# tru_recorder = get_prebuilt_trulens_recorder(query_engine,
#                                              app_id="Direct Query Engine")

# with tru_recorder as recording:
#     for question in eval_questions:
#         response = query_engine.query(question)

# records, feedback = tru.get_records_and_feedback(app_ids=[])

# records.head()

# # launches on http://localhost:8501/
# tru.run_dashboard()