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
load_dotenv(override=True)

# Configure logging for detailed output. Adjust or disable for less verbosity.
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
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
naive_query_engine = index.as_query_engine()

####################################################################################
# Query the Model with naive rag
####################################################################################


print("\n#######\nQuerying the model to get it loaded before evals...\n")
# Time it takes to load and send 1 querys
import time
start_time = time.time()
query = "What are steps to take when finding projects to build your experience?"
print(f"Query: {query}")
response = naive_query_engine.query(query)
print(f"Response: {response}")
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
print("\n#######\n")


######################
### TrueLens Evals ###
######################
from trulens_eval import Tru
tru = Tru()
# tru.reset_database() # This is needed in order to resolve some bug with setting the right ollama url


from trulens_eval.feedback.provider import OpenAI
from trulens_eval import Feedback
import numpy as np

# Initialize provider class
provider = OpenAI()

# Use ollama to run evals with locally hosted llama3 model
# NOTE: This doesn't work but cannot figure out why
# from trulens_eval.feedback.provider import LiteLLM
# provider = LiteLLM(
#     model_engine=f"ollama/llama3", 
#     endpoint="http://localhost:11435"
# )

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App
context = App.select_context(naive_query_engine)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons)
    .on(context.collect()) # collect context chunks into a list
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance)
    .on_input_output()
)
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

def execute_eval_run(run_name, number_of_runs=1):
    print("Starting eval run: ", run_name)

    from trulens_eval import TruLlama
    tru_query_engine_recorder = TruLlama(naive_query_engine,
        app_id=run_name,
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance])

    # or as context manager
    with tru_query_engine_recorder as recording:
        for question in eval_questions:
            # Run number_of_runs
            for i in range(number_of_runs):
                print (f"Run {i+1} for question: {question}")
                naive_query_engine.query(question) 

execute_eval_run(f"{model}_naive_run_ollama_evalTha", 1)

records, feedback = tru.get_records_and_feedback(app_ids=[])

records.head()

tru.run_dashboard() # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed