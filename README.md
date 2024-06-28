Relevant learning materials
* Intro to building with GenAI using LangChain and Open AI <https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/3/rag-triad-of-metrics>


## Setup

### 0) Setup the .env file
```bash
cp .env.example .env
```

Now fill out the .env file:
* You will need to specify the url for ollama if you already have it setup, or follow the prepare ollama step below if not. Make sure afterward that the port is right
* You will need to set the OPENAI_API_KEY since it used to run the evals

### 1) Prepare the ollama
* Follow the official docs to get setup: <https://github.com/ollama/ollama>
* Ensure you have the needed model pulled down

You can check the .env file to see what model is specified, you will need to ensure that model is pulled down. 
```
ollama pull llama3:8b-text-fp16
# ollama pull llama3
```

#### Configure your .env file as needed
* Ensure the `MODEL` defined is one you have downloaded with ollama
* Make sure the `BASE_OLLAMA_URL` specified is correct for your setup of Ollama

### 2) Install prereqs
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

## Try out the example scripts

LangChain Chain
```
python lesson-1-Advanced-RAG-Pipeline.py
```

## Want to contribute? 

Would love to accept some contributions or requests for other examples you'd like to see. I am running all this on my personal hardware and trying to come up with fun and useful examples for myself.
