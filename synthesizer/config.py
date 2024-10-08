import os  # noqa: F401

import cohere  # noqa: F401
from dotenv import load_dotenv

load_dotenv("../.env", override=True)

# SELECT WHICH EMBEDDING MODEL AND GENERATIVE MODELL SHALL BE USED

## COHERE
#llm_model = "cohere"
#embedding_model = "embed-english-light-v3.0"
#COHERE_API_KEY = os.getenv("COHERE_API_KEY")
#COHERE_API_KEY = ""
#co = cohere.Client(COHERE_API_KEY)

## LAPTOP OLLAMA
llm_model = "ollama"
embedding_model = "snowflake-arctic-embed"
generation_model = "llama3.1:latest"

co = ""

# SELECT WHAT KIND OF DATAINPUT; JSON for Discovery v2 Input; PDF for PDF Files AND IF THERE ARE MANY DOCUMENTS HOW MANY
#input_type = "JSON" # the JSON input is specified according to what you get from Discovery V2
input_type = "PDF"
number_of_texts = 5 # random selection of XY texts = documents

# CONFIGURE TEXTCHUNK SIZE AND OVERLAP IN TOKEN (mind the max context window!)
CHUNK_SIZE=512 # Parametrize in according to the embedding model used and what you want to test.
CHUNK_OVERLAP=0 # Parametrize in according to what you want to test. For the test data creation though it might be better not to overlap the splits.

NUM_EVO = 1 # Minimum is 1 evolution step and maximum is 3 evolution steps. The more evolution steps the longer it takes to ceate the synthetic test data set.
