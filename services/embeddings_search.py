import os
import pickle
import dotenv
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import cosine_similarity, get_embedding
dotenv.load_dotenv()

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
dotenv.load_dotenv(dotenv_path=env_path)
embedding_model = "text-embedding-ada-002"
openai.api_key = os.getenv("OPENAI_API_KEY")

def init_db():
    """
    Retrieves the embeddings.pickle file and loads it into memory
    """
    file_path = os.path.join(os.getcwd(), '..', 'chat-completions', 'services', 'embeddings.pickle')
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def search(prompt, embeddings_store, n=2, threshold=0.8):
    """
    Embeds a query using the OpenAI API.
    n = number of plaintext knowledge articles to return
    Threshold sets a similarity cutoff for providing knowledge articles
    """
    prompt_embed = get_embedding(prompt, embedding_model)
    print(len(prompt_embed))
    embeddings_store['similarity'] = embeddings_store["embeddings"].apply(lambda x: cosine_similarity(x, prompt_embed))
    filtered_store = embeddings_store[embeddings_store['similarity'] > threshold]
    filtered_store = filtered_store.sort_values(by='similarity', ascending=False)
    top_n_results = filtered_store.head(n)
    print((top_n_results['plaintext'].to_list(), top_n_results['similarity'].to_list()))
    return top_n_results['plaintext'].to_list()