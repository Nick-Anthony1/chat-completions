import pickle
import os
import dotenv
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
dotenv.load_dotenv(dotenv_path=env_path)
key = os.getenv("OPENAI_API_KEY")
embedding_model = "text-embedding-ada-002"
openai.api_key = key

def embed_query(prompt):
    """
    Embeds a single column csv using the OpenAI embeddings API.
    """
    return get_embedding(prompt, embedding_model)

def create_knowledge_base(csv_file, chunk_input=False, header_name=None):
    """
    Embeds a single column csv using the OpenAI embeddings API.
    Using chunking for slightly better memory optimisation if required
    """
    if chunk_input:
        chunks = []
        for chunk in pd.read_csv(csv_file, header=header_name, usecols=[0], chunksize=1000):
            chunks.append(chunk)
        df = pd.concat(chunks, axis=0)

    else:
        df = pd.read_csv(csv_file, header=header_name, usecols=[0])

    df.columns = ['plaintext']
    df.dropna(inplace=True)
    df["embeddings"] = df["plaintext"].apply(embed_query)
    with open('embeddings.pickle', 'wb') as f:
        pickle.dump(df, f)

create_knowledge_base("subs.csv")