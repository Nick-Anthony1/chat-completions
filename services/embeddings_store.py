from openai import api_key
import dotenv
import pandas as pd
from openai.embeddings_utils import get_embedding
import os

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("OPENAI_EMBEDDINGS_MODEL")

def embed_query(prompt):
    """
    Embeds a single column csv using the OpenAI embeddings API.
    """
    return get_embedding(prompt, embedding_model)

def embed_knowledge_base(csv_file, output_filename="embeddings.csv", chunk_input=False, header_name=None):
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
    df.to_csv(output_filename, index=False)


    
embed_knowledge_base("subs.csv")