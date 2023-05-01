from openai.embeddings_utils import get_embedding, cosine_similarity
from openai import api_key
import dotenv
import os
from embeddings_store import embed_query

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("OPENAI_EMBEDDINGS_MODEL")

def embeddings_search(prompt, n=1 ):
    """
    Embeds a query using the OpenAI API.
    n = number of plaintext knowledge articles to return
    """
    prompt_embed = embed_query(prompt)
    
    return None

