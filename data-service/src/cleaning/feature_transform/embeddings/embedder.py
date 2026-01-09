from typing import List, Union
import numpy as np
from src.config.config_manager import ConfigurationManager
from sentence_transformers import SentenceTransformer




config = ConfigurationManager()
model_congig = config.get_model_training_config()
MODEL_NAME = model_congig.embed_model_names if model_congig.embed_model_names else "all-MiniLM-L6-v2"

embed_model = SentenceTransformer(MODEL_NAME)


def get_text_embedding(
    text: Union[str, List[str]]
) -> Union[List[float], List[List[float]]]:
    

  
    if isinstance(text, str):
        if not text:
            return [0.0] * embed_model.get_sentence_embedding_dimension()
        emb = embed_model.encode(text, convert_to_numpy=True)
        return emb.tolist()

   
    elif isinstance(text, list):
        if not text:
            return []

        processed_texts = [t if t else " " for t in text]
        embs = embed_model.encode(processed_texts, convert_to_numpy=True)
        return embs.tolist()

    else:
        raise TypeError("Input must be a string or list of strings")


def get_embedding_dim() -> int:
    """
    Returns the embedding dimension of the current model.
    """
    return embed_model.get_sentence_embedding_dimension()
