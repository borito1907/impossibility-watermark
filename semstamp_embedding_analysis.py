import logging
import hydra
from watermarker_factory import get_watermarker
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
import pandas as pd
from omegaconf import OmegaConf
import re
from typing import *
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from watermarkers.SemStamp.sbert_lsh_model import SBERTLSHModel
import torch
import numpy as np

log = logging.getLogger(__name__)

def similarity(text1, text2, lsh_model):
    lsh_seed = lsh_model.get_hash([text1])[0]
    embedding1 = lsh_model.get_embeddings([text1])[0]

    log.info(f"Embedding 1: {embedding1}")

    log.info(f"LSH Seed of Text 1: {lsh_seed}")

    lsh_seed = lsh_model.get_hash([text2])[0]
    
    log.info(f"LSH Seed of Text 2: {lsh_seed}")

    embedding2 = lsh_model.get_embeddings([text2])[0]

    log.info(f"Embedding 2: {embedding2}")

    cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    log.info(f"Cosine Similarity: {cosine_similarity}")

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    import time
    import textwrap

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Using the generic SentenceTransformer...")

    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v1")
    embedder.eval()

    lsh_model = SBERTLSHModel(lsh_model_path=None,
                                  device=device, batch_size=1, lsh_dim=3, sbert_type='base', embedder=embedder)
    
    text1 = "Doesn't the ergonomic design fit like a glove?"
    text2 = "Won't the ergonomic design fit like a glove?"

    similarity(text1, text2, lsh_model)

    text1 = "(Sweating slightly) Just look at this beauty!"
    text2 = "(Laughs slightly) Just look at this beauty!"

    similarity(text1, text2, lsh_model)

if __name__ == "__main__":
    test()
