import random
import numpy as np
import torch


def similarity_search(retrieval, collection, input_sentence, k, query_id=None):
    input_embedding = retrieval.encode(input_sentence).tolist()

    results = collection.query(
        query_embeddings=[input_embedding],
        n_results=k + 1
    )

    ids = results['ids'][0]
    metadatas = results['metadatas'][0]

    filtered = [
                   meta for id_, meta in zip(ids, metadatas)
                   if id_ != query_id
               ][:k]

    return filtered

def set_randomness(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False