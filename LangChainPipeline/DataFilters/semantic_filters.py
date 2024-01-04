from evaluation.sentence_embedder import get_cosine_similarity_scores

def filter_metadata_by_semantic_similarity(targets, candidates, k, neighbors_left, neighbors_right):
    """
    Semantic similarity filter.
    :param target: target word
    :param candidates: candidate words
    :param k: number of candidates to return
    :param neighbors_left: number of neighbors to the left
    :param neighbors_right: number of neighbors to the right
    :return: top k candidates
    """
    
    if len(candidates) == 0 or k == 0:
        return []
    
    candidate_texts = [candidate["data"] for candidate in candidates]
    cosine_scores, top_10_pairs = get_cosine_similarity_scores(candidate_texts, targets)

    # remove transcript segments that are in the skipped segments
    for i, candidate in enumerate(candidates):
        candidate["score"] = max(cosine_scores[i])
        candidate["index"] = i

    candidates.sort(key=lambda x: x["score"], reverse=True)
    keep_indexes = [candidate["index"] for candidate in candidates[0:int(k)]]

    candidates.sort(key=lambda x: x["index"])

    all_keep_indexes = []

    for index in keep_indexes:
        all_keep_indexes.extend(range(int(index - neighbors_left), int(index + neighbors_right + 1)))
    
    all_keep_indexes = set(all_keep_indexes)

    return [candidate for i, candidate in enumerate(candidates) if i in all_keep_indexes]