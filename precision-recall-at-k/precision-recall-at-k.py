def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    # 1. Consider only the first k items
    top_k_recommendations = set(recommended[:k])
    relevant_set = set(relevant)
    
    # 2. Count the overlapping hits
    hits = len(top_k_recommendations.intersection(relevant_set))
    
    # 3. Calculate metrics
    precision_at_k = hits / k
    recall_at_k = hits / len(relevant)
    
    return [precision_at_k, recall_at_k]
