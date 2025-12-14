import math
from typing import List, Dict, Any


class RetrievalEvaluator:
    """Simple evaluator for measuring retrieval quality."""

    def precision_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate Precision@K: fraction of top K results that are relevant.

        Args:
            retrieved_ids: List of retrieved chunk/document IDs
            relevant_ids: List of relevant IDs (ground truth)
            k: The number of results to consider

        Returns:
            Precision@K: fraction of top K results that are relevant
        """

        if k == 0 or len(retrieved_ids) == 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_found = sum(1 for id in top_k if id in relevant_ids)
        return relevant_found / k

    def recall_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate Recall@K: fraction of relevant items found in top K.

        Args:
            retrieved_ids: List of retrieved chunk/document IDs
            relevant_ids: List of relevant IDs (ground truth)
            k: The number of results to consider

        Returns:
            Recall@K: fraction of relevant items found in top K
        """

        if len(relevant_ids) == 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_found = sum(1 for id in top_k if id in relevant_ids)
        total_relevant = len(relevant_ids)
        return relevant_found / total_relevant

    def mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank: 1/rank of first relevant result.

        Args:
            retrieved_ids: List of retrieved chunk/document IDs
            relevant_ids: List of relevant IDs (ground truth)

        Returns:
            Mean Reciprocal Rank: 1/rank of first relevant result (0.0 if no matches)
        """
        
        if len(retrieved_ids) == 0:
            return 0.0
        
        # Convert to sets for faster lookup (and handle nested lists)
        # Flatten retrieved_ids if needed
        flat_retrieved = []
        for id in retrieved_ids:
            if isinstance(id, list):
                flat_retrieved.extend(id)
            else:
                flat_retrieved.append(id)
        
        relevant_set = set(relevant_ids)
        
        # Find first relevant result
        for rank, id in enumerate(flat_retrieved, start=1):
            if id in relevant_set:
                return 1.0 / rank
        
        # No relevant results found
        return 0.0

    def ndcg_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate NDCG@K: normalized discounted cumulative gain.

        Args:
            retrieved_ids: List of retrieved chunk/document IDs
            relevant_ids: List of relevant IDs (ground truth)
            k: The number of results to consider

        Returns:
            NDCG@K: normalized discounted cumulative gain
        """

        if k == 0 or len(retrieved_ids) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, id in enumerate(retrieved_ids[:k], start=1):
            if id in relevant_ids:
                dcg += 1.0 / math.log2(i + 1)
        
        # Calculate ideal DCG
        ideal_scores = [1.0] * min(k, len(relevant_ids))
        idcg = sum(score / math.log2(i + 1) for i, score in enumerate(ideal_scores, start=1))
        
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self, retrieved_ids: List[str], relevant_ids: List[str], 
                k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        Evaluate retrieval with multiple metrics.
        
        Args:
            retrieved_ids: List of retrieved chunk/document IDs
            relevant_ids: List of relevant IDs (ground truth)
            k_values: List of k values to evaluate at
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "precision": {},
            "recall": {},
            "mrr": self.mrr(retrieved_ids, relevant_ids),
            "ndcg": {}
        }
        
        for k in k_values:
            results["precision"][f"P@{k}"] = self.precision_at_k(retrieved_ids, relevant_ids, k)
            results["recall"][f"R@{k}"] = self.recall_at_k(retrieved_ids, relevant_ids, k)
            results["ndcg"][f"NDCG@{k}"] = self.ndcg_at_k(retrieved_ids, relevant_ids, k)
        
        return results


