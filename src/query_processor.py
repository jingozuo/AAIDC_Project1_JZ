import re
from typing import Dict, Any


class QueryProcessor:
    """Simple query processor for improving retrieval quality."""

    def preprocess(self, query: str) -> str:
        """
        Normalize and clean the query.

        Args:
            query: The query to process

        Returns:
            The processed query
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Ensure proper punctuation for questions
        if query and not query[-1] in ['.', '?', '!']:
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
            if any(q_word in query.lower() for q_word in question_words):
                query = query.rstrip('.!') + '?'
        
        return query

    def classify(self, query: str) -> str:
        """
        Classify query type based on intent.

        Args:
            query: The query to classify

        Returns:
            The query type
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'what are', 'define', 'explain', 'meaning']):
            return "definition"
        elif any(word in query_lower for word in ['how', 'process', 'steps', 'procedure']):
            return "how_to"
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
            return "comparison"
        elif any(word in query_lower for word in ['features', 'list', 'examples', 'types']):
            return "list"
        elif any(word in query_lower for word in ['when', 'where', 'who', 'which']):
            return "factual"
        else:
            return "general"

    def process(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the pipeline.

        Args:
            query: The query to process

        Returns:
            Dictionary with processed query and metadata
        """
        # Preprocess (normalize, clean)
        processed = self.preprocess(query)
        
        # Classify query type
        query_type = self.classify(processed)
        
        return {
            "original": query,
            "processed": processed,  # Use this for search
            "query_type": query_type
        }