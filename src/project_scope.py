from typing import Dict, Any, List, Optional

class ProjectScope:
    """Manages project scope for document domains."""

    def __init__(self):
        """Initialize with project scope configuration."""
        # Domain keywords mapping
        self.domains = {
            "rag": ["rag", "retrieval augmented", "retrieval", "augmented generation", "llm", "large language model"],
            "document_ai": ["document ai", "document processing", "arctic-tilt", "arctic tilt", "zero-shot", "zero shot", "fine-tuning", "fine-tune"],
            "agentic_ai": ["agentic ai", "agentic", "proactive", "adaptable", "collaborative", "autonomous", "ai agents", "specialized"],
            "chunking": ["chunking", "semantic chunking", "text chunking", "fixed-sized chunking", "fixed-sized", "fixed size", "nlp"],
            "contextual_retrieval": ["contextual retrieval", "bm25", "semantic search", "keyword search", "anthropic"],
            "autopilot": ["autopilot", "uipath", "automation", "studio", "apps", "test manager", "communications mining", "process mining", "clipboard ai"]
        }
        
        # Project metadata
        self.project_name = "Building a RAG Assistant project"
        self.description = "Building a RAG Assistant project for the AAIDC course module 1"

    def identify_domain(self, query: str) -> Optional[str]:
        """
        Identify which domain a query belongs to.

        Args:
            query: The query to classify

        Returns:
            The domain name
        """
        query_lower = query.lower()
        
        for domain_name, keywords in self.domains.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain_name
        
        return None

    def validate_scope(self, query: str) -> Dict[str, Any]:
        """
        Validate if a query is within project scope.

        Args:
            query: The query to validate

        Returns:
            Dictionary with validation results
        """
        # Identify domain
        domain = self.identify_domain(query)
        is_in_scope = domain is not None
        
        return {
            "is_in_scope": is_in_scope,
            "domain": domain,
            "suggestion": "This query appears to be out of scope for this knowledge base." if not is_in_scope else None
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get the full configuration.

        Returns:
            The configuration
        """
        return {
            "project_name": self.project_name,
            "description": self.description,
            "domains": [
                {"name": name, "display_name": name.replace("_", " ").title(), "keywords": keywords}
                for name, keywords in self.domains.items()
            ]
        }

