import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from query_processor import QueryProcessor
from retrieval_evaluator import RetrievalEvaluator
from project_scope import ProjectScope

# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to the project root, then into data directory
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found at {data_dir}")
        return results

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty documents
                        results.append(content)
                        print(f"Loaded {filename} successfully")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(results)} documents successfully")
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Initialize query processor
        # Query processing is added in query_processor.py
        self.query_processor = QueryProcessor()

        # Initialize retrieval evaluator
        self.evaluator = RetrievalEvaluator()

        # Initialize project scope
        self.project_scope = ProjectScope()

        # Create RAG prompt template
        self.prompt_template = None

        # Define the prompt template string. This string includes placeholders for context and user's question.
        # The system message provides instructions to the model.
        self.prompt_template = ChatPromptTemplate(
            [
                ("system", 
                """
                Role: You are an expert assistant using Retrieval Augmented Generation (RAG).
                Instructions: Use the provided context to answer the user's question.
                Style or tone guidelines:
                - Use clear and concise language.
                - Use bullet points where appropriate.
                - Provide a detailed answer based on the above context.
                - Avoid hype or promotional language
                - Avoid deeply technical jargon
                Output constraints:
                - If the context does not contain the answer, respond with "I don't know".
                - Never use information outside the provided context.
                """),
                ("user", "Context: {context}\n\nQuestion: {question}"),
            ]
        )

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant with query processing and scope validation.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        # Validate query scope
        scope_validation = self.project_scope.validate_scope(input)
        
        # Process the query (normalize and clean)
        query_info = self.query_processor.process(input)
        search_query = query_info["processed"]

        # Search using processed query
        search_results = self.vector_db.search(search_query, n_results=n_results)
        context_chunks = search_results.get("documents", [])
        distances = search_results.get("distances", [])

        # Handle nested lists
        if context_chunks and isinstance(context_chunks[0], list):
            context_chunks = context_chunks[0]
        if distances and isinstance(distances[0], list):
            distances = distances[0]

        # Handle both list of strings and list of dictionaries
        flat_chunks = []
        for chunk in context_chunks:
            if isinstance(chunk, dict):
                flat_chunks.append(chunk.get("content", str(chunk)))
            elif isinstance(chunk, list):
                flat_chunks.extend(chunk)
            else:
                flat_chunks.append(chunk)

        combined_context = "\n\n".join(flat_chunks)

        # Generate answer
        llm_answer = self.chain.invoke(
            {"context": combined_context, "question": query_info["processed"]}
        )
        return llm_answer

    def evaluate_query(self, query: str, relevant_ids: List[str] = None, 
                      n_results: int = 5) -> Dict[str, Any]:
        """
        Evaluate retrieval quality for a query.
        If relevant_ids provided, calculates precision/recall/MRR.
        If not provided, shows retrieval quality based on similarity.
        
        Args:
            query: User query
            relevant_ids: Optional list of relevant chunk IDs (ground truth)
            n_results: Number of results to retrieve
            
        Returns:
            Dictionary with evaluation information
        """
        # Process query
        query_info = self.query_processor.process(query)
        search_query = query_info["processed"]
        
        # Search
        search_results = self.vector_db.search(search_query, n_results=n_results)
        retrieved_ids = search_results.get("ids", [])
        distances = search_results.get("distances", [])
        
        # Handle nested lists from ChromaDB
        if retrieved_ids and isinstance(retrieved_ids[0], list):
            retrieved_ids = retrieved_ids[0]
        if distances and isinstance(distances[0], list):
            distances = distances[0]
        
        result = {
            "query": query,
            "processed_query": search_query,
            "query_type": query_info.get("query_type"),
            "retrieved_count": len(retrieved_ids),
            "retrieved_ids": retrieved_ids[:10]
        }
        
        # If ground truth provided (including empty list), calculate metrics
        if relevant_ids is not None:
            print(f"\n[Evaluation Debug]")
            print(f"Retrieved IDs: {retrieved_ids[:5]}...")
            print(f"Relevant IDs: {relevant_ids}")
            print(f"Matches found: {len(set(retrieved_ids) & set(relevant_ids))}")
            
            metrics = self.evaluator.evaluate(retrieved_ids, relevant_ids)
            result["relevant_count"] = len(relevant_ids)
            result["metrics"] = metrics
        else:
            # No ground truth - show similarity-based quality
            if distances:
                avg_similarity = (1 - sum(distances) / len(distances)) * 100
                min_similarity = (1 - max(distances)) * 100 if distances else 0
                result["average_similarity"] = avg_similarity
                result["min_similarity"] = min_similarity
                result["quality_status"] = "good" if avg_similarity > 70 else "fair" if avg_similarity > 50 else "poor"
                result["warning"] = avg_similarity < 50
        
        return result

def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)
        print("Added sample documents")

        # Show project scope information
        scope_config = assistant.project_scope.get_config()
        print(f"\n{'='*60}")
        print(f"Project: {scope_config.get('project_name')}")
        print(f"Description: {scope_config.get('description')}")
        print(f"\nSupported Domains:")
        for domain in scope_config.get('domains', []):
            print(f"  - {domain.get('display_name')}")
        print(f"{'='*60}\n")

        done = False
        show_quality = False

        while not done:
            print('='*60)
            print("Welcome to the RAG Assistant")
            print('\n')

            question = input("Enter a question, or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                print('\n')
                print("Do you need to enable retrieval quality evaluation? (y/n)")
                enable_quality = input().strip().lower()
                if enable_quality.lower() == "y":
                    show_quality = True
                    print("Retrieval quality evaluation enabled.\n")
                else:
                    show_quality = False
                    print("Retrieval quality evaluation disabled.\n")
                
                # Get answer
                print('Getting answer...')
                result = assistant.invoke(question)
                
                # Evaluate retrieval quality if enabled
                if show_quality:
                    # First, get retrieval info to show user
                    eval_result = assistant.evaluate_query(question, relevant_ids=None)
                    retrieved_ids = eval_result.get('retrieved_ids', [])
                    avg_similarity = eval_result.get('average_similarity', 0)
                    
                    print(f"\n[Retrieval Quality - Basic]")
                    print(f"  Average similarity: {avg_similarity:.1f}%")
                    print(f"  Quality status: {eval_result.get('quality_status', 'unknown')}")
                    if eval_result.get('warning'):
                        print(f"  ⚠️  Warning: Low similarity - answer may not be accurate")
                        print(f"  ⚠️  Your query might be out of scope for the documents!")
                    
                    # Show retrieved chunks with content preview
                    query_info = assistant.query_processor.process(question)
                    search_results = assistant.vector_db.search(query_info["processed"], n_results=5)
                    documents = search_results.get("documents", [])
                    distances = search_results.get("distances", [])
                    
                    if documents and isinstance(documents[0], list):
                        documents = documents[0]
                    if distances and isinstance(distances[0], list):
                        distances = distances[0]
                    
                    print(f"\n{'='*60}")
                    print("Retrieved Chunks (review to identify which are actually relevant):")
                    print(f"{'='*60}")
                    for i, (chunk_id, doc, dist) in enumerate(zip(retrieved_ids, documents, distances), 1):
                        similarity = (1 - dist) * 100
                        preview = doc[:100] + "..." if len(doc) > 100 else doc
                        print(f"\n{i}. ID: {chunk_id} (similarity: {similarity:.1f}%)")
                        print(f"   Content: {preview}")
                    print(f"{'='*60}")
                    
                    # Ask if user wants to provide ground truth for full evaluation
                    print("\nDo you want to provide ground truth (relevant_ids) for full evaluation? (y/n)")
                    provide_ground_truth = input().strip().lower()
                    
                    if provide_ground_truth == "y":
                        print("\nOptions:")
                        print("  a) Enter chunk IDs manually (comma-separated)")
                        print("     Example: doc_0_chunk_0, doc_0_chunk_1")
                        print("  b) Type 'use_retrieved' to mark all retrieved as relevant")
                        print("     (Only use if you're sure they all answer your question!)")
                        print("  c) Type 'none' if query is out of scope (no chunks are relevant)")
                        
                        relevant_input = input("\nYour choice: ").strip()
                        
                        if relevant_input.lower() == "none" or relevant_input.lower() == "no":
                            relevant_ids = []
                            print("No chunks marked as relevant (query is out of scope).")
                        elif relevant_input.lower() == "use_retrieved":
                            relevant_ids = retrieved_ids
                            if avg_similarity < 50:
                                print(f"⚠️  Warning: Low similarity detected. These may not be truly relevant!")
                            print(f"Using all {len(relevant_ids)} retrieved chunks as relevant.")
                        else:
                            relevant_ids = [id.strip() for id in relevant_input.split(",") if id.strip()]
                            print(f"Using {len(relevant_ids)} chunk(s) as relevant.")
                        print('\n')
                        
                        # Always evaluate (even with empty list for out-of-scope queries)
                        full_eval = assistant.evaluate_query(question, relevant_ids=relevant_ids)
                        print(f"\n{'='*60}")
                        print("Full Evaluation Results (with ground truth):")
                        print(f"{'='*60}")
                        print(f"Retrieved: {full_eval['retrieved_count']} | Relevant: {full_eval.get('relevant_count', 0)}")
                        if 'metrics' in full_eval:
                            metrics = full_eval['metrics']
                            print(f"\nMetrics:")
                            print(f"  Precision@1: {metrics['precision'].get('P@1', 0):.3f}")
                            print(f"  Precision@3: {metrics['precision'].get('P@3', 0):.3f}")
                            print(f"  Precision@5: {metrics['precision'].get('P@5', 0):.3f}")
                            print(f"  Recall@5: {metrics['recall'].get('R@5', 0):.3f}")
                            print(f"  MRR: {metrics['mrr']:.3f}")
                            print(f"  NDCG@5: {metrics['ndcg'].get('NDCG@5', 0):.3f}")
                        print(f"{'='*60}\n")
                    print()
                
                print(f"{result}\n")

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
