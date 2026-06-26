Document Intelligence Project:An end-to-end Hybrid-RAG pipeline featuring Reciprocal Rank Fusion, Streamlit UI, and multi-provider LLM routing.
1. Project Overview:This Document Intelligence assistant is designed to query complex documents accurately without hallucinating. Built with a robust Hybrid Retrieval-Augmented Generation (RAG) architecture, the application dynamically routes queries between semantic search and keyword search depending on the user's intent. It handles document ingestion, parallel retrieval, cross-encoder reranking, and generation through a clean Streamlit interface.
2. Key Features:-
(a).Hybrid Retrieval Pipeline: Executes parallel vector and vectorless searches (BM25/SQLite/Elasticsearch) for maximum context retrieval.
(b).Reciprocal Rank Fusion (RRF): Merges conflicting retrieval scores mathematically (using a standard (k=60) dampening factor) to surface the absolute best document chunks. The formula used is:
RRF(d) = \sum_{i \in \{\text{vector, vectorless}\}} \frac{1}{k + rank_i(d)}
(c).Multi-Model LLM Routing: Seamlessly switches between Groq (e.g., Llama models) and Google GenAI APIs based on configuration.(d).Cross-Encoder Reranking: Re-evaluates and heavily scrutinizes retrieved chunks before feeding them to the LLM to guarantee high-fidelity answers.
(e).Deterministic Vector Indexing: Utilizes a deterministic hashing method, such as generating a UUID from the chunk ID, to prevent database duplication. Qdrant natively accepts these UUID strings as valid point IDs.
(f).Secure Secrets Management: Follows industry standards for environment variables and Streamlit secrets to ensure API keys are never leaked to the frontend.
3. Technology Stack-
Frontend: Streamlit 
Vector Database: Qdrant (via Docker) 
Embedding Models:BAAI/bge-large-en-v1.5 (Local) and Google GenAI 
LLM Providers: Groq, Google GenAI 
Data Models: Pydantic V2 
4. Setup & Installation
Step 1: Install Dependencies Ensure you have a modern Python environment active, then install the required packages:using command :- uv add -r requirements.txt
(Note: To resolve potential import errors for evaluation metrics, ensure rouge-score, bert-score, datasets, and ragas are included in your requirements.)
Step 2: Configure Environment Variables Create a .env file in the root directory of the project and securely add your API keys:Code snippet --
GROQ_API_KEY="your_groq_api_key"
GOOGLE_API_KEY="your_google_api_key"
Step 3: Start the Qdrant Vector Database
You must run the Qdrant vector database locally using Docker before starting the pipeline. Use the following command, ensuring you define an absolute path for your local volume mapping:
docker run -p 6333:6333 -p 6334:6334 -v "C:\Your\Actual\Path\Here:/qdrant/storage" qdrant/qdrant
Step 4: Run the Application
Run your ingestion/indexing script first to populate the database with text embeddings.Launch the Streamlit user interface:Bash
streamlit run app/streamlit.py