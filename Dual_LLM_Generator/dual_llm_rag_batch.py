import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
import google.generativeai
import ollama
import requests
# import streamlit as st # REMOVED Streamlit
import pandas as pd # ADDED Pandas
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine
from google import genai
from google.genai import types
from langchain.chains import RetrievalQA
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai import VertexAI
import re
import json
import argparse
import sys

# --- Configuration Constants ---
# Keep existing constants, adjust defaults if needed for non-UI run
DEBUG = False # Set to True for more verbose logging
PROMPT_FOLDER = "../Prompts"
SYSTEM_PROMPT_FILE = "system_instruction.txt"
REPHRASER_PROMPT_FILE = "rephraser.txt"
SUMMARIZER_PROMPT_FILE = "summarizer.txt"
JUDGE_MODEL_NAME_FILE = "judge_model_name.txt" # Added constant
JUDGE_PROMPT_FILE = "judge_prompt.txt" # Added constant
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/tags"
DEFAULT_OLLAMA_TIMEOUT = 10 # seconds
DEFAULT_GEMINI_TEMPERATURE = 0.3
DEFAULT_GEMINI_MAX_TOKENS = 2048
DEFAULT_GEMINI_TOP_P = 0.95
DEFAULT_VERTEX_SEARCH_MODEL = "gemini-2.0-flash-001" # Often used for RAG/retriever
DEFAULT_DATA_STORE_LOCATION = "global"
DEFAULT_BRANCH_NAME = "default_branch"
DEFAULT_COLLECTION = "default_collection"
RAG_SKIP_PHRASES = ["I am not able to answer this question", "No RAG required"]
GEMINI_MODEL_PREFIXES = ["gemini-1.5", "gemini-2."] # Adjusted for common models
LEFT_MODEL_NAME = "gemini-2.0-flash-001"
RIGHT_MODEL_NAME = "gemini-2.0-flash-lite-001"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__) # Use logger instance

# --- Suppress Specific Warnings ---
warnings.filterwarnings("ignore") # Suppress broadly for simplicity, refine if needed

# --- Helper Functions (Modified to remove Streamlit) ---

def log_debug(message: str):
    """Logs a debug message if DEBUG is True."""
    if DEBUG:
        logger.debug(message)
        # Removed st.sidebar.write


def load_text_file(filename: str) -> str: # Changed return type, raise error on failure
    """
    Loads text content from a file.

    Args:
        filename: The path to the file.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
        Exception: For other unexpected errors.
    """
    log_debug(f"Attempting to load file: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filename}.")
        raise # Re-raise the exception
    except IOError as e:
        logger.error(f"Error: Could not read file {filename}. Reason: {e}")
        raise # Re-raise the exception
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filename}: {e}")
        raise # Re-raise the exception


def create_gemini_client(project_id: Optional[str] = None, location: Optional[str] = None) -> genai.Client:
    """
    Creates a Gemini client, using Vertex AI if project/location provided, else API Key.

    Args:
        project_id: Google Cloud Project ID for Vertex AI.
        location: Google Cloud Location for Vertex AI.

    Raises:
        ValueError: If required environment variables or args are missing.

    Returns:
        An initialized genai.Client.
    """
    # Check if Vertex AI configuration is intended (explicit args override env var logic)
    use_vertex = bool(project_id and location)
    if not use_vertex and os.environ.get("GOOGLE_VERTEXAI", "").lower() == "true":
        # Fallback to env vars if args not provided but env var suggests Vertex
         project_id = project_id or os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
         location = location or os.getenv("LOCATION") or os.getenv("GOOGLE_CLOUD_REGION")
         if project_id and location:
              use_vertex = True
         else:
              logger.warning("GOOGLE_VERTEXAI is true, but PROJECT_ID or LOCATION missing. Trying API Key.")


    if use_vertex:
        if not project_id or not location:
             # This case should ideally be caught by argument parsing if required
            raise ValueError("Vertex AI requires Project ID and Location.")
        log_debug(f"Creating Vertex AI client for project {project_id} in {location}")
        try:
            return genai.Client(vertexai=True, project=project_id, location=location)
        except Exception as e:
             logger.error(f"Failed to create Vertex AI Client: {e}")
             raise ValueError(f"Could not initialize Vertex AI Client: {e}") from e

    else:
        # Gemini Developer API configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API requires GOOGLE_API_KEY environment variable set "
                "(or provide Project ID/Location for Vertex AI)."
            )
        log_debug("Creating Gemini API client using API Key.")
        try:
             # Use google.generativeai directly for API key
             google.generativeai.configure(api_key=api_key)
             # Return a configured model object perhaps, or manage client differently
             # For simplicity, let's assume generate_gemini_response will handle it
             # Returning a dummy or specific client might be needed depending on exact google.generativeai usage
             # Let's return None and handle client creation within generate_gemini_response for API key case
             # Revisit this if direct client object is needed beforehand.
             logger.info("Google API Key configured. Client creation deferred to generation function.")
             return None # Placeholder, signifies API Key usage
        except Exception as e:
             logger.error(f"Failed to configure Google API Key Client: {e}")
             raise ValueError(f"Could not initialize Google API Key Client: {e}") from e

def _load_prompt_template(filename: str) -> str: # Ensure non-optional return
    """Loads a prompt template file from the PROMPT_FOLDER."""
    filepath = os.path.join(PROMPT_FOLDER, filename)
    # load_text_file now raises error on failure
    return load_text_file(filepath)

def _apply_system_prompt(prompt_text: str, system_instruction: Optional[str]) -> str:
    """Prepends the system prompt if provided."""
    if system_instruction and system_instruction.strip():
        return f"{system_instruction.strip()}\n\n{prompt_text}"
    return prompt_text

def get_rephraser_prompt(query: str, system_instruction: Optional[str]) -> str:
    """
    Loads and formats the rephraser prompt.

    Args:
        query: The user's original query.
        system_instruction: Optional system instruction text.

    Returns:
        The formatted prompt string.

    Raises:
        FileNotFoundError, IOError, KeyError, Exception: If loading/formatting fails.
    """
    try:
        prompt_template = _load_prompt_template(REPHRASER_PROMPT_FILE)
        # Assuming prompt_template is the static part, and query needs adding.
        # Adjust formatting based on actual template structure.
        formatted_prompt_body = f"""Now, please rephrase the following customer query:

{query}
"""
        # Combine template + body + system prompt
        full_prompt = prompt_template + formatted_prompt_body
        return _apply_system_prompt(full_prompt, system_instruction)

    except KeyError as e:
        logger.error(f"Error formatting rephraser prompt: Missing key {e}")
        raise # Re-raise
    except Exception as e:
        logger.error(f"Unexpected error formatting rephraser prompt: {e}")
        raise # Re-raise


def get_summarizer_prompt(documents: List[str], query: str, system_instruction: Optional[str]) -> str:
    """
    Loads and formats the summarizer prompt with retrieved documents.

    Args:
        documents: A list of context documents.
        query: The user's query (potentially rephrased).
        system_instruction: Optional system instruction text.

    Returns:
        The formatted prompt string.

    Raises:
        FileNotFoundError, IOError, KeyError, Exception: If loading/formatting fails.
    """
    try:
        prompt_template = _load_prompt_template(SUMMARIZER_PROMPT_FILE)
        document_vars = {}
        if documents:
            for i, doc in enumerate(documents):
                # Limit document length if necessary before formatting
                document_vars[f"Text_of_Document_{i + 1}"] = doc #[:3000] # Example limit
        else:
            document_vars["Text_of_Document_1"] = "No relevant documents found."

        format_args = {"query": query}
        # Ensure template placeholders match required args
        max_docs_in_template = 3 # Assume template expects up to 3 docs
        for i in range(1, max_docs_in_template + 1):
            key = f"Text_of_Document_{i}"
            format_args[key] = document_vars.get(key, "No document provided for this slot.")

        # Define the dynamic part (query, docs) - adjust based on actual template needs
        formatted_prompt_body = f"""
Now it's your turn! Here is the query and relevant documents:
Customer Search Query: {format_args['query']}

Document Texts:
[Start of Document 1]
{format_args['Text_of_Document_1']}
[End of Document 1]

[Start of Document 2]
{format_args['Text_of_Document_2']}
[End of Document 2]

[Start of Document 3]
{format_args['Text_of_Document_3']}
[End of Document 3]
"""
        # Combine template + body + system prompt
        full_prompt = prompt_template + formatted_prompt_body
        return _apply_system_prompt(full_prompt, system_instruction)

    except KeyError as e:
        logger.error(f"Error formatting summarizer prompt: Missing key {e}")
        raise # Re-raise
    except Exception as e:
        logger.error(f"Unexpected error formatting summarizer prompt: {e}")
        raise # Re-raise


def initialize_models() -> Tuple[List[str], List[str]]:
    """
    Initializes and lists available Gemini and Ollama models.

    Returns:
        A tuple containing two lists: (ollama_model_names, gemini_model_names).
        Returns empty lists if errors occur during listing.
    """
    ollama_models = []
    gemini_models = []

    # List Gemini models (Requires client creation logic or assumes auth is set)
    try:
        # Attempt client creation to list models - might need project/location args here?
        # Or assume default ADC works for listing.
        # Let's assume default ADC works for listing for simplicity here.
        # If specific project/location is needed, this needs more context.
        client = create_gemini_client() # Tries Vertex/API Key based on env
        if client: # Only if Vertex client was created
            response = client.models.list(config={"page_size": 200, "query_base": True})
            gemini_models = [
                model.name.split("/")[-1]
                for model in response.page
                if any(prefix in model.name for prefix in GEMINI_MODEL_PREFIXES)
            ]
        else: # Handle API Key case - google.generativeai list models
            models_list = google.generativeai.list_models()
            gemini_models = [
                m.name.split("/")[-1] for m in models_list if 'generateContent' in m.supported_generation_methods and any(prefix in m.name for prefix in GEMINI_MODEL_PREFIXES)
            ]
        log_debug(f"Found Gemini models: {gemini_models}")
    except Exception as e:
        # Removed st.error
        logging.error(f"Failed to list Gemini models: {e}")

    # List Ollama models
    try:
        response = requests.get(DEFAULT_OLLAMA_URL, timeout=DEFAULT_OLLAMA_TIMEOUT)
        response.raise_for_status()
        ollama_models = [model["name"] for model in response.json().get("models", [])]
        log_debug(f"Found Ollama models: {ollama_models}")
        if not ollama_models:
             logging.warning("Ollama server responded, but no models listed.") # Replaced st.warning
    except requests.exceptions.ConnectionError:
        logging.warning("Ollama server not reachable. Is it running?") # Replaced st.warning
    except requests.exceptions.Timeout:
        logging.warning("Ollama server timed out.") # Replaced st.warning
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama connection failed: {e}") # Replaced st.error
    except Exception as e:
        logging.error(f"Error processing Ollama response: {e}") # Replaced st.error

    return ollama_models, gemini_models

# --- Generation Functions (Modified) ---

def generate_gemini_response(
    prompt: str,
    system_instruction: Optional[str],
    model_config: Dict[str, Any],
    vertex_project: Optional[str] = None, # Added for Vertex context
    vertex_location: Optional[str] = None # Added for Vertex context
) -> Dict[str, Any]:
    """
    Generates a response using the Gemini API (Vertex or API Key).

    Args:
        prompt: The complete prompt string (excluding system instruction).
        system_instruction: The system instruction text.
        model_config: Dictionary containing 'name' and 'temperature'.
        vertex_project: Google Cloud Project ID (if using Vertex).
        vertex_location: Google Cloud Location (if using Vertex).


    Returns:
        A dictionary containing 'text', 'prompt_tokens', 'candidate_tokens',
        'total_tokens', 'time_ms', 'error'.
    """
    start_time_ms = time.time() * 1000.0
    model_name = model_config["name"]
    temperature = model_config.get("temperature", DEFAULT_GEMINI_TEMPERATURE)
    result: Dict[str, Any] = {
        "text": "", "prompt_tokens": None, "candidate_tokens": None,
        "total_tokens": None, "time_ms": 0, "error": None
    }

    log_debug(
        f"Calling Gemini API: model={model_name}, temp={temperature}, "
        f"Vertex=({vertex_project is not None})"
    )
    # Log truncated prompt (apply system prompt conceptually for logging)
    log_prompt = _apply_system_prompt(prompt, system_instruction)
    log_debug(f"Full prompt concept for Gemini:\n{log_prompt[:500]}...")

    try:
        # --- Client and Config Setup ---
        client = None
        
        is_vertex = bool(vertex_project and vertex_location)
        chat_content_config = {
            "system_instruction":system_instruction,
            "temperature": temperature,
            "max_output_tokens": DEFAULT_GEMINI_MAX_TOKENS,
        }

        if is_vertex:
             # Create Vertex Client Instance
             # Note: Consider creating client once outside if calling repeatedly
            client = create_gemini_client(project_id=vertex_project, location=vertex_location)
            if not client: # Should not happen if project/location provided, but check
                 raise ValueError("Failed to create Vertex AI client.")

            client = client.chats.create(model=model_name, config=chat_content_config)
            result["client"] = client

        else:
            # Assume API Key via google.generativeai
            client = model_config["client"]
            
        #### This is a chat model ######
        response = client.send_message_stream(prompt)
        time_taken = 1000.0*(time.time()) - start_time_ms

        full_response = ""
        usage_metadata = None
        for chunk in response:
            chunk_text = chunk.text
            full_response += chunk_text
            # Capture usage metadata from the *last* chunk (usually contains totals)
            if hasattr(chunk, 'usage_metadata'):
                 usage_metadata = chunk.usage_metadata

        result["text"] = full_response
        if usage_metadata:
            result["prompt_tokens"] = getattr(usage_metadata, 'prompt_token_count', None)
            result["candidate_tokens"] = getattr(usage_metadata, 'candidates_token_count', None)
            result["total_tokens"] = getattr(usage_metadata, 'total_token_count', None)

        log_debug(f"Gemini Raw Response: {full_response[:500]}...")
        if usage_metadata:
             log_debug(f"Gemini Usage: {usage_metadata}")

    except Exception as e:
        # Removed st.error
        logging.exception(f"Error during Gemini API call for model {model_name}:")
        result["error"] = str(e)

    result["time_ms"] = (time.time() * 1000.0) - start_time_ms
    log_debug(f"Gemini call took {result['time_ms']:.0f} ms")
    return result


def generate_ollama_response(
    prompt: str, model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates a streaming response using the Ollama Chat API.

    Args:
        prompt: The complete prompt string.
        model_config: Dictionary containing 'name'.

    Returns:
        A dictionary containing 'text', 'time_ms', 'error'.
    """
    import ollama
    
    start_time_ms = time.time() * 1000.0
    model_name = model_config["name"]
    result: Dict[str, Any] = {"text": "", "time_ms": 0, "error": None}

    log_debug(f"Calling Ollama API: model={model_name} (Streaming)")
    log_debug(f"Full prompt for Ollama:\n{prompt[:500]}...")

    full_response = ""
    try:
        # Use the ollama library's streaming chat
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in stream:
            message_content = chunk.get("message", {}).get("content")
            if message_content:
                full_response += message_content
            # Check for final chunk metadata if needed (e.g., duration)
            # The 'ollama' library might not expose detailed tokens/timing per chunk easily.
            # We rely on manual timing.

        result["text"] = full_response
        log_debug(f"Ollama Raw Response: {full_response[:500]}...")

    except Exception as e:
        # Removed st.error
        logging.exception(f"Error during Ollama API call for model {model_name}:")
        result["error"] = str(e)

    result["time_ms"] = (time.time() * 1000.0) - start_time_ms
    log_debug(f"Ollama call took {result['time_ms']:.0f} ms")
    return result


def generate_rephraser_response(
    model_config: Dict[str, Any],
    question: str,
    system_instruction: Optional[str],
    vertex_project: Optional[str] = None, # Pass through context
    vertex_location: Optional[str] = None, # Pass through context
) -> Dict[str, Any]:
    """
    Orchestrates rephrased query generation using the selected model.

    Args:
        model_config: Configuration for the selected model {'type', 'name', 'temperature'}.
        question: The original user question.
        system_instruction: Optional system instruction.
        vertex_project: Project ID if using Vertex Gemini.
        vertex_location: Location if using Vertex Gemini.


    Returns:
        Dict containing 'text' (rephrased query), 'time_ms', 'error'.
    """
    start_time = time.time()*1000
    result: Dict[str, Any] = {"text": question, "time_ms": 0, "error": None} # Default to original question

    try:
        log_debug(f"Generating rephrased query with config: {model_config}")
        # Get formatted prompt (raises error on failure)
        rephraser_prompt = get_rephraser_prompt(question, system_instruction)

        model_type = model_config.get("type")
        generation_result: Dict[str, Any] = {}

        # Removed UI placeholders

        # Generate response
        if model_type == "gemini":
            generation_result = generate_gemini_response(
                 rephraser_prompt, system_instruction, model_config, vertex_project, vertex_location
            )
        elif model_type == "3p_models": # Assuming this means Ollama now
            generation_result = generate_ollama_response(
                 rephraser_prompt, model_config
            )
        else:
            raise ValueError(f"Unsupported model type for rephrasing: {model_type}")

        if generation_result.get("error"):
             raise Exception(f"Generation failed: {generation_result['error']}")
        
        rephrased_query_raw = generation_result.get("text", "")

        # Clean up response (JSON parsing and RAG skip phrases)
        # Assuming process_rephraser_response and clean_json are defined elsewhere
        # and primarily handle potential JSON output and cleanup.
        try:
            # Pass raw response to cleaning/parsing
            rephrased_query_cleaned = process_rephraser_response(rephrased_query_raw)
            # Remove RAG skip phrases after potential JSON parsing
            for phrase in RAG_SKIP_PHRASES:
                rephrased_query_cleaned = rephrased_query_cleaned.replace(phrase, "").strip()
            result["text"] = rephrased_query_cleaned

        except Exception as clean_e:
            logger.warning(f"Could not clean or parse rephraser response: {clean_e}. Using raw response.")
            # Use raw response but still remove skip phrases
            rephrased_query_cleaned_fallback = rephrased_query_raw
            for phrase in RAG_SKIP_PHRASES:
                rephrased_query_cleaned_fallback = rephrased_query_cleaned_fallback.replace(phrase, "").strip()
            result["text"] = rephrased_query_cleaned_fallback


        log_debug(f"Rephrased Query (Cleaned): {result['text']}")
        
    except Exception as e:
        logger.error(f"Error generating rephrased response: {e}", exc_info=True)
        result["error"] = str(e)
        

    result["time_ms"] = (time.time()*1000 - start_time) 
    result["client"] = generation_result["client"]
    result['raw_text'] = rephrased_query_raw
    log_debug(f"Rephrasing time: {result['time_ms']:.0f} ms")
    return result


def generate_summarizer_response(
    model_config: Dict[str, Any],
    question: str, # Potentially rephrased
    context: List[str],
    system_instruction: Optional[str],
    vertex_project: Optional[str] = None, # Pass through context
    vertex_location: Optional[str] = None, # Pass through context
) -> Dict[str, Any]:
    """
    Orchestrates final response generation using the selected model and context.

    Args:
        model_config: Configuration for the selected model.
        question: The query to answer (potentially rephrased).
        context: A list of context documents (strings). Empty list if no RAG.
        system_instruction: Optional system instruction.
        vertex_project: Project ID if using Vertex Gemini.
        vertex_location: Location if using Vertex Gemini.

    Returns:
        Dict containing 'text', 'time_ms', 'error', plus LLM usage if available.
    """
    start_time = (time.time())*1000
    result: Dict[str, Any] = {"text": "", "time_ms": 0, "error": None} # Initialize common keys

    try:
        log_debug(f"Generating summarizer response with config: {model_config}")
        # Get formatted prompt (raises error on failure)
        summarizer_prompt = get_summarizer_prompt(context, question, system_instruction)

        model_type = model_config.get("type")
        generation_result: Dict[str, Any] = {}

        # Removed UI placeholders

        if model_type == "gemini":
            generation_result = generate_gemini_response(
                summarizer_prompt, system_instruction, model_config, vertex_project, vertex_location
            )
        elif model_type == "3p_models": # Assuming Ollama
            generation_result = generate_ollama_response(
                 summarizer_prompt, model_config
            )
        else:
            raise ValueError(f"Unsupported model type for summarizing: {model_type}")

        # Merge generation result into the final result dict
        result.update(generation_result) # Copies text, tokens, time, error etc.

        if result.get("error"):
             raise Exception(f"Generation failed: {result['error']}")

        log_debug(f"Final Raw Response: {result.get('text', '')[:500]}...")

    except Exception as e:
        logger.error(f"Error generating summarizer response: {e}", exc_info=True)
        result["error"] = str(e) # Ensure error is captured

    # Recalculate time for the whole summarization step if needed,
    # or rely on time from generation_result
    result["time_ms"] = time.time()*1000 - start_time
    log_debug(f"Summarization step time: {result['time_ms']:.0f} ms")

    # Clean up result dict - remove redundant time if nested, ensure keys exist
    final_result = {
         "text": result.get("text", ""),
         "time_ms": result.get("time_ms", 0), # Use overall time
         "error": result.get("error"),
         "prompt_tokens": result.get("prompt_tokens"),
         "candidate_tokens": result.get("candidate_tokens"),
         "total_tokens": result.get("total_tokens"),
    }

    return final_result


def _get_discoveryengine_client_options(location: str) -> Optional[ClientOptions]:
    """ Helper to create client options for Discovery Engine."""
    if location != "global":
        api_endpoint = f"{location}-discoveryengine.googleapis.com"
        log_debug(f"Using Discovery Engine endpoint: {api_endpoint}")
        return ClientOptions(api_endpoint=api_endpoint)
    log_debug("Using global Discovery Engine endpoint.")
    return None


# --- RAG Functions (Modified) ---

def setup_retriever_and_llm(
    project_id: str,
    location: str, # Location for the datastore listing/retrieval
    data_store_id: str,
    data_store_region: str,
    retriever_llm_model: str = DEFAULT_VERTEX_SEARCH_MODEL,
) -> Tuple[Optional[VertexAISearchRetriever], Optional[VertexAI]]:
    """
    Sets up the Vertex AI Search retriever and the LLM used by it.

    Args:
        project_id: Google Cloud project ID.
        location: Location for the datastore.
        data_store_id: The ID of the target data store.
        retriever_llm_model: The Vertex AI model name for the retriever LLM.

    Returns:
        A tuple (retriever, llm), or (None, None) if setup fails.
    """
    retriever = None
    llm = None
    log_debug(f"Setting up retriever for {project_id}/{location}/{data_store_id}")

    if not project_id or not location or not data_store_id:
        logger.warning("Missing project ID, location, or data store ID for retriever setup.")
        return None, None

    # Define the LLM for the retriever
    try:
        llm = VertexAI(model_name=retriever_llm_model, project=project_id, location=location) # Use location for LLM too? Check API reqs.
        log_debug(f"Retriever LLM ({retriever_llm_model}) initialized.")
    except Exception as e:
        # Removed st.error
        logging.exception(f"Failed to initialize VertexAI LLM for retriever ({retriever_llm_model}):")
        return None, None

    # Create the retriever
    try:
        retriever = VertexAISearchRetriever(
            project_id=project_id,
            location_id=data_store_region, # Use provided location
            data_store_id=data_store_id,
            get_extractive_answers=False, # Keep configurable if needed
            max_documents=3,
        )
        log_debug(f"Vertex AI Search Retriever created successfully for {data_store_id}")
    except Exception as e:
        # Removed st.error
        logging.exception(f"Failed to create Vertex AI Search retriever for {data_store_id}:")
        return None, llm # Return LLM even if retriever fails? Maybe return None, None

    return retriever, llm


def get_relevant_docs(
    search_query: str,
    retriever: Optional[VertexAISearchRetriever],
    llm: Optional[VertexAI]
) -> List[str]:
    """
    Retrieves relevant documents from Vertex AI Search using the provided retriever.

    Args:
        search_query: The query string (potentially rephrased).
        retriever: The initialized VertexAISearchRetriever instance.
        llm: The initialized VertexAI LLM instance for the QA chain.

    Returns:
        A list of document contents as strings. Returns empty list if error,
        no retriever, or no query.
    """
    if not retriever or not llm:
        log_debug("Retriever or LLM not available. Skipping RAG.")
        return []
    if not search_query or not search_query.strip():
        log_debug("Empty search query received. Skipping RAG.")
        return []

    log_debug(f"Retrieving documents for query: {search_query}")
    start_time = (time.time())*1000
    documents_content = []
    
    try:
        # Setup RetrievalQA chain (consider creating once if multiple calls)
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        # Langchain invoke might raise errors directly
        results = retrieval_qa.invoke({"query": search_query}) # Ensure input matches chain expectations
        elapsed_time = (time.time())*1000 - start_time
        log_debug(f"Document retrieval via Langchain time: {elapsed_time:.1f} seconds")

        if results and "source_documents" in results:
            for i, doc in enumerate(results["source_documents"]):
                # Format consistently for summarizer prompt
                header = f"[Start of Document {i+1}]\n"
                footer = f"\n[End of Document {i+1}]"
                doc_content = getattr(doc, 'page_content', '')
                # Simple formatting, adjust if needed
                documents_content.append(header + doc_content + footer)
                log_debug(f"Retrieved Doc {i+1}: {len(doc_content)} chars")

        log_debug(f"Retrieved {len(documents_content)} documents.")


    except Exception as e:
        # Removed st.error
        logging.exception("Error during document retrieval:")
        # Return empty list on failure
        return []

    return documents_content


# --- Judge Function (Modified) ---

def judge_responses(
    left_question: str, left_response_text: str, left_context: List[str],
    right_question: str, right_response_text: str, right_context: List[str],
    judge_model_name: str, # Pass model name
    judge_system_instruction: Optional[str], # Pass system prompt
    vertex_project: Optional[str] = None, # Pass context
    vertex_location: Optional[str] = None # Pass context
) -> Dict[str, Any]:
    """
    Judges the responses from two models using a specified Gemini model.

    Args:
        left_question, right_question: The final queries used for each model.
        left_response_text, right_response_text: The generated text responses.
        left_context, right_context: The context documents used (list of strings).
        judge_model_name: The name of the Gemini model to use for judging.
        judge_system_instruction: The system prompt for the judge model.
        vertex_project: Project ID if using Vertex Gemini.
        vertex_location: Location if using Vertex Gemini.

    Returns:
        Dict containing 'text' (judge's verdict), 'time_ms', 'error', plus usage.
    """
    start_time = time.time()
    result: Dict[str, Any] = {"text": "", "time_ms": 0, "error": None}

    log_debug(f"Judging responses using model: {judge_model_name}")

    # Format context for inclusion in the prompt
    def format_context(context_list: List[str]) -> str:
        if not context_list:
            return "No context provided (RAG likely disabled or failed)."
        # Join documents, potentially adding separators
        return "\n\n".join(context_list)

    formatted_left_context = format_context(left_context)
    formatted_right_context = format_context(right_context)

    try:
        # Format the judge prompt
        # Ensure placeholders match the actual prompt file content
        judge_prompt_body = f"""Here are two responses from different language models based on the QUESTION and CONTEXT:

Response from model on the left:
QUESTION:
{left_question}

CONTEXT:
{formatted_left_context}

Response A (Model on the Left):
{left_response_text}

Response from model on the right:
QUESTION:
{right_question}

CONTEXT:
{formatted_right_context}

Response B (Model on the Right):
{right_response_text}

Which one accurately responds to the question using the source of truth? Make sure your verdict is based on each model's strict adherence to the source of truth.
"""
        # Assuming judge_system_instruction comes from loaded file
        # Apply system prompt if provided for judge
        judge_full_prompt = _apply_system_prompt(judge_prompt_body, judge_system_instruction)


        # Configure judge model (hardcoded temp/params for now)
        judge_model_config = {
            "name": judge_model_name,
            "temperature": 0.5 # Example temperature for judge
        }

        # Generate the judgment using the Gemini function
        judge_generation_result = generate_gemini_response(
            judge_prompt_body, # Pass main prompt body
            judge_system_instruction, # Pass system instruction separately
            judge_model_config,
            vertex_project,
            vertex_location
        )

        # Update result dict with judge's response and metadata
        result.update(judge_generation_result)

        if result.get("error"):
            raise Exception(f"Judge model failed: {result['error']}")

        log_debug(f"Judge Raw Verdict: {result.get('text', '')[:500]}...")

    except Exception as e:
        logger.error(f"Error in judge_responses: {e}", exc_info=True)
        result["error"] = str(e)

    result["time_ms"] = (time.time() - start_time) * 1000.0 # Overall judge time
    log_debug(f"Judging time: {result['time_ms']:.0f} ms")
    # Clean up result dict for consistency
    final_result = {
         "text": result.get("text", ""),
         "time_ms": result.get("time_ms", 0),
         "error": result.get("error"),
         "prompt_tokens": result.get("prompt_tokens"),
         "candidate_tokens": result.get("candidate_tokens"),
         "total_tokens": result.get("total_tokens"),
    }
    return final_result


# --- Utility Functions (Kept from original code) ---
def clean_json(respo):
    # removing any markdown block that might appear
    respo = respo.replace("{{","{").replace("}}","}")
    pattern = r"(?:^```.*)"
    modified_text = re.sub(pattern, '', respo, 0, re.MULTILINE)
    try:
        result = json.loads(modified_text)
    except:
        # Simple fallback, consider more robust error handling or returning None
        result = {"intent": modified_text, "es_intent": modified_text, "is_trouble":"No", "cot": "None"}
    return result

def process_rephraser_response(txt):
    """
    Extracts relevant part from potentially JSON-formatted rephraser output.
    """
    try:
        # Assuming clean_json tries to parse JSON
        eval_txt = clean_json(txt)
        log_debug(f"Parsed rephrased query object: {eval_txt}")
        # Prefer specific keys if they exist, otherwise return the main text/intent
        if isinstance(eval_txt, dict):
             if "es_intent" in eval_txt: return eval_txt["es_intent"]
             if "rephrased_query" in eval_txt: return eval_txt["rephrased_query"]
             if "intent" in eval_txt: return eval_txt["intent"]
             # Fallback if dict but no expected keys
             return str(eval_txt)
        else:
             # If clean_json returned a string (parse failed)
             return str(eval_txt)
    except Exception as e:
        # Log error during processing, return original text as fallback
        log_debug(f"Error processing rephrased query, returning as-is: {e}")
        return txt # Return original text on any processing error

# --- Main Execution Logic (Refactored) ---
def run_dual_llm_pipeline(
    args: argparse.Namespace,
    questions: List[str], # List of questions to process
    left_config: Dict[str, Any],
    right_config: Dict[str, Any],
    system_instruction: Optional[str],
    judge_system_instruction: Optional[str],
    judge_model_name: Optional[str],
) -> pd.DataFrame:
    """
    Runs the run_dual_llm_pipeline for a list of questions without UI.

    Args:
        args: Parsed command-line arguments containing RAG/Judge settings.
        questions: A list of user questions to evaluate.
        left_config: Configuration dict for the 'left' model.
        right_config: Configuration dict for the 'right' model.
        system_instruction: System prompt for rephraser/summarizer.
        judge_system_instruction: System prompt for the judge model.
        judge_model_name: Name of the judge model.

    Returns:
        A Pandas DataFrame containing the results for all questions.
    """
    all_results = []
    retriever = None
    retriever_llm = None
    rag_active = False

    # --- Setup RAG (if enabled via args) ---
    if args.use_rag:
        if not args.project_id or not args.location or not args.datastore_id:
             logger.warning("RAG enabled via args, but project/location/datastore ID missing. Disabling RAG.")
             print("Warning: RAG requested but connection details missing. Skipping RAG.", file=sys.stderr)
        else:
             logger.info(f"Setting up RAG with Vertex AI Search: {args.project_id}/{args.location}/{args.datastore_id}")
             retriever, retriever_llm = setup_retriever_and_llm(
                 args.project_id, args.location, args.datastore_id, args.datastore_region
             )
             if retriever and retriever_llm:
                 rag_active = True
                 logger.info("RAG setup successful.")
             else:
                 logger.error("RAG setup failed. Proceeding without RAG.")
                 print("Error: Failed to set up RAG. Continuing without RAG.", file=sys.stderr)
    else:
         logger.info("RAG is disabled via arguments.")


    # --- Process Each Question ---
    for i, user_question in enumerate(questions):
        logger.info(f"--- Processing Question {i+1}/{len(questions)}: '{user_question}' ---")
        result_for_question: Dict[str, Any] = {
            "question_index": i + 1,
            "timestamp": time.time(),
            "user_question": user_question,
            "use_rag_configured": args.use_rag,
            "rag_active_for_run": rag_active, # Was RAG successfully set up?
            #"left_config": json.dumps(left_config), # Store config as string
            #"right_config": json.dumps(right_config),
        }

        # --- Step 1: Rephrase Query ---
        # Pass Vertex project/location if left/right model is Gemini on Vertex
        left_vp = args.project_id if left_config.get("type") == "gemini" else None
        left_vl = args.location if left_config.get("type") == "gemini" else None
        right_vp = args.project_id if right_config.get("type") == "gemini" else None
        right_vl = args.location if right_config.get("type") == "gemini" else None

        logger.info("Rephrasing query for Left model...")
        left_rephraser_result = generate_rephraser_response(
            left_config, user_question, system_instruction, left_vp, left_vl
        )
        result_for_question[f"{LEFT_MODEL_NAME}_rephrased_raw"] = left_rephraser_result.get('raw_text')
        result_for_question[f"{LEFT_MODEL_NAME}_rephrased_text"] = left_rephraser_result.get('text')
        result_for_question["left_rephraser_time_ms"] = left_rephraser_result.get('time_ms')
        result_for_question["left_rephraser_error"] = left_rephraser_result.get('error')
        left_config["client"] = left_rephraser_result["client"]

        logger.info("Rephrasing query for Right model...")
        right_rephraser_result = generate_rephraser_response(
             right_config, user_question, system_instruction, right_vp, right_vl
        )
        result_for_question[f"{RIGHT_MODEL_NAME}_rephrased_raw"] = right_rephraser_result.get('raw_text')
        result_for_question[f"{RIGHT_MODEL_NAME}_rephrased_text"] = right_rephraser_result.get('text')
        result_for_question["right_rephraser_time_ms"] = right_rephraser_result.get('time_ms')
        result_for_question["right_rephraser_error"] = right_rephraser_result.get('error')
        right_config["client"] = right_rephraser_result["client"]

        left_query_for_rag = result_for_question[f"{LEFT_MODEL_NAME}_rephrased_text"]
        right_query_for_rag = result_for_question[f"{RIGHT_MODEL_NAME}_rephrased_text"]

        # --- Step 2: Retrieve Context (Conditional RAG) ---
        left_context, right_context = [], []

        if rag_active:
            # Check if rephrased query suggests skipping RAG
            left_skip_rag = any(phrase in (left_query_for_rag or "") for phrase in RAG_SKIP_PHRASES)
            if left_skip_rag:
                 logger.info("Left query suggests skipping RAG based on skip phrases.")
                 left_final_query = user_question # Use original if RAG skipped by phrase
            else:
                 logger.info("Retrieving context for Left query...")
                 left_context = get_relevant_docs(left_query_for_rag, retriever, retriever_llm)
                 left_final_query = left_query_for_rag # Use rephrased

            right_skip_rag = any(phrase in (right_query_for_rag or "") for phrase in RAG_SKIP_PHRASES)
            if right_skip_rag:
                 logger.info("Right query suggests skipping RAG based on skip phrases.")
                 right_final_query = user_question
            else:
                 logger.info("Retrieving context for Right query...")
                 right_context = get_relevant_docs(right_query_for_rag, retriever, retriever_llm)
                 right_final_query = right_query_for_rag
        else:
             # No RAG active, use rephrased (or original if rephrase failed) query directly
             left_final_query = left_query_for_rag
             right_final_query = right_query_for_rag

        result_for_question["left_num_docs"] = len(left_context)
        result_for_question["left_final_query"] = left_final_query
        result_for_question["right_num_docs"] = len(right_context)
        result_for_question["right_final_query"] = right_final_query
        # Optionally store context text itself (can be large)
        # result_for_question["left_context_text"] = json.dumps(left_context)
        # result_for_question["right_context_text"] = json.dumps(right_context)

        # --- Step 3: Generate Final Answer (Summarization) ---
        logger.info("Generating final answer for Left model...")
        left_response_data = generate_summarizer_response(
            left_config, left_final_query, left_context, system_instruction, left_vp, left_vl
        )
        result_for_question[f"{LEFT_MODEL_NAME}_response_text"] = left_response_data.get('text')
        result_for_question["left_response_time_ms"] = left_response_data.get('time_ms')
        result_for_question["left_response_error"] = left_response_data.get('error')
        result_for_question["left_response_prompt_tokens"] = left_response_data.get('prompt_tokens')
        result_for_question["left_response_candidate_tokens"] = left_response_data.get('candidate_tokens')
        result_for_question["left_response_total_tokens"] = left_response_data.get('total_tokens')


        logger.info("Generating final answer for Right model...")
        right_response_data = generate_summarizer_response(
             right_config, right_final_query, right_context, system_instruction, right_vp, right_vl
        )
        result_for_question[f"{RIGHT_MODEL_NAME}_response_text"] = right_response_data.get('text')
        result_for_question["right_response_time_ms"] = right_response_data.get('time_ms')
        result_for_question["right_response_error"] = right_response_data.get('error')
        result_for_question["right_response_prompt_tokens"] = right_response_data.get('prompt_tokens')
        result_for_question["right_response_candidate_tokens"] = right_response_data.get('candidate_tokens')
        result_for_question["right_response_total_tokens"] = right_response_data.get('total_tokens')


        # --- Step 4: Run Judge Model (Conditional) ---
        result_for_question["judge_enabled"] = args.judge
        result_for_question["judge_verdict"] = None
        result_for_question["judge_error"] = None
        result_for_question["judge_time_ms"] = None
        # Add judge token counts if needed

        if args.judge:
            if not judge_model_name:
                 logger.warning("Judge model enabled but judge model name not loaded. Skipping judge.")
                 result_for_question["judge_error"] = "Judge model name missing"
            elif result_for_question["left_response_error"] or result_for_question["right_response_error"]:
                logger.warning("Skipping judge because one or both models failed to generate a response.")
                result_for_question["judge_error"] = "Skipped due to generation error in base models"
            else:
                logger.info("Running Judge model...")
                # Assume judge uses Vertex AI Gemini, pass project/location
                judge_result = judge_responses(
                     left_final_query, result_for_question[f"{LEFT_MODEL_NAME}_response_text"], left_context,
                     right_final_query, result_for_question[f"{RIGHT_MODEL_NAME}_response_text"], right_context,
                     judge_model_name,
                     judge_system_instruction,
                     args.project_id, # Pass project/location for judge model
                     args.location
                )
                result_for_question["judge_verdict"] = judge_result.get('text')
                result_for_question["judge_error"] = judge_result.get('error')
                result_for_question["judge_time_ms"] = judge_result.get('time_ms')
                # Add judge token counts if needed
                result_for_question["judge_prompt_tokens"] = judge_result.get('prompt_tokens')
                result_for_question["judge_candidate_tokens"] = judge_result.get('candidate_tokens')
                result_for_question["judge_total_tokens"] = judge_result.get('total_tokens')


        all_results.append(result_for_question)
        logger.info(f"--- Finished Processing Question {i+1} ---")
        # Optional: Add a small delay between questions if needed
        # time.sleep(1)

    # --- Convert results to DataFrame ---
    if not all_results:
        logger.warning("No results were generated.")
        return pd.DataFrame() # Return empty DataFrame

    try:
        results_df = pd.DataFrame(all_results)
        logger.info(f"Created results DataFrame with shape: {results_df.shape}")
    except Exception as e:
        logger.error(f"Failed to create DataFrame from results: {e}", exc_info=True)
        print(f"Error: Could not convert results to DataFrame: {e}", file=sys.stderr)
        return pd.DataFrame() # Return empty DataFrame on error

    return results_df

########################### MAIN IMPLEMENTATION LOOP ##########################
if __name__ == "__main__":
    # --- Argument Parsing ---
    # Modify parser to accept input file instead of single question
    parser = argparse.ArgumentParser(description="Run RAG Comparison without Streamlit UI from a TSV file.")

    parser.add_argument(
        "-i", "--input-tsv",
        type=str,
        default="dataset.tsv", # Default to dataset.tsv
        help="Path to the input TSV file containing questions (expected column: 'query'). Default: dataset.tsv"
    )
    
    # Model Configs (Simplified - use fixed or load from separate config file?)
    # Using fixed configs for demonstration
    DEFAULT_LEFT_CONFIG = {"type": "gemini", "name": LEFT_MODEL_NAME, "temperature": DEFAULT_GEMINI_TEMPERATURE}
    #DEFAULT_RIGHT_CONFIG = {"type": "3p_models", "name": "llama3"} # Ensure 'llama3' is pulled in Ollama
    DEFAULT_RIGHT_CONFIG = {"type": "gemini", "name": RIGHT_MODEL_NAME, "temperature": 0.3} # Same as Gemini
    
    # Keep other arguments (RAG, Judge, Output, Project, Location, Datastore etc.)
    parser.add_argument("--use-rag", action="store_true", help="Enable RAG using Vertex AI Search.")
    parser.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"), help="Google Cloud Project ID for Vertex AI/Search.")
    parser.add_argument("--location", type=str, default=os.getenv("LOCATION") or os.getenv("GOOGLE_CLOUD_REGION"))
    parser.add_argument("--datastore-id", type=str, help="Vertex AI Search Data Store ID (required if --use-rag).")
    parser.add_argument("--datastore-region", type=str, help="Vertex AI Search Data Store Region (required if --use-rag).")

    # Judge Config
    parser.add_argument("--judge", action="store_true", help="Enable the Judge Model evaluation.")

    # Output Config
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        default="evaluation_results.csv", # Save as CSV by default
        help="Path to save the results DataFrame (e.g., results.csv or results.json)."
    )

    args = parser.parse_args() # Parse arguments from command line
    
    # --- Validate Args (Keep RAG validation) ---
    if args.use_rag and not args.datastore_id:
        parser.error("--datastore-id is required when --use-rag is specified.")
    if args.use_rag and (not args.project_id or not args.location):
         parser.error("--project-id and --location are required when --use-rag is specified (or set corresponding env vars).")

    # --- Load Prompts (Keep existing logic) ---
    system_instruction = None
    judge_system_instruction = None
    judge_model_name = None
    try:
        sys_prompt_path = os.path.join(PROMPT_FOLDER, SYSTEM_PROMPT_FILE)
        system_instruction = load_text_file(sys_prompt_path)
        logger.info("Loaded system instruction.")

        if args.judge:
            judge_prompt_path = os.path.join(PROMPT_FOLDER, JUDGE_PROMPT_FILE)
            judge_system_instruction = load_text_file(judge_prompt_path)
            logger.info("Loaded judge system instruction.")
            judge_model_name_path = os.path.join(PROMPT_FOLDER, JUDGE_MODEL_NAME_FILE)
            judge_model_name = load_text_file(judge_model_name_path).strip() # Clean newline
            logger.info(f"Loaded judge model name: {judge_model_name}")

    except Exception as e:
        logger.error(f"Failed to load required prompt files from '{PROMPT_FOLDER}': {e}", exc_info=True)
        print(f"Error: Could not load necessary prompt files. Exiting.", file=sys.stderr)
        sys.exit(1)


    # --- Load Questions from TSV --- ## MODIFIED SECTION ##
    questions_to_process = []
    input_file_path = args.input_tsv
    logger.info(f"Attempting to load questions from input file: {input_file_path}")

    if not os.path.exists(input_file_path):
        logger.error(f"Input TSV file not found at: {input_file_path}")
        print(f"Error: Input TSV file not found at '{input_file_path}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    try:
        input_df = pd.read_csv(input_file_path, sep='\t') # Read TSV
        logger.info(f"Read input file {input_file_path}, shape: {input_df.shape}")

        # Check if 'query' column exists
        if 'query' not in input_df.columns:
            logger.error(f"Mandatory column 'query' not found in {input_file_path}")
            print(f"Error: Column 'query' not found in '{input_file_path}'. Exiting.", file=sys.stderr)
            sys.exit(1)

        # Extract non-empty questions from the 'query' column
        # Drop rows where 'query' is NaN or None first
        input_df.dropna(subset=['query'], inplace=True)
        # Filter out empty strings after stripping whitespace
        queries_series = input_df['query'].astype(str).str.strip()
        questions_to_process = queries_series[queries_series != ''].tolist()

        if not questions_to_process:
             logger.warning(f"No valid, non-empty questions found in the 'query' column of {input_file_path}.")
             print(f"Warning: No valid questions found in '{input_file_path}'. Nothing to process.", file=sys.stderr)
             sys.exit(0) # Exit gracefully if no questions found

        logger.info(f"Loaded {len(questions_to_process)} questions to process from '{input_file_path}'.")

    except pd.errors.EmptyDataError:
        logger.error(f"Input TSV file '{input_file_path}' is empty.")
        print(f"Error: Input TSV file '{input_file_path}' is empty. Exiting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to read or process input TSV file '{input_file_path}': {e}", exc_info=True)
        print(f"Error reading input file '{input_file_path}': {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    ## END OF MODIFIED SECTION ##


    # --- Define Model Configs (Keep existing logic) ---
    # Use default model configs defined above, or load from a file/args if needed
    left_model_config = DEFAULT_LEFT_CONFIG
    right_model_config = DEFAULT_RIGHT_CONFIG
    logger.info(f"Using Left Config: {left_model_config}")
    logger.info(f"Using Right Config: {right_model_config}")


    # --- Run Main Evaluation ---
    logger.info("Starting evaluation run...")
    # Pass the list of questions loaded from the file to the evaluation function
    # The function must process a list of questions internally and return a single DataFrame.
    results_dataframe = run_dual_llm_pipeline( 
        args,
        questions_to_process, # Pass the list loaded from file
        left_model_config,
        right_model_config,
        system_instruction,
        judge_system_instruction,
        judge_model_name,
    )

    # --- Save Results (Keep existing logic) ---
    if not results_dataframe.empty:
        logger.info(f"Saving results DataFrame to {args.output_path}...")
        try:
            # Save based on output file extension
            if args.output_path.lower().endswith(".csv"):
                results_dataframe.to_csv(args.output_path, index=False)
            elif args.output_path.lower().endswith(".json"):
                results_dataframe.to_json(args.output_path, orient="records", indent=2)
            elif args.output_path.lower().endswith(".tsv"):
                results_dataframe.to_csv(args.output_path, sep='\t', index=False)
            else:
                 # Default to CSV if extension is unknown/missing
                logger.warning(f"Unknown output file extension for '{args.output_path}'. Saving as TSV file.")
                results_dataframe.to_csv(args.output_path, index=False, sep='\t')

            print(f"\nResults saved successfully to {args.output_path}")
            # Optionally print head of dataframe
            print("\nResults DataFrame Head:")
            # Use max_rows display option for better console output if needed
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
                print(results_dataframe.head(1))
            # print(results_dataframe.head().to_string()) # Alternative for full width

        except Exception as e:
            logger.error(f"Failed to save results DataFrame: {e}", exc_info=True)
            print(f"Error: Could not save results to {args.output_path}: {e}", file=sys.stderr)
            # Optionally print DF to console as fallback
            print("\nResults DataFrame (print fallback):")
            print(results_dataframe.to_string()) # Print full df on save error
    else:
        logger.warning("No results generated, DataFrame is empty. Nothing saved.")
        print("\nNo results generated.", file=sys.stderr)


    logger.info("Script finished.")
    print("\nScript finished.")
