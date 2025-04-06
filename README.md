# Multi-Step RAG Demo with Vertex AI and Ollama

This repository demonstrates the iterative development of a Retrieval-Augmented Generation (RAG) system. Starting from a basic local RAG setup using FAISS, the demo progressively adds features like advanced prompting, dual LLM comparison (Vertex AI Gemini vs. local Ollama), cloud-based vector search with Vertex AI Search, and finally, an LLM-based judge model for automated response evaluation.

The demo application uses Streamlit for the user interface.

## Features

* Step-by-step implementation showcasing the evolution of a RAG system.
* Supports both local vector stores (FAISS) and cloud-native vector databases (Google Cloud Vertex AI Search).
* Integrates multiple LLMs: Google Gemini (via Vertex AI) and local Ollama models.
* Implements advanced prompting techniques: Query Rephrasing and Response Summarization.
* Optional direct chat mode (non-RAG) for general LLM comparison.
* Vector store caching (FAISS implementation in Step 2).
* LLM-based Judge model (Gemini 1.5 Pro by default) for automated response quality evaluation.
* Interactive Streamlit UI.

## Demo Evolution (Step-by-Step)

The demo is broken down into sequential Python scripts, each adding new functionality:

The demo begins with a fictional use case for an Asian Chef Advisor chatbot for an electronic book publisher. The publisher wants to sell more books. Hence they have tasked you with creating a chatbot that will provide Asian recipes for users coming to the site so that if they like the recipes, they may buy the books to get more of the same.  

1.  **Step 0 (`Asian_Chef_Advisor_Vertex_ES_step0.py`):**
    * **Basic RAG:** Implements a fundamental RAG pipeline.
    * **Local Data:** Reads PDF documents from the local `recipes/` folder.
    * **Local Vector Store:** Uses FAISS to create and query a local vector index.
    * **Simple Chat:** Provides a basic chatbot interface using a single LLM (likely Vertex AI Gemini). Best suited for simple questions related to the indexed documents.

2.  **Step 1 (`Asian_Chef_Advisor_Vertex_ES_step1.py`):**
    * **Enhanced Prompting:** Builds upon Step 0 by adding **Query Rephrasing** and **Response Summarization** prompts. This allows the system to handle more complex, long-winded questions more effectively.

3.  **Step 2 (`Asian_Chef_Advisor_Vertex_ES_step2.py`):**
    * **Dual LLM Comparison:** Introduces a second LLM (**Ollama** running locally) alongside the Vertex AI model, allowing for side-by-side response comparison for the same query.
    * **Direct Chat Mode:** Adds an option to bypass the RAG pipeline and chat directly with the selected LLMs for general knowledge questions.
    * **FAISS Caching:** Implements caching for the FAISS vector store to speed up document retrieval for previously seen queries or contexts.

4.  **Step 3 (`Asian_Chef_Advisor_Vertex_ES_step3.py`):**
    * **Cloud Vector Store:** Upgrades the vector store from local FAISS to **Google Cloud Vertex AI Search**.
    * **GCS Integration:** Requires documents to be uploaded to a Google Cloud Storage (GCS) bucket.
    * **Vertex AI Search Setup:** Needs a Vertex AI Search App/Datastore configured to index the documents in the GCS bucket. The Datastore ID is selected/entered in the UI.
    * **Cloud RAG:** Continues the dual LLM comparison (Gemini + Ollama) but now uses Vertex AI Search for document retrieval.
    * **Prompt Loading:** May load prompts from external files (e.g., a `prompts/` directory - check script for details).

5.  **Step 4 (`Asian_Chef_Advisor_Vertex_ES_step4.py`):**
    * **LLM Judge:** Introduces an LLM-based "Judge" (using Vertex AI Gemini 1.5 Pro by default, configurable in the script).
    * **Automated Evaluation:** The Judge model automatically reads the responses from the two primary LLMs (Gemini and Ollama) based on the retrieved context and provides a verdict on which answer is better and why.

## File Structure

* `Asian_Chef_Advisor_Vertex_ES_step[0-4].py`: Python scripts for each stage of the Streamlit demo application.
* `recipes/`: Directory containing sample PDF documents used for RAG (Steps 0-2) and to be uploaded to GCS (Steps 3-4). Contains 4 sample docs.
* `faiss_index/`: Directory where the local FAISS vector store index is saved by default (Steps 0-2). *(Recommendation: Add this directory to your `.gitignore` file)*.
* `requirements.txt`: Lists necessary Python packages for installation.
* `system_instruction.txt`: Contains base system prompts or instructions for the LLMs.
* `prompts/` (Optional/Implied): May contain specific prompt templates used in Steps 3 & 4. Check the scripts for details.

## Prerequisites

* Python 3.8+
* pip (Python package installer)
* Git (for cloning the repository)
* **Google Cloud Account:** With billing enabled.
* **Google Cloud Project:** With the following APIs enabled:
    * Vertex AI API
    * Vertex AI Search API (for Steps 3 & 4)
* **Google Cloud Storage (GCS):** A bucket to store your documents for Vertex AI Search (Steps 3 & 4).
* **Vertex AI Search:** A configured App/Datastore linked to your GCS bucket (Steps 3 & 4). Make note of your **Datastore ID**.
* **Ollama:** (Optional, for Steps 2, 3, 4)
    * Install Ollama locally by following the instructions at [https://ollama.com/](https://ollama.com/).
    * Pull the desired models (e.g., `ollama pull llama3`, `ollama pull mistral`).
    * Ensure the Ollama service is running when executing Steps 2, 3, or 4.
* **Google Cloud SDK (`gcloud`):** Installed and configured for authentication.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # OR
    # .venv\Scripts\activate  # Windows
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Authenticate with Google Cloud:**
    ```bash
    gcloud auth application-default login
    ```
5.  **Configure Google Cloud Project:**
    ```bash
    gcloud config set project YOUR_PROJECT_ID
    ```
    Replace `YOUR_PROJECT_ID` with your actual Google Cloud project ID.
6.  **(Steps 3 & 4 Only) Prepare Cloud Resources:**
    * Create a GCS bucket in your project.
    * Upload the documents from the `recipes/` directory (or your own documents) to the GCS bucket.
    * Go to the Vertex AI Search console in Google Cloud and create a new Search App/Datastore. Configure it to index the data from your GCS bucket. Note the **Datastore ID**.
7.  **(Steps 2, 3, 4 Only) Prepare Ollama:**
    * Ensure Ollama is installed and the service is running.
    * Pull the models you intend to use (check the Python scripts for defaults, e.g., `ollama pull llama3`).

## Configuration

* **Vertex AI Search (Steps 3/4):** When running `step3.py` or `step4.py`, the Streamlit UI will likely prompt you to select or enter your Vertex AI Search **Datastore ID**. Ensure the credentials used (via `gcloud auth application-default login`) have the necessary permissions (e.g., Vertex AI User, Storage Object Viewer).
* **Prompts:** Review `system_instruction.txt` and check Steps 3 & 4 for how specific prompts (Rephraser, Summarizer, Judge) are defined or loaded. Modify them as needed.
* **Ollama Model:** The specific Ollama model used might be hardcoded or selectable in the UI for Steps 2, 3, and 4. Ensure the selected model name matches one you have pulled via `ollama pull`.
* **Judge Model (Step 4):** Defaults to Gemini 1.5 Pro via Vertex AI. You can modify the model name directly within `Asian_Chef_Advisor_Vertex_ES_step4.py` if needed.

## Running the Demo

Make sure you are in the repository's root directory and your virtual environment is activated.

Execute each step using Streamlit:

## Basic local RAG with local FAISS store and a simple prompt
```bash
streamlit run Asian_Chef_Advisor_Vertex_ES_step0.py
```

## Adds Rephraser/Summarizer prompts
```bash
streamlit run Asian_Chef_Advisor_Vertex_ES_step1.py
```

## Adds Ollama comparison, Direct Chat, FAISS Caching (Ensure Ollama is running if you need Ollama models)
```bash
streamlit run Asian_Chef_Advisor_Vertex_ES_step2.py
```

## Switches RAG to Vertex AI Search RAG (Ensure Ollama running, Vertex AI Search setup complete)
```bash
streamlit run Asian_Chef_Advisor_Vertex_ES_step3.py
```

## Adds Judge Model (Ensure Ollama running, Vertex AI Search setup complete)
```bash
streamlit run Asian_Chef_Advisor_Vertex_ES_step4.py
```
## License
Apache 2.0 License

## Copyright: Ram Seshadri (2025)



