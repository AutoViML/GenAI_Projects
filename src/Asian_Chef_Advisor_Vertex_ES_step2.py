##############################################################################
#############       This is not an Officially Supported Google Product! ######
##############################################################################
#Copyright 2025 Google LLC                                              ######
#                                                                       ######
#Licensed under the Apache License, Version 2.0 (the "License");        ######
#you may not use this file except in compliance with the License.       ######
#You may obtain a copy of the License at                                ######
#                                                                       ######
#    https://www.apache.org/licenses/LICENSE-2.0                        ######
#                                                                       ######
#Unless required by applicable law or agreed to in writing, software    ######
#distributed under the License is distributed on an "AS IS" BASIS,      ######
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.#####
#See the License for the specific language governing permissions and    ######
#limitations under the License.                                         ######
##############################################################################
###########           Vertex RAG Comparator with Judge Model            ######
###########             Developed by Ram Seshadri                       ######
###########             Last Updated:  Feb 2025                         ######
##############################################################################
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import time
from google import genai
from google.genai import types

# --- Configuration ---
DEBUG = False  # Set to False in production

# --- Helper Functions ---
def log_debug(message):
    if DEBUG:
        st.sidebar.write(f"DEBUG: {message}")

# --- App Setup ---
st.set_page_config("Demo 2: Vertex RAG Compare 2 Models", layout="wide")
st.title("📄 Demo 2: Vertex RAG Compare 2 Models")

def load_system_instruction():
    try:
        with open("./prompts/system_instruction.txt", "r") as file:
            system_instruction = file.read()
    except FileNotFoundError:
        print("Error: system_instruction.txt not found.")
    except IOError:
        print("Error: Could not read system_instruction.txt.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return system_instruction

def create_gemini_client():
    """Create client with automatic API selection and auth handling"""
    if os.environ.get("GOOGLE_VERTEXAI", "").lower() == "true":
        # Vertex AI configuration
        project = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION")
       
        if not project or not location:
            raise ValueError("VertexAI requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables")
       
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            #http_options=types.HttpOptions(api_version='v1')
        )
    else:
        # Gemini Developer API configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API requires GOOGLE_API_KEY environment variable")
       
        return genai.Client(
            api_key=api_key,
            #http_options=types.HttpOptions(api_version='v1alpha')
        )

def get_rephraser_prompt(query):

    # Setup the Query Rephraser Prompt here 
    prompt = f"""
    "You are a search query rephraser for a chef. A customer will provide a lengthy question or request for a dish. 
    Your mission is to rephrase their input into a concise, 20-word or less query suitable for a Google-like search engine. 
    Focus on the customer's core ingredients and tastes. Do not include anything else.
    
    Example:
    
    Customer Input: 'I'm trying to find information about the best way to use the huge number of apples from my apple trees in the late winter to ensure that I use them all to create a healthy set of different jams and pickles for my family and I'm particularly interested that do not damage their teeth with excess sugar and promote healthy growth in their young minds. This will be great for Xmas. Can you suggest a recipe?'
    
    Chatbot Output: 'Apple jams and or pickles with less sugar.'
    
    Now, please rephrase the following customer query:"
    
    {query}
    """
    # Assuming system_instruction exists
    if st.session_state.system_prompt:
        system_instruction = st.session_state.system_prompt
        rephraser_prompt = f"{system_instruction}\n{prompt}"
    else:
        rephraser_prompt = prompt

    return rephraser_prompt

def get_summarizer_prompt(context, question):
    
    prompt = f"""
    You have been provided with a customer's search query and the text content of several documents retrieved from a recipe database. Your task is to:
    
    Analyze the Search Query: Understand the core ingredients or cooking techniques the customer is asking about.
    Evaluate Document Relevance: Carefully read each provided document and determine its relevance to the customer's search query.
    Select the Best Dish: Based on the analysis, identify the one dish that best matches the customer's request.
    Provide Recipes: From the relevant documents, extract and present one or two clear and concise recipes for the selected dish.
    
    Question: \n{question}\n
    
    Document Texts:
    Context:\n {context}?\n
    
    Output Format:
    
    Best Dish: [Dish Name]
    
    Recipe 1:
    
    Ingredients: [List ingredients]
    Instructions: [Step-by-step instructions]
    Recipe 2 (Optional):
    
    Ingredients: [List ingredients]
    Instructions: [Step-by-step instructions]
    """
    # Assuming system_instruction exists
    if st.session_state.system_prompt:
        system_instruction = st.session_state.system_prompt 
        summarizer_prompt = f"{system_instruction}\n{prompt}"
    else:
        summarizer_prompt = prompt
    
    return summarizer_prompt

def initialize_models():
    """Initializes Gemini and Ollama models and returns their lists."""
    try:        
        client = create_gemini_client()
        response=client.models.list(config={'page_size': 100, 'query_base': True})
        # Initialize Gemini        
        gemini_models = []
        for i, _ in enumerate(response.page):
            if 'gemini-2.0' in response.page[i].name or 'gemini-1.5' in response.page[i].name:
                gemini_models.append(response.page[i].name.split("/")[-1])
        #print(gemini_models)

        # Initialize Ollama
        ollama_models = []
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                ollama_models = [model['name'] for model in response.json()['models']]
                #log_debug(f"Found Ollama models: {ollama_models}")
            else:
                st.error("🔴 Ollama server not responding - make sure it's running!")
        except requests.exceptions.RequestException as e:
            st.error(f"Ollama connection failed: {str(e)}")

        return ollama_models, gemini_models  # Return both lists

    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return [], []

# --- Document Processing ---
def process_documents(pdf_docs):
    try:
        with st.status("📄 Processing documents...", expanded=True) as status:
            # Extract text
            st.write("Extracting text from PDFs...")
            raw_text = "".join([page.extract_text() for pdf in pdf_docs 
                              for page in PdfReader(pdf).pages])
            
            # Split text
            st.write("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            text_chunks = text_splitter.split_text(raw_text)
            
            # Create vector store
            st.write("Creating search index...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            
            status.update(label="✅ Document processing complete!", state="complete")
            return True
    except Exception as e:
        st.error(f"Document processing failed: {str(e)}")
        return False


@st.cache_data
def get_relevant_docs(question, _vector_store):
    """Retrieves and caches relevant documents from the vector store."""
    log_debug("Retrieving relevant documents from vector store")
    docs = _vector_store.similarity_search(question)
    log_debug(f"Found {len(docs)} relevant document chunks")
    context = "\n".join([doc.page_content for doc in docs])
    log_debug(f"Generated context (first 100 chars): {context[:100]}...")
    return context

def generate_gemini_response(prompt, model_config, col): # Added model config!
    """Generates a response using the Gemini Chat API and writes to the Streamlit column."""
    log_debug(f"Calling Gemini Chat API directly - model={model_config['name']}, temp={model_config['temperature']}") #Model config
    model_name = model_config['name'] ## get name of model
    temperature = model_config["temperature"] # get temp from config
    # Create a unique session key for each column's model
    session_key = f"gemini_chat_model_{col}"  # Create a unique session key based on each column
    full_response = ""
    
    if st.session_state.system_prompt.strip():
        system_instruction = st.session_state.system_prompt.strip()
    generate_content_config = types.GenerateContentConfig(
        temperature = temperature,
        top_p = 0.95,
        max_output_tokens = 2048,
        response_modalities = ["TEXT"],
        system_instruction=[types.Part.from_text(text=system_instruction)],
      )
    
    ### You need to reload the genai if it is not already in session ##
    ### If the model is in memory all the time (no need to configure each time)
    if session_key not in st.session_state:
        log_debug(f'Setting genai model to {model_config["name"]}')
        client = create_gemini_client()
        
        st.session_state[session_key] = client #Set the session model object

    else: #Retrieve the model from column instead.
        client = st.session_state[session_key]
    
    log_debug(f"full prompt:\n{prompt}")

    with col: #Added col so it all belongs inside streamlit!
        with st.chat_message("ai"):
            response = client.models.generate_content_stream(
                model = model_name,
                contents = prompt,
                config = generate_content_config,
                )
            
            for chunk in response:
                chunk_text = chunk.text
                full_response += chunk_text
            st.write(full_response) # Directly writes the response
        #log_debug(f"Gemini Response Status: {response.prompt_feedback}") ## getting empty string
        #log_debug(f'Usage meta data: token count = {usage_metadata.total_token_count}') ## getting error

    return full_response

def generate_ollama_response(prompt, model_config, col):
    """Generates a streaming response using the Ollama Chat API and writes to Streamlit column."""
    log_debug("Calling Ollama API directly using streaming")
    full_response = ""
    model_name = model_config["name"] # get name of model
    with col: #Added col so it all belongs inside streamlit!
        with st.chat_message("ai"):  # Streamlit messaging
            chat_placeholder = st.empty()  # For incremental updates

            stream = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': prompt}
            ], stream=True)

            for chunk in stream:
                if 'content' in chunk['message']:
                    chunk_text = chunk['message']['content']
                    full_response += chunk_text
                    #chat_placeholder.write(chunk_text + "▌")
                else:
                    print("Skipping chunk without data content")

            chat_placeholder.write(full_response)

    return full_response

def generate_response(model_config, question, context, col):
    """
    Orchestrates response generation using either Gemini or Ollama, calling the models directly.
    """
    start_time = time.time()
    try:
        log_debug(f"Generating rephrased query with config: {model_config}")

        rephraser_prompt = get_rephraser_prompt(question)
        
        if model_config["type"] == "gemini":
            log_debug(f"Using Gemini: model={model_config['name']}, temp={model_config['temperature']}")
            rephrased_query = generate_gemini_response(rephraser_prompt, model_config, col)
        elif model_config["type"] == "ollama":
            log_debug(f"Using Ollama: model={model_config['name']}")
            rephrased_query = generate_ollama_response(rephraser_prompt, model_config, col)
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")

        log_debug(f"Gemini Rephrased Query:\n {rephrased_query}")

        summarizer_prompt = get_summarizer_prompt(context=context, question=rephrased_query)
        
        if model_config["type"] == "gemini":
            log_debug(f"Using Gemini: model={model_config['name']}, temp={model_config['temperature']}")
            response_text = generate_gemini_response(summarizer_prompt, model_config, col)
        elif model_config["type"] == "ollama":
            log_debug(f"Using Ollama: model={model_config['name']}")
            response_text = generate_ollama_response(summarizer_prompt, model_config, col)
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")

        log_debug(f"Response:\n {response_text}")
        return {
            "text": response_text,
            "time": time.time() - start_time,
            "error": None
        }

    except Exception as e:
        log_debug(f"Error generating response: {str(e)}")
        return {
            "text": None,
            "time": 0,
            "error": str(e)
        }

def model_selection(column, key_prefix):
    """Streamlit app for chatbot with multiple models, including Gemini model selection."""
    with column:
        model_type = st.radio(
            "Model Type",
            ["Gemini", "Ollama"],
            index=0 if key_prefix == "left" else 1,
            key=f"{key_prefix}_type"
        )

        config = {"type": model_type.lower()}


        if key_prefix == "right" and model_type.lower() == 'ollama':
            ### let's make this Ollama models
            model_name = st.selectbox(
                "Select Model",
                st.session_state.ollama_models,
                key=f"{key_prefix}_ollama_model" # NEW Distinct key

            )
            config["name"] = model_name
            config["temperature"] = 0.0  # No temperature in Ollama, but needs to be there
        else:  # Gemini
            model_name = st.selectbox(
                "Select Model",
                st.session_state.gemini_models,
                key=f"{key_prefix}_gemini_model",  # NEW Distinct key
                index=0  # Select the first as a default
            )
            config["name"] = model_name  # store model name
            temperature = st.slider(
                "Temperature",
                0.0, 1.0, 0.3,
                key=f"{key_prefix}_temp"
            )
            config["temperature"] = temperature

        #st.write("key prefix: ", key_prefix)
        #st.write("model name: ", model_name)
        return config

##################### MAIN APP BELOW ##################################
def main():
    # --- Main App Flow ---
    if 'ollama_models' not in st.session_state:
        st.session_state.ollama_models, st.session_state.gemini_models = initialize_models()

    # --- Initialization ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.system_prompt = load_system_instruction() 

    # --- Document Upload --- 
    with st.sidebar:
        st.header("📁 Document Setup")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        #pdf_docs = ['./recipes/asian_recipes1.pdf','./recipes/asian_recipes2.pdf','./recipes/asian_recipes3.pdf', './recipes/asian_recipes4.pdf']
        if st.button("Process Documents"):
            if pdf_docs:
                if process_documents(pdf_docs):
                    st.success("Documents ready for analysis!")
            else:
                st.warning("Please upload PDF files first!")

    # --- Response Columns ---
    left_col, right_col = st.columns(2)

    # --- Model Selection ---
    left_config = model_selection(left_col, "left")
    right_config = model_selection(right_col, "right")

    #configure generative api
    if left_config['type'] == "gemini" or right_config['type'] == "gemini":
        pass
        
    ### Create a new checkbox to see if we need RAG at all ###
    with st.sidebar:
        use_rag = st.checkbox("Use RAG for chat (check mark if needed)", value=False)  # set default to False

    # --- Modified Main Interaction Section ---
    user_question = st.chat_input("Ask your question:")

    if user_question:  # Moved logic into the if statement
        # Get document context (conditionally)
        if use_rag and "vector_store" in st.session_state:  # if check box is ticked otherwise it will still run it with "Not vector store"
            context = get_relevant_docs(user_question, st.session_state.vector_store)
            log_debug(f"Found relevant context: {context[:100]}...")
        else:
            context = ""  # Provide empty context if RAG is skipped
            log_debug("Skipping vector store and use rag. Providing empty context to model")

        # Display the user's question
        with left_col:
            with st.chat_message("user"):
                st.write(user_question)
        with right_col:
            with st.chat_message("user"):
                st.write(user_question)


        # Generate responses: now you pass it to the Streamlit col correctly.
        left_response = generate_response(left_config, user_question, context, left_col)
        right_response = generate_response(right_config, user_question, context, right_col)

        # Add to chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "left": left_response,
            "right": right_response,
            "timestamp": time.time()
        })

if __name__ == "__main__":
    main()
