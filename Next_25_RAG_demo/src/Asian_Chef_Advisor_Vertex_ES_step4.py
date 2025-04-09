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
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine
from google.cloud import aiplatform
###### Now we set up langchain to talk to Vertex Search
from langchain.chains import (
    ConversationalRetrievalChain,
    RetrievalQA,
    RetrievalQAWithSourcesChain,
)
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_community import (
    VertexAIMultiTurnSearchRetriever,
    VertexAISearchRetriever,
)
from langchain_google_vertexai import VertexAI

import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
################################################################################

# --- Configuration ---
DEBUG = False  # Set to False in production

# --- Helper Functions ---
def log_debug(message):
    if DEBUG:
        st.sidebar.write(f"DEBUG: {message}")

# --- App Setup ---
st.set_page_config("Demo 4: Vertex RAG with Judge Model", layout="wide")
st.title("📄 Demo 4: Vertex RAG with Judge Model")

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
        st.session_state["project_id"] = project
        st.session_state["location"] = location
        
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

    If the question appears to be a follow-up to a previous question or a response from the AI bot, then return the question as is. You can add the tag <No RAG required> to the output as follows:
    
    Example:
    Customer Input: 'Can you print the list of ingredients again?'
    
    Chatbot Output: 'Can you print the list of ingredients again? ++No RAG required++'
    
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

def get_summarizer_prompt(documents, query):
    
    Text_of_Document_1, Text_of_Document_2, Text_of_Document_3 = documents[0], documents[1], documents[2] 
    
    ##### Now summarize the documents and present a final answer here #######
    prompt = f"""
    You are an expert Asian cuisine advisor. You have been provided with a customer's search query and the text content of several documents retrieved from a recipe database. Your task is to:
    
    Analyze the Search Query: Understand the core ingredients or cooking techniques the customer is asking about.
    Evaluate Document Relevance: Carefully read each provided document and determine its relevance to the customer's search query.
    Select the Best Dish: Based on the analysis, identify the one dish that best matches the customer's request.
    Provide Recipes: From the relevant documents, extract and present one or two clear and concise recipes for the selected dish.
    
    Customer Search Query: {query}
    
    Document Texts:
    [Start of Document 1]
    {Text_of_Document_1}
    [End of Document 1]
    
    [Start of Document 2]
    {Text_of_Document_2}
    [End of Document 2]
    
    [Start of Document 3]
    {Text_of_Document_3}
    [End of Document 3]
    
    Output Format:
        
    Recipe 1: [Dish Name]
    
    Ingredients: each ingredient must be listed in a new line like below
    ingredient 1
    ingredient 2
    etc.
    
    Instructions: [Step-by-step instructions]
    
    Recipe 2 (Optional): [Dish Name]
    
    Ingredients: each ingredient must be listed in a new line like below
    ingredient 1
    ingredient 2
    etc.

    Instructions: [Step-by-step instructions]
    
    If the query pertains to a previous conversation or a general question about Asian cooking, you can answer it without looking at the context.
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
        
    ### use this for a text model ###
    generate_content_config = types.GenerateContentConfig(
        temperature = temperature,
        top_p = 0.95,
        max_output_tokens = 2048,
        response_modalities = ["TEXT"],
        system_instruction=[types.Part.from_text(text=system_instruction)],
      )
    
    ### Use this only for a chat model ####
    chat_content_config = {
        "system_instruction":system_instruction,
        "temperature": 0.3,
        "max_output_tokens": 2048,
    }
    
    ### You need to reload the genai if it is not already in session ##
    ### If the model is in memory all the time (no need to configure each time)
    if session_key not in st.session_state:
        log_debug(f'Setting genai model to {model_config["name"]}')
        client = create_gemini_client()
        
        ### Use this if you are using a chat model. Otherwise, comment it out.
        client = client.chats.create(model=model_name, config=chat_content_config)
        
        st.session_state[session_key] = client #Set the session model object

    else: #Retrieve the model from column instead.
        client = st.session_state[session_key]
    
    log_debug(f"full prompt:\n{prompt}")

    with col: #Added col so it all belongs inside streamlit!
        with st.chat_message("ai"):
            ### Use this only for a text model - not a chat model! 
            #response = client.models.generate_content_stream(
            #    model = model_name,
            #    contents = prompt,
            #    config = generate_content_config,
            #    )
            
            #### This is a chat model ######
            response = client.send_message_stream(prompt)
            
            ### Both chat and text work the same way in extracting text ##
            for chunk in response:
                chunk_text = chunk.text
                full_response += chunk_text
            st.write(full_response.replace("++No RAG required++","")) # Remove RAG related text
            ### this is the input token count is the prompt token count
            st.write(f'\t\tinput token count = {chunk.usage_metadata.prompt_token_count}')
            # Directly provides output count
            st.write(f'\t\toutput token count = {chunk.usage_metadata.candidates_token_count}')
        log_debug(f"Gemini Response: {full_response}") ## getting empty string
        log_debug(f'Usage meta data: total token count = {chunk.usage_metadata.total_token_count}') ## getting error

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

def generate_rephraser_response(model_config, question, col):
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
        log_debug("Time taken: %0.1f seconds" %(time.time() - start_time))
        
        return rephrased_query
    except Exception as e:
        log_debug(f"Error generating response: {str(e)}")
        return question
    

def generate_summarizer_response(model_config, question, context, col):
    """

    Orchestrates response generation using either Gemini or Ollama, calling the models directly.
    """
    start_time = time.time()
    try:
        if isinstance(context, str):
            summarizer_prompt = "Given this context:\n" + context + "\nAnswer the question below to the best of your knowledge.\n" + question
        else:
            summarizer_prompt = get_summarizer_prompt(context, question)
        
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
            "text": "",
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

def list_data_stores(
    project_id: str,
    location: str,
) -> discoveryengine.ListDataStoresResponse:
    #  this is the way to setup a client
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.DataStoreServiceClient(client_options=client_options)

    request = discoveryengine.ListDataStoresRequest(
        # The full resource name of the data store
        parent=client.collection_path(
            project_id, location, collection="default_collection"
        )
    )

    # Make the request
    response = client.list_data_stores(request=request)

    for i, data_store in enumerate(response):
        #if i == 0:
        #    print(data_store)
        pass

    return [x.name.split("/")[-1] for x in response]


#### list the documents in a datastore
from typing import Any
def list_documents(project_id: str, location: str, data_store_id: str) -> Any:
    #  For setting up a connection to datastore
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.DocumentServiceClient(client_options=client_options)

    # The full resource name of the search engine branch.
    # e.g. projects/{project}/locations/{location}/dataStores/{data_store_id}/branches/{branch}
    parent = client.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch="default_branch",
    )

    response = client.list_documents(parent=parent)
    ls = []
    for result in response:
        ls.append(result)

    return ls

def get_retriever():
    # # Setup Vertex Search
    ### Vertex Search datastore ID
    PROJECT_ID = st.session_state["project_id"]
    DATA_STORE_LOCATION = "global"  # @param {type:"string"}
    
    st.session_state["datastore_ids"] = list_data_stores(PROJECT_ID, DATA_STORE_LOCATION) 
    
    datastore_id = st.selectbox(
        "Select Vertex Data Store",
        st.session_state.datastore_ids,
        key=f"datastore_id",  # NEW Distinct key
        index=0  # Select the first as a default
        )
    
    if st.button("Connect"):
        ls = list_documents(PROJECT_ID, DATA_STORE_LOCATION, datastore_id)
        if len(ls) > 0:
            st.success(f"Connected! There are {len(ls)} documents in datastore.")
        else:
            st.warning("Please select a correct data store first!")

    # Define the LLM that will be used in vertex ai search
    MODEL = "gemini-2.0-flash-001" 
    
    # Call Vertex Search here #######
    llm = VertexAI(model_name=MODEL)
    
    retriever = VertexAISearchRetriever(
        project_id=PROJECT_ID,
        location_id=DATA_STORE_LOCATION,
        data_store_id=datastore_id,
        get_extractive_answers=False,
        max_documents=1,
        max_extractive_segment_count=5,
        max_extractive_answer_count=5,
    )
    st.session_state.vector_store = retriever
    return llm

def get_relevant_docs(search_query, llm):
    # get the retriever from session state
    retriever = st.session_state.vector_store
    
    # Retrieve answer from Vertex Search
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    
    results = retrieval_qa.invoke(search_query)
    
    #print("*" * 79)
    #print(results["result"])
    #print("*" * 79)
    
    documents = []
    for i, doc in enumerate(results["source_documents"]):
        ls = f"{'-'*30} Document {i+1} {'-'*30}"
        #print(ls)
        doc_content = doc.page_content
        documents.append(ls+doc_content)
        #print(doc_content)
    
    return documents


def judge_responses(left_question, left_response, left_context, right_question, right_response, right_context):
    """
    Judges the responses from two models (left and right) using a hardcoded Gemini model.

    Args:

        question (str): The user's question.
        left_response (dict): The response from the left model (including text).
        right_response (dict): The response from the right model (including text).


    Returns:
        str: The judgment from the Gemini model indicating which response is better.
    """
    try:

        # Construct the prompt for the judge model
        judge_prompt = f"""Given the following QUESTION and the CONTEXT which is the source of truth to use, judge each model's response:
        
        Here are two responses from different language models:

        Response from model on the left:
        QUESTION:
        {left_question}
        
        CONTEXT:
        {left_context}

        Response A (Model on the Left):
        {left_response['text']}
        
        Response from model on the right:
        QUESTION:
        {right_question}
        
        CONTEXT:
        {right_context}

        Response B (Model on the Right):
        {right_response['text']}

        Which one more accurately responds to the question using the source of truth? Make sure your verdict is based on each model's strict adherence to the source of truth.
        """
        # Initialize the Gemini model for judging (hardcoded)
        model_name = 'gemini-2.0-flash-001'
        #model_name = 'gemini-2.0-flash-thinking-exp-01-21'
        
        #si_text1 = "Your mission is to judge the responses from two models (left and right) for a given question."
        si_text1 = """You are an AI Bot who acts as a judge to analyze two responses to the same question. 
        Please analyze both responses and provide a clear, concise judgment
        indicating which response is better and why. Your judgment should

        clearly state either "Response A is better" or "Response B is better",
        followed by a brief explanation. If both responses are equal in quality,
        state 'Responses A and B are equal'"""
        
        client = create_gemini_client()
        generate_content_config = types.GenerateContentConfig(
            temperature = 0.5,
            top_p = 0.95,
            max_output_tokens = 2048,
            response_modalities = ["TEXT"],
            system_instruction=[types.Part.from_text(text=si_text1)],
          )
        response = client.models.generate_content_stream(
            model = model_name,
            contents = judge_prompt,
            config = generate_content_config,
            )
        log_debug("Judge response: %s " %response)
        return response

    except Exception as e:
        log_debug(f"Error in judge_responses: {e}")
        return f"Error during judgment: {e}"
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
        st.header("📁 Document Store connection")
        llm = get_retriever()
        

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
        # Display the user's question
        with left_col:
            with st.chat_message("user"):
                st.write(user_question)
        with right_col:
            with st.chat_message("user"):
                st.write(user_question)

        # Generate responses: now you pass it to the Streamlit col correctly.
        left_rephrased = generate_rephraser_response(left_config, user_question, left_col)
        right_rephrased = generate_rephraser_response(right_config, user_question, right_col)


        # Get document context (conditionally)
        if use_rag and "vector_store" in st.session_state:  # if check box is ticked otherwise it will still run it with "Not vector store"
            ls = ["I am not able to answer this question" ,"++No RAG required++"]
            if sum([x in left_rephrased for x in ls]) and len(st.session_state.chat_history)>0:
            #if "++No RAG required++" in left_rephrased and len(st.session_state.chat_history)>0:
                left_context = st.session_state.chat_history[-1]["left"]["text"]
                log_debug("Skipping document RAG for left model since it is a continuation question")
                log_debug(f"\ttype of context...{type(left_context)}")
                log_debug(f"\tProviding chat history for left model...{left_context}")
                left_rephrased = user_question
            else:
                left_context = get_relevant_docs(left_rephrased, llm)
                log_debug(f"Found relevant context for left model: {left_context[:100]}...")

            ### Double check if there is a followup to the previous question 
            if sum([x in right_rephrased for x in ls]) and len(st.session_state.chat_history)>0:
            #if "++No RAG required++" in right_rephrased and len(st.session_state.chat_history)>0:
                right_context = st.session_state.chat_history[-1]["right"]["text"]
                log_debug("Skipping document RAG for right model since it is a continuation question")
                log_debug(f"\ttype of context...{type(right_context)}")
                log_debug(f"\tProviding chat history for right model...{right_context}")
                right_rephrased = user_question
            else:
                right_context = get_relevant_docs(right_rephrased, llm)
                log_debug(f"Found relevant context for right model: {right_context[:100]}...")
            
        else:
            left_context = ""  # Provide empty context if RAG is skipped
            right_context = ""  # Provide empty context if RAG is skipped
            log_debug("Skipping document store and use rag. Providing empty context to model")

        # Generate responses: now you pass it to the Streamlit col correctly.
        left_response = generate_summarizer_response(left_config, left_rephrased, left_context, left_col)
        right_response = generate_summarizer_response(right_config, right_rephrased, right_context, right_col)

        # Add to chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "left_question": left_rephrased,
            "right_question": right_rephrased,
            "left": left_response,
            "right": right_response,
            "timestamp": time.time()
        })

        ## wait a couple of seconds to finish streaming
        time.sleep(1)
        
        # Generate the Judgment
        judgement = judge_responses(left_rephrased, left_response, left_context, right_rephrased, right_response, right_context) # function calls that get the judge response!

        # Display Chat history at the initialization, this part should remain before printing chat
        full_response = ""
        with st.container():
            st.subheader("Judge's Analysis")
            #st.write(left_response)
            for chunk in judgement:
                full_response += chunk.text
            st.write(full_response) # add that line to print it to screen!            
##################### MAIN APP COMPLETE ##################################

if __name__ == "__main__":
    main()
