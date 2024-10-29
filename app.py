import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import torch
import torch.nn.functional as F
import re
import requests
from dotenv import load_dotenv
from embedding_processor import SentenceTransformerRetriever, process_data
import pickle

import os
import warnings
import json  # Add this import



# Load environment variables
load_dotenv()

# Add the new function here, right after imports and before API configuration
@st.cache_data
@st.cache_data
def load_from_drive(file_id: str):
    """Load pickle file directly from Google Drive"""
    try:
        # Direct download URL for Google Drive
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        # First request to get the confirmation token
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check if we need to confirm download
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                # Add confirmation parameter to the URL
                url = f"{url}&confirm={value}"
                response = session.get(url, stream=True)
                break
        
        # Load the content and convert to pickle
        content = response.content
        print(f"Successfully downloaded {len(content)} bytes")
        return pickle.loads(content)
        
    except Exception as e:
        print(f"Detailed error: {str(e)}")  # This will help debug
        st.error(f"Error loading file from Drive: {str(e)}")
        return None

# Hugging Face API configuration
#HUGGINGFACE_API_KEY = "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"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
#headers = {"Authorization": "Bearer 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"}

class RAGPipeline:

    def __init__(self, data_folder: str, k: int = 3):  # Reduced k for faster retrieval
        self.data_folder = data_folder
        self.k = k
        self.retriever = SentenceTransformerRetriever()
        cache_data = process_data(data_folder)
        self.documents = cache_data['documents']
        self.retriever.store_embeddings(cache_data['embeddings'])
    

    # Alternative API call with streaming
    def query_model(self, payload):
        """Query the Hugging Face API with streaming"""
        try:
            # Add streaming parameters
            payload["parameters"]["stream"] = True
            
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            # Collect the entire response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if isinstance(json_response, list) and len(json_response) > 0:
                            chunk_text = json_response[0].get('generated_text', '')
                            if chunk_text:
                                full_response += chunk_text
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue
            
            return [{"generated_text": full_response}]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            raise
        
    def preprocess_query(self, query: str) -> str:
        """Clean and prepare the query"""
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        return query

    def postprocess_response(self, response: str) -> str:
        """Clean up the generated response"""
        response = response.strip()
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:\+\d{2}:?\d{2})?', '', response)
        return response


    def process_query(self, query: str, placeholder) -> str:
        try:
            # Preprocess query
            query = self.preprocess_query(query)
            
            # Show retrieval status
            status = placeholder.empty()
            status.write("🔍 Finding relevant information...")
            
            # Get embeddings and search using tensor operations
            query_embedding = self.retriever.encode([query])
            similarities = F.cosine_similarity(query_embedding, self.retriever.doc_embeddings)
            scores, indices = torch.topk(similarities, k=min(self.k, len(self.documents)))
            
            # Print search results for debugging
            print("\nSearch Results:")
            for idx, score in zip(indices.tolist(), scores.tolist()):
                print(f"Score: {score:.4f} | Document: {self.documents[idx][:100]}...")
            
            relevant_docs = [self.documents[idx] for idx in indices.tolist()]
            
            # Update status
            status.write("💭 Generating response...")
            
            # Prepare context and prompt
            context = "\n".join(relevant_docs[:3])  # Only use top 3 most relevant docs
            prompt = f"""Answer this question using the given context. Be specific and detailed.

Context: {context}

Question: {query}

Answer (provide a complete, detailed response):"""
            
            # Generate response
            response_placeholder = placeholder.empty()
            
            try:
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 1024,
                            "temperature": 0.5,
                            "top_p": 0.9,
                            "top_k": 50,
                            "repetition_penalty": 1.03,
                            "do_sample": True
                        }
                    },
                    timeout=30
                ).json()
                
                if response and isinstance(response, list) and len(response) > 0:
                    generated_text = response[0].get('generated_text', '').strip()
                    if generated_text:
                        # Find and extract only the answer part
                        if "Answer:" in generated_text:
                            answer_part = generated_text.split("Answer:")[-1].strip()
                        elif "Answer (provide a complete, detailed response):" in generated_text:
                            answer_part = generated_text.split("Answer (provide a complete, detailed response):")[-1].strip()
                        else:
                            answer_part = generated_text.strip()
                            
                        # Clean up the answer
                        answer_part = answer_part.replace("Context:", "").replace("Question:", "")
                        
                        final_response = self.postprocess_response(answer_part)
                        response_placeholder.markdown(final_response)
                        return final_response
                    
                message = "No relevant answer found. Please try rephrasing your question."
                response_placeholder.warning(message)
                return message
                    
            except Exception as e:
                print(f"Generation error: {str(e)}")
                message = "Had some trouble generating the response. Please try again."
                response_placeholder.warning(message)
                return message
                
        except Exception as e:
            print(f"Process error: {str(e)}")
            message = "Something went wrong. Please try again with a different question."
            placeholder.warning(message)
            return message
def check_environment():
    """Check if the environment is properly set up"""
    if not headers['Authorization']:
        st.error("HUGGINGFACE_API_KEY environment variable not set!")
        st.stop()
        return False
    
    try:
        import torch
        import sentence_transformers
        return True
    except ImportError as e:
        st.error(f"Missing required package: {str(e)}")
        st.stop()
        return False

# @st.cache_resource
# def initialize_rag_pipeline():
#     """Initialize the RAG pipeline once"""
#     data_folder = "ESPN_data"
#     return RAGPipeline(data_folder)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline once"""
    data_folder = "ESPN_data"
    drive_file_id = "1MuV63AE9o6zR9aBvdSDQOUextp71r2NN"
    
    with st.spinner("Loading embeddings from Google Drive..."):
        cache_data = load_from_drive(drive_file_id)
        if cache_data is None:
            st.error("Failed to load embeddings from Google Drive")
            st.stop()
        
        rag = RAGPipeline(data_folder)
        rag.documents = cache_data['documents']
        rag.retriever.store_embeddings(cache_data['embeddings'])
        return rag

def main():
    # Environment check
    if not check_environment():
        return

    # Page config
    st.set_page_config(
        page_title="The Sport Chatbot",
        page_icon="🏆",
        layout="wide"
    )

    # Improved CSS styling
    st.markdown("""
        <style>
        /* Container styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            width: 100%;
        }
        
        /* Button styling */
        .stButton > button {
            width: 200px;
            margin: 0 auto;
            display: block;
            background-color: #FF4B4B;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        
        /* Title styling */
        .main-title {
            text-align: center;
            padding: 1rem 0;
            font-size: 3rem;
            color: #1F1F1F;
        }
        
        .sub-title {
            text-align: center;
            padding: 0.5rem 0;
            font-size: 1.5rem;
            color: #4F4F4F;
        }
        
        /* Description styling */
        .description {
            text-align: center;
            color: #666666;
            padding: 0.5rem 0;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        /* Answer container styling */
        .stMarkdown {
            max-width: 100%;
        }

        /* Streamlit default overrides */
        .st-emotion-cache-16idsys p {
            font-size: 1.1rem;
            line-height: 1.6;
        }
        
        /* Container for main content */
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header section with improved styling
    st.markdown("<h1 class='main-title'>🏆 The Sport Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-title'>Using ESPN API</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p class='description'>
            Hey there! 👋 I can help you with information on Ice Hockey, Baseball, American Football, Soccer, and Basketball. 
            With access to the ESPN API, I'm up to date with the latest details for these sports up until October 2024.
        </p>
        <p class='description'>
            Got any general questions? Feel free to ask—I'll do my best to provide answers based on the information I've been trained on!
        </p>
    """, unsafe_allow_html=True)

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize the pipeline
    try:
        with st.spinner("Loading resources..."):
            rag = initialize_rag_pipeline()
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        st.error("Unable to initialize the system. Please check if all required files are present.")
        st.stop()

    # Create columns for layout with golden ratio
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # Query input with label styling
        query = st.text_input("What would you like to know about sports?")
        
        # Centered button
        if st.button("Get Answer"):
            if query:
                response_placeholder = st.empty()
                try:
                    response = rag.process_query(query, response_placeholder)
                    print(f"Generated response: {response}")
                except Exception as e:
                    print(f"Query processing error: {str(e)}")
                    response_placeholder.warning("Unable to process your question. Please try again.")
            else:
                st.warning("Please enter a question!")

    # Footer with improved styling
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #666666; padding: 1rem 0;'>
            Powered by ESPN Data & Mistral AI 🚀
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()