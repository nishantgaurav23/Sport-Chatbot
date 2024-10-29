import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Callable
import glob
from tqdm import tqdm
import pickle
import torch.nn.functional as F
import functools
from datetime import datetime

# Force CPU device
torch.device('cpu')

# Logging configuration
LOGGING_CONFIG = {
    'enabled': True,
    'functions': {
        'encode': True,
        'store_embeddings': True,
        'search': True,
        'load_and_process_csvs': True
    }
}

def log_function(func: Callable) -> Callable:
    """Decorator to log function inputs and outputs"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not LOGGING_CONFIG['enabled'] or not LOGGING_CONFIG['functions'].get(func.__name__, False):
            return func(*args, **kwargs)
        
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
        else:
            class_name = func.__module__

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        log_args = args[1:] if class_name != func.__module__ else args
        
        def format_arg(arg):
            if isinstance(arg, torch.Tensor):
                return f"Tensor(shape={list(arg.shape)}, device={arg.device})"
            elif isinstance(arg, list):
                return f"List(len={len(arg)})"
            elif isinstance(arg, str) and len(arg) > 100:
                return f"String(len={len(arg)}): {arg[:100]}..."
            return arg

        formatted_args = [format_arg(arg) for arg in log_args]
        formatted_kwargs = {k: format_arg(v) for k, v in kwargs.items()}

        print(f"\n{'='*80}")
        print(f"[{timestamp}] FUNCTION CALL: {class_name}.{func.__name__}")
        print(f"INPUTS:")
        print(f"  args: {formatted_args}")
        print(f"  kwargs: {formatted_kwargs}")

        result = func(*args, **kwargs)

        formatted_result = format_arg(result)
        print(f"OUTPUT:")
        print(f"  {formatted_result}")
        print(f"{'='*80}\n")

        return result
    return wrapper

class SentenceTransformerRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "embeddings_cache"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.device = torch.device("cpu")
            self.model = SentenceTransformer(model_name, device="cpu")
            self.doc_embeddings = None
            self.cache_dir = cache_dir
            self.cache_file = "embeddings.pkl"
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self) -> str:
        return os.path.join(self.cache_dir, self.cache_file)
    
    @log_function
    def save_cache(self, cache_data: dict):
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
            print(f"Cache saved at: {cache_path}")
    
    @log_function
    def load_cache(self) -> dict:
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                print(f"Loading cache from: {cache_path}")
                return pickle.load(f)
        return None
    
    @log_function
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        return F.normalize(embeddings, p=2, dim=1)

    @log_function
    def store_embeddings(self, embeddings: torch.Tensor):
        self.doc_embeddings = embeddings

def process_data(data_folder: str):
    retriever = SentenceTransformerRetriever()
    documents = []
    
    # Check cache first
    cache_data = retriever.load_cache()
    if cache_data is not None:
        print("Using cached embeddings")
        return cache_data
    
    # Process CSV files
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    for csv_file in tqdm(csv_files, desc="Reading CSV files"):
        try:
            df = pd.read_csv(csv_file)
            texts = df.apply(lambda x: " ".join(x.astype(str)), axis=1).tolist()
            documents.extend(texts)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
            continue
    
    # Generate embeddings
    embeddings = retriever.encode(documents)
    
    # Save cache
    cache_data = {
        'embeddings': embeddings,
        'documents': documents
    }
    retriever.save_cache(cache_data)
    
    return cache_data

if __name__ == "__main__":
    data_folder = "ESPN_data"
    process_data(data_folder)