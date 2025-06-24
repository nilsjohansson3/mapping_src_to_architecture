from sentence_transformers import SentenceTransformer
import torch
from lib.unixcoder import UniXcoder
import numpy as np
import transformers
import openai
from openai import OpenAI
from openai import AzureOpenAI
import tiktoken
from tqdm import tqdm  # For progress bar
import time

#from azure.ai.openai import OpenAIClient
#from azure.core.credentials import AzureKeyCredential

config = {
    'OPENAI_API_BASE': ## TODO ADD,
    'OPENAI_API_KEY': ## TODO ADD,
    'OPENAI_API_VERSION': ## TODO ADD,
    'EMBEDDING_MODEL': ## TODO ADD
    'CODE_EMBEDDING_MODEL': ""
}

# Set up the OpenAI API client for Azure
#openai.api_type = "azure"
openai.azure_endpoint = config['OPENAI_API_BASE']
openai.api_key = config['OPENAI_API_KEY']
openai.api_version = config['OPENAI_API_VERSION']


class FeatureExtractor:
    def __init__(self, model, useLocal=True):
        """
        Initialize the FeatureExtractor with the specified model.

        :param model: The model to use ('sentencetransformer', 'unixcoder-base-nine', or 'openai-embedding').
        :param useLocal: If True, use local models; otherwise, fetch from external APIs.
        :param openai_api_key: OpenAI API key, required if model is 'openai-embedding'.
        """
        self.model_name = model
        self.useLocal = useLocal

        if model == 'openai-code-embedding':
            self.client = OpenAI(
                api_key='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' # LNU key
            )
        elif model == 'openai-embedding':
            pass
        

        else:
            raise ValueError("Invalid model name. Choose 'sentencetransformer', 'unixcoder-base-nine', 'openai-embedding', or 'openai-code-embedding'.")

    def get_feature_vectors(self, X, batch_size=16, max_tokens_per_batch=8191, max_text_length=4000, max_retries=5):
        """
        Get the feature vectors for the input text using the specified model.

        :param X: A list of texts to be embedded.
        :param batch_size: Number of texts to send in each API request (for OpenAI API).
        :param max_tokens_per_batch: Maximum number of tokens to include in each batch request (for OpenAI API).
        :param max_text_length: Maximum length of each text to avoid exceeding token limits.
        :param max_retries: Maximum number of retries for rate-limited API requests.
        :return: A list or array of embedding vectors.
        """
        def retry_request(request_func, *args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return request_func(*args, **kwargs)
                except openai.RateLimitError:
                    wait_time = 2 ** retries
                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                except Exception as e:
                    print(f"Unexpected error: {e}. Retrying...")
                    wait_time = 2 ** retries
                    time.sleep(wait_time)
                    retries += 1
            raise Exception("Maximum retries exceeded. Failed to fetch embeddings.")
        

        if self.model_name == 'openai-embedding':
            client = AzureOpenAI(
                azure_endpoint=config['OPENAI_API_BASE'],
                api_key=config['OPENAI_API_KEY'],
                api_version=config['OPENAI_API_VERSION']
            )

            tokenizer = tiktoken.get_encoding("cl100k_base")
            embeddings = []
            batch_texts = []
            token_count = 0
            embedding_dim = None
            failed_texts = []

            try:
                for text in tqdm(X, desc="Fetching embeddings"):
                    if not text or text.strip() == "":
                        # Append a zero vector if the text is empty or invalid
                        if embedding_dim is not None:
                            zero_vector = np.zeros(embedding_dim)
                            embeddings.append(zero_vector)
                        else:
                            # Temporarily append None; will replace later
                            embeddings.append(None)
                        continue

                    if len(text) > max_text_length:
                        text = text[:max_text_length]

                    text_tokens = tokenizer.encode(text)
                    text_token_count = len(text_tokens)

                    if token_count + text_token_count > max_tokens_per_batch or len(batch_texts) >= batch_size:
                        if batch_texts:
                            try:
                                response = retry_request(client.embeddings.create, input=batch_texts, model=config["EMBEDDING_MODEL"])
                                batch_embeddings = []
                                for data in response.data:
                                    if hasattr(data, 'embedding') and data.embedding:
                                        batch_embeddings.append(data.embedding)
                                    else:
                                        print("Missing embedding in response data.")
                                        if embedding_dim is not None:
                                            batch_embeddings.append(np.zeros(embedding_dim))
                                        else:
                                            batch_embeddings.append(None)

                                # Set embedding_dim if not already set
                                if embedding_dim is None and batch_embeddings:
                                    embedding_dim = len(batch_embeddings[0])

                                # Ensure the response length matches batch_texts length
                                if len(batch_embeddings) != len(batch_texts):
                                    print(f"Error: Expected {len(batch_texts)} embeddings but got {len(batch_embeddings)}")
                                    # Append zero vectors for missing embeddings
                                    for _ in range(len(batch_texts) - len(batch_embeddings)):
                                        if embedding_dim is not None:
                                            batch_embeddings.append(np.zeros(embedding_dim))
                                        else:
                                            batch_embeddings.append(None)

                                # Replace any None with zero vectors if embedding_dim is known
                                if embedding_dim is not None:
                                    batch_embeddings = [emb if emb is not None else np.zeros(embedding_dim) for emb in batch_embeddings]

                                embeddings.extend(batch_embeddings)

                            except Exception as e:
                                print(f"Error fetching embeddings for batch: {e}")
                                # Log failed texts
                                failed_texts.extend(batch_texts)
                                # Append zero vectors for failed batch if embedding_dim is known
                                if embedding_dim is not None:
                                    zero_vector = np.zeros(embedding_dim)
                                    for _ in batch_texts:
                                        embeddings.append(zero_vector)
                                else:
                                    # Cannot determine embedding_dim; append None
                                    for _ in batch_texts:
                                        embeddings.append(None)

                        # Reset batch
                        batch_texts = [text]
                        token_count = text_token_count
                    else:
                        batch_texts.append(text)
                        token_count += text_token_count

                # Final batch processing
                if batch_texts:
                    try:
                        response = retry_request(client.embeddings.create, input=batch_texts, model=config["EMBEDDING_MODEL"])
                        batch_embeddings = []
                        for data in response.data:
                            if hasattr(data, 'embedding') and data.embedding:
                                batch_embeddings.append(data.embedding)
                            else:
                                print("Missing embedding in response data.")
                                if embedding_dim is not None:
                                    batch_embeddings.append(np.zeros(embedding_dim))
                                else:
                                    batch_embeddings.append(None)

                        # Set embedding_dim if not already set
                        if embedding_dim is None and batch_embeddings:
                            embedding_dim = len(batch_embeddings[0])

                        # Ensure the response length matches batch_texts length
                        if len(batch_embeddings) != len(batch_texts):
                            print(f"Error: Expected {len(batch_texts)} embeddings but got {len(batch_embeddings)}")
                            # Append zero vectors for missing embeddings
                            for _ in range(len(batch_texts) - len(batch_embeddings)):
                                if embedding_dim is not None:
                                    batch_embeddings.append(np.zeros(embedding_dim))
                                else:
                                    batch_embeddings.append(None)

                        # Replace any None with zero vectors if embedding_dim is known
                        if embedding_dim is not None:
                            batch_embeddings = [emb if emb is not None else np.zeros(embedding_dim) for emb in batch_embeddings]

                        embeddings.extend(batch_embeddings)

                    except Exception as e:
                        print(f"Error fetching embeddings for final batch: {e}")
                        # Log failed texts
                        failed_texts.extend(batch_texts)
                        # Append zero vectors for failed batch if embedding_dim is known
                        if embedding_dim is not None:
                            zero_vector = np.zeros(embedding_dim)
                            for _ in batch_texts:
                                embeddings.append(zero_vector)
                        else:
                            # Cannot determine embedding_dim; append None
                            for _ in batch_texts:
                                embeddings.append(None)

                # Replace any None with zero vectors if embedding_dim is known
                if embedding_dim is not None:
                    for idx, emb in enumerate(embeddings):
                        if emb is None:
                            embeddings[idx] = np.zeros(embedding_dim)
                else:
                    raise ValueError("Unable to determine embedding dimension.")

                # Final check to ensure alignment
                if len(embeddings) != len(X):
                    print(f"Error: Mismatch between input and output lengths. Input: {len(X)}, Output: {len(embeddings)}")
                    # Fill missing embeddings with zero vectors
                    while len(embeddings) < len(X):
                        embeddings.append(np.zeros(embedding_dim))

                if failed_texts:
                    print(f"Warning: Failed to fetch embeddings for {len(failed_texts)} texts.")

            except Exception as e:
                print(f"Error fetching embeddings: {e}")
                print(f"Error fetching embeddings for: {batch_texts}")

            return np.array(embeddings)

        elif self.model_name == 'openai-code-embedding':
            tokenizer = tiktoken.get_encoding("cl100k_base")
            embeddings = []
            batch_texts = []
            token_count = 0
            embedding_dim = None

            try:
                for text in tqdm(X, desc="Fetching code embeddings"):
                    if not text or text.strip() == "":
                        if embedding_dim is not None:
                            zero_vector = np.zeros(embedding_dim)
                            embeddings.append(zero_vector)
                        continue

                    if len(text) > max_text_length:
                        text = text[:max_text_length]

                    text_tokens = tokenizer.encode(text)
                    text_token_count = len(text_tokens)

                    if token_count + text_token_count > max_tokens_per_batch or len(batch_texts) >= batch_size:
                        if batch_texts:
                            response = retry_request(self.client.Embedding.create, input=batch_texts, model=config["CODE_EMBEDDING_MODEL"])
                            batch_embeddings = [data["embedding"] for data in response["data"]]

                            if embedding_dim is None and batch_embeddings:
                                embedding_dim = len(batch_embeddings[0])

                            embeddings.extend(batch_embeddings)

                        batch_texts = [text]
                        token_count = text_token_count
                    else:
                        batch_texts.append(text)
                        token_count += text_token_count

                if batch_texts:
                    response = retry_request(self.client.Embedding.create, input=batch_texts, model=config["CODE_EMBEDDING_MODEL"])
                    batch_embeddings = [data["embedding"] for data in response["data"]]

                    if embedding_dim is None and batch_embeddings:
                        embedding_dim = len(batch_embeddings[0])

                    embeddings.extend(batch_embeddings)

                if embedding_dim is not None:
                    for text in X:
                        if not text or text.strip() == "":
                            zero_vector = np.zeros(embedding_dim)
                            embeddings.append(zero_vector)

            except Exception as e:
                print(f"Error fetching code embeddings: {e}")
                print(f"Error fetching embeddings of: {batch_texts}")

            return np.array(embeddings)
        else:
            raise ValueError("Invalid model name. Choose either 'sentencetransformer', 'unixcoder-base-nine', 'openai-embedding'.")