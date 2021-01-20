"""
Inherit Haystack Base Retriever to create a 2 layers Retriever for TopDup NLP Project
For more information, please read (github.com/topdup) about the project
This part is the 2 layer of the model
"""

import logging
from typing import List, Union, Tuple, Optional
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from haystack.document_store.base import BaseDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store.elasticsearch import Elasticsearch
from haystack.document_store.memory import InMemoryDocumentStore
from haystack import Document
from haystack.retriever.base import BaseRetriever

from farm.infer import Inferencer
from farm.modeling.tokenization import Tokenizer
from farm.modeling.language_model import LanguageModel
from farm.modeling.biadaptive_model import BiAdaptiveModel 
from farm.modeling.prediction_head import TextSimilarityHead
from farm.data_handler.processor import TextSimilarityProcessor
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.dataloader import NamedDataLoader
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from torch.utils.data.sampler import SequentialSampler


logger = logging.getLogger(__name__)


class SecondLayerTopDupRetriever(BaseRetriever):
    """
            Recreate Haystack Dense Retiever for 2nd layer in TopDup Model combine with FAISS
            Dense Retriever is a bi-encoder (in this situation - one transformer for top k output in layer 1, one transformer for the original passage)
            Instead of aiming for Q&A purpose, this Dense Retriever aim to return a similarity score comparison between 3,4 different passages.
            With the output is top k = 1 and similarity score of the passage compare to the original passage 
    """

    def __init__(self, 
                 document_store: BaseDocumentStore,
                 original_embedding_model: Union[Path, str] = "#",
                 passage_embedding_model: Union[Path, str] = "#",
                 use_gpu: bool = True,
                 batch_size: int = 16,
                 embed_title: bool = True,
                 use_fast_tokenize: bool = True,
                 similarity_function: str = "dot_product"
                 ):
        """
        Init the Retriever included encoder models from a local or remote checkpoint.
        The checkpoint format matches huggingface transformers' model format

        **Example:**
            ```python
            |   DensePassageRetriever(document_store = doc_store,
            |                          original_embedding_model = "model_directory/passage-encoder",
            |                          passage_embedding_model = "model_directory/ topk-encoder")
            ```

        """

        self.document_store = document_store
        self.batch_size = batch_size
        
        if document_store is None:
            logger.warning("Retriever has zero information. Please provide a document store to retrive info")
        elif document_store.similarity != "dot_product":
            logger.warning("Recommend to use dot_product for Dense Retriever")
        
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
         
        

    
    def retrieve(self):
        pass

    def train(self):
        pass

    @classmethod
    def load():
        pass

    def embed_original(self):
        pass

    def embed_topk(self):
        pass

    def get_sim_score(self):
        pass

    def save_model(self):
        pass



