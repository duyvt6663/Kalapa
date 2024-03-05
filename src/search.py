from langchain.schema.vectorstore import VectorStoreRetriever, BaseRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from pydantic import BaseModel, Extra
from typing import ClassVar
import numpy as np
import ahocorasick
import random
import pickle
import json
import os


class AdvRetriever(BaseRetriever):
    bm25: ClassVar[BM25Retriever]
    vector: ClassVar[VectorStoreRetriever]
    document_path: ClassVar[str] = ""
    
    class Config:
        validate_assignment=False
        extra = Extra.allow
        
    def __init__(self, automaton_path, document_path, embedding, k1=3, k2=2):            
        super().__init__()
        
        assert document_path != ""
        with open(automaton_path, 'rb') as f:
            self.automaton = pickle.load(f)
            self.automaton.make_automaton()
        
        self.embedding = embedding
        self.set_retriever(document_path, embedding, k1, k2)
#         hyperparams for retrievers
        self.k1 = k1 # bm25
        self.k2 = k2 # vector
        # diseases
        self.diseases = set(self.automaton.keys())
    
    @staticmethod
    def retrieve_texts_and_embeddings(document_path:str="", folders:list=None):
        if not folders:
            assert document_path != ""
            folders = os.listdir(document_path)
            
        texts, embeddings = [], []
        for folder in folders:
            folder_path = os.path.join(document_path or AdvRetriever.document_path, folder)
            docs = json.load(open(folder_path + '/' + 'data.json', 'r', encoding='utf-8'))
            docs_embs = np.load(folder_path + '/' + 'embeddings.npy')
            texts.extend(docs)
            embeddings.extend(docs_embs)
        text_embeddings = list(zip(texts, embeddings))
        return texts, text_embeddings
    
    @classmethod
    def set_retriever(cls, document_path, embedding, k1, k2):
        if cls.document_path == document_path: # already set
            return
        
        #         set the default 
        texts, text_embeddings = cls.retrieve_texts_and_embeddings(document_path)
        cls.bm25 = BM25Retriever.from_texts(texts)
        cls.bm25.k = k1
        vectorstore = FAISS.from_embeddings(text_embeddings = text_embeddings, embedding=embedding)
        cls.vector = vectorstore.as_retriever(search_kwargs=dict(k=k2))
        cls.document_path = document_path      
    
    def _get_relevant_documents(
        self, query: str, *, run_manager
    ):
        disease_matches = set()
        for _, v in self.automaton.iter(query):
            disease_matches.add(v)
        if 'Ho' in disease_matches: # outlier where Ho can be matched in the middle of a word like "Cho"
            toks = query.split(' ')
            if 'Ho' not in toks:
                disease_matches.remove('Ho')
        
        if not disease_matches: # normal cases 
            bm25_retriever = self.bm25
            vector_retriever = self.vector
        else: # with automaton
            texts, text_embeddings = self.retrieve_texts_and_embeddings(folders=disease_matches)
        
    #         set up retriever
            bm25_retriever = BM25Retriever.from_texts(texts)
            bm25_retriever.k = self.k1
            vectorstores = FAISS.from_embeddings(text_embeddings = text_embeddings, embedding=self.embedding)
            vector_retriever = vectorstores.as_retriever(search_kwargs=dict(k=self.k2))
            
        retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],
                             weights=[0.5, 0.5])
        return retriever.get_relevant_documents(query)
    
    def get_irrelevant_doc(self, query):
        disease_matches = set()
        for _, v in self.automaton.iter(query):
            disease_matches.add(v)
        if 'Ho' in disease_matches: # outlier where Ho can be matched in the middle of a word like "Cho"
            toks = query.split(' ')
            if 'Ho' not in toks:
                disease_matches.remove('Ho')
        
        if not disease_matches:
            seed = random.choice(list(self.diseases))
        else:
            new_disease_set = self.diseases - disease_matches
            new_diseases = random.sample(new_disease_set, len(disease_matches))
            for new_disease, old_disease in zip(new_diseases, disease_matches):
                query = query.replace(old_disease, new_disease)
            seed = query 
        
        return self.bm25.get_relevant_documents(seed)
