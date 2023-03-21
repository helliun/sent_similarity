#requirements: torch, pandas, nltk, sentence-transformers, scikit-learn

#imports
import torch
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#code
class TextProcessor:
    def __init__(self, model_name):
        self.model = self.initialize_sentence_transformer(model_name)

    @staticmethod
    def tokenize_sentences(input_text):
        sentences = nltk.sent_tokenize(input_text)
        return sentences

    def initialize_sentence_transformer(self, model_name):
        model = SentenceTransformer(model_name)
        return model

    def encode_text_and_sentences(self, input_text):
        sentences = self.tokenize_sentences(input_text)
        embedding_text = self.model.encode(input_text)
        embedding_sentences = [self.model.encode(sentence) for sentence in sentences]
        return embedding_text, embedding_sentences

    @staticmethod
    def calculate_cosine_similarity(input_text, embedding_text, embedding_sentences):
        similarity_scores = [
            cosine_similarity(embedding_text.reshape(1, -1), sentence_embedding.reshape(1, -1))
            for sentence_embedding in embedding_sentences
        ]
        return similarity_scores

    def sort_sentences_by_similarity(self, input_text, embedding_text, embedding_sentences):
        similarity_scores = self.calculate_cosine_similarity(
            input_text, embedding_text, embedding_sentences
        )
        sentences = self.tokenize_sentences(input_text)
        sorted_sentences = sorted(
            zip(sentences, similarity_scores), key=lambda x: x[1], reverse=True
        )
        return sorted_sentences

    def select_top_n_sentences(self, input_text, embedding_text, embedding_sentences, n):
        sorted_sentences = self.sort_sentences_by_similarity(
            input_text, embedding_text, embedding_sentences
        )
        top_n_sentences = [sentence_score[0] for sentence_score in sorted_sentences[:n]]
        return top_n_sentences

    def get_top_n_sentences(self, input_text, n):
        embedding_text, embedding_sentences = self.encode_text_and_sentences(input_text)
        top_n_sentences = self.select_top_n_sentences(
            input_text, embedding_text, embedding_sentences, n
        )
        return top_n_sentences