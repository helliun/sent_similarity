#requirements: torch, pandas, nltk, sentence-transformers, scikit-learn

#imports
import nltk
from sentence_transformers import SentenceTransformer
from sent_similarity import TextProcessor

#code
def main():
    input_text = "The quick brown fox jumps over the lazy dog. The lazy dog just lays there. The brown fox laughs."
    model_name = "sentence-transformers/distilbert-base-nli-mean-tokens"
    n = 2

    text_processor = TextProcessor(model_name)
    top_n_sentences = text_processor.get_top_n_sentences(input_text, n)
    
    print("Top", n, "most similar sentences:")
    for sentence in top_n_sentences:
        print(sentence)

if __name__ == "__main__":
    main()