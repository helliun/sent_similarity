This library was created entirely by GPT-4 from scratch, given a one-sentence description as a prompt

# Sent-Similarity: Sentence Similarity Python Module

Sent-Similarity is a Python module based on the TextProcessor class, which allows you to find the top n sentences that are most similar to a given input text using a sentence transformer model.

## Installation

Before using sent_similarity.py, you will need to install the required packages:


pip install torch pandas nltk sentence-transformers scikit-learn


## Usage

1. Import the module and the class:

python
from sent_similarity import TextProcessor


2. Initialize the TextProcessor class by providing the sentence transformer model name:

python
text_processor = TextProcessor(model_name='distilbert-base-nli-stsb-mean-tokens')


3. Call the get_top_n_sentences method with the input text and a value for n:

python
input_text = "Your input text here."
top_n_sentences = 3  # Number of top similar sentences you want to retrieve

most_similar_sentences = text_processor.get_top_n_sentences(input_text, top_n_sentences)


4. Handle the returned list of sentences:

python
print("Top", top_n_sentences, "most similar sentences:")
for i, sentence in enumerate(most_similar_sentences):
    print(i + 1, "-", sentence)


## Example

Here is a complete example of how to use the sent_similarity.py module:

python
from sent_similarity import TextProcessor

# Initialize TextProcessor with a model name
text_processor = TextProcessor(model_name='distilbert-base-nli-stsb-mean-tokens')

# Input text and number of top sentences
input_text = "Your input text here."
top_n_sentences = 3

# Get the top n most similar sentences
most_similar_sentences = text_processor.get_top_n_sentences(input_text, top_n_sentences)

# Print the results
print("Top", top_n_sentences, "most similar sentences:")
for i, sentence in enumerate(most_similar_sentences):
    print(i + 1, "-", sentence)


Simply replace "Your input text here." with the desired input text, and the module will return the top n most similar sentences, as specified by top_n_sentences.
