import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

class NgramModel:
    def __init__(self, n: int = 2):
        self.n = n
        self.vocab = defaultdict(lambda: 0)
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.vocab_size = 0
        
    def build_vocab(self, corpus: List[List[str]], max_words: int = 2000):
        word_counts = Counter([word for sent in corpus for word in sent])
        vocab_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_words]
        self.vocab = {word: idx+1 for idx, (word, _) in enumerate(vocab_words)}
        self.vocab['<UNK>'] = 0
        self.vocab_size = len(self.vocab)
        
    def get_ngrams(self, sentence: List[str]) -> List[Tuple[tuple, str]]:
        padded = ['<START>'] * (self.n-1) + sentence + ['<END>']
        ngrams = []
        for i in range(len(padded)-self.n+1):
            context = tuple(padded[i:i+self.n-1])
            target = padded[i+self.n-1]
            ngrams.append((context, target))
        return ngrams
    
    def train(self, corpus: List[List[str]]):
        # Count n-grams
        for sentence in corpus:
            ngrams = self.get_ngrams(sentence)
            for context, target in ngrams:
                self.ngram_counts[context][target] += 1
                
    def get_probability(self, context: tuple, word: str) -> float:
        # Maximum Likelihood Estimation
        if context not in self.ngram_counts:
            return 1.0 / self.vocab_size
        total_count = sum(self.ngram_counts[context].values())
        word_count = self.ngram_counts[context][word]
        return word_count / total_count
    
    def sentence_probability(self, sentence: List[str]) -> float:
        prob = 1.0
        ngrams = self.get_ngrams(sentence)
        for context, target in ngrams:
            prob *= self.get_probability(context, target)
        return prob
    
    def perplexity(self, sentence: List[str]) -> float:
        prob = self.sentence_probability(sentence)
        if prob == 0:
            print("Warning: probability is zero, returning infinity")
            return float('inf')
        return (1/prob) ** (1/len(sentence))
    
    def generate(self, context: tuple, max_length: int = 20) -> List[str]:
        generated = list(context)
        for _ in range(max_length):
            curr_context = tuple(generated[-(self.n-1):])
            if curr_context not in self.ngram_counts:
                break
                
            # Convert counts to probabilities
            counts = self.ngram_counts[curr_context]
            total = sum(counts.values())
            words = list(counts.keys())
            probs = [count/total for count in counts.values()]
            
            # Sample next word based on probabilities
            next_word = np.random.choice(words, p=probs)
            if next_word == '<END>':
                break
            generated.append(next_word)
        return generated
      
    def get_unigram_counts(self) -> Dict[str, int]:
      """Returns counts of individual words in the corpus"""
      unigram_counts = defaultdict(int)
      for context in self.ngram_counts.keys():
          for word, count in self.ngram_counts[context].items():
              unigram_counts[word] += count
      return dict(unigram_counts)

# Example usage
def example():
    # Example corpus from your question
    corpus = [
        ["the", "cat", "chases", "the", "mouse"],
        ["the", "mouse", "eats", "the", "cheese"],
        ["the", "cheese", "attracts", "the", "mouse"],
        ["the", "mouse", "fears", "the", "cat"]
    ]
      
    # Create and train trigram model
    """
Unigram:
A unigram is a single word or token.
Example: For the sentence "I love machine learning", the unigrams are ["I", "love", "machine", "learning"].

Bigram:
A bigram is a sequence of two adjacent words or tokens.
Example: For the sentence "I love machine learning", the bigrams are [("I", "love"), ("love", "machine"), ("machine", "learning")].
Trigram:

A trigram is a sequence of three adjacent words or tokens.
Example: For the sentence "I love machine learning", the trigrams are [("I", "love", "machine"), ("love", "machine", "learning")].
Higher-order n-grams:

These are sequences of n adjacent words or tokens, where n > 3.
Example: For the sentence "I love machine learning", the 4-gram is [("I", "love", "machine", "learning")].
    """
    
    model = NgramModel(n=3)
    model.build_vocab(corpus)
    model.train(corpus)
    
    # Calculate probability of a sentence
    test_sent = ["i", "love", "machine", "learning"]
    prob = model.sentence_probability(test_sent)
    print(f"Probability of '{' '.join(test_sent)}': {prob}")
    
    # Generate text
    context = ("the", "cat")
    generated = model.generate(context)
    print(f"Generated text from context '{' '.join(context)}': {' '.join(generated)}")
    
    perb = model.perplexity(test_sent)
    print(f"Perplexity of '{' '.join(test_sent)}': {perb}")

if __name__ == "__main__":
    example()