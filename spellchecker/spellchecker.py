import re
from collections import Counter
import sys
import time

# Define the autocorrect functions
def words(text): 
    return re.findall(r'\w+', text.lower())

def load_corpus(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return Counter(words(f.read()))
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please provide a valid corpus file.")
        return Counter()

# Load the corpus (use your desired file here)
WORDS = load_corpus('shakespeare.txt')
N = sum(WORDS.values())

def P(word): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# Simulate real-time input
def stream_typing():
    print("Start typing (press Enter to finish):")
    input_text = ""
    current_word = ""

    try:
        while True:
            # Capture single character input
            char = sys.stdin.read(1)  # Use getch-like behavior for real-time systems
            if char in (' ', '\n', '.', ',', '!', '?'):  # Word boundaries
                if current_word:
                    # Autocorrect the current word
                    corrected_word = correction(current_word)
                    print(f"\rCorrected: {corrected_word} ", end='', flush=True)
                    input_text += corrected_word + char
                    current_word = ""  # Reset for the next word
                if char == '\n':  # Exit on Enter key
                    print("\nFinished typing.")
                    break
            else:
                # Append character to the current word
                current_word += char
                print(f"\rTyping: {current_word} ", end='', flush=True)

        print("\nFinal Text:", input_text)
    except KeyboardInterrupt:
        print("\nTyping interrupted.")

# Run the real-time autocorrect simulation
if __name__ == "__main__":
    stream_typing()
