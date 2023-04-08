import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import random
import string

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Opening and reading the data from the text file
with open('mental_health.txt', 'r', errors='ignore') as file:
    corpus = file.read().lower()

# Tokenizing and preprocessing the data
sent_tokens = nltk.sent_tokenize(corpus)
word_tokens = nltk.word_tokenize(corpus)

lemmatizer = WordNetLemmatizer()

def lemTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lemNormalize(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Defining the chat function
def chat(user_input):
    bot_response = ''
    sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=lemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        bot_response = bot_response + "I am sorry! I don't understand you."
        return bot_response
    else:
        bot_response = bot_response + sent_tokens[idx]
        return bot_response

# Starting the chat with the user
print("Hello! I am a mental health chatbot. How can I help you?")
while True:
    user_input = input()
    if user_input.lower() == 'bye':
        print("Bye! Take care.")
        break
    else:
        if chat(user_input) != "I am sorry! I don't understand you.":
            print(chat(user_input))
        else:
            print(chat(user_input))  # Repeat the same user_input to continue the chat
