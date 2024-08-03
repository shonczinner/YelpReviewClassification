from dataclasses import dataclass
from typing import List
from collections import Counter
import torch
import re
# the class "Text_classification_data" takes a list of strings and
# labels aka the data. 

@dataclass
class Text_Classification_Data:
    texts: List[str]
    labels: List[int]

    def get_text(self):
        return " ".join(self.texts)
    
def clean_string(s: str):
    return re.sub(pattern="\\s\\s+", repl=" ", string=re.sub(pattern="[^a-z ^\s]", repl='', string=s.lower()))

class Text_Encoder:
    def __init__(self):
        self.vocab_size = None
        self.top_k_words = None
        self.word_to_index = None
        self.index_to_word = None
        self.padding_token = "<PADDING>"
        self.unknown_token = "<UNKNOWN>"
        self.padding_index = 0
        self.unknown_index = 1
    
    # fit_top_k takes a string, cleans the string, 
    # and sets the attributes with the fitted top_k words 
    def fit_top_k(self, text: str,k: int):
        text = clean_string(text)
        # Split the string into words
        words = text.split(" ")

        # Count word frequencies
        word_counts = Counter(words)

        top_k_words = [word for word,count in word_counts.most_common(k)] # most_common returns List of tuples of (word, count)

        self.vocab_size = len(top_k_words)
        self.top_k_words = top_k_words

        self.word_to_index = {word:index+2 for index,word in enumerate(top_k_words)}
        self.word_to_index[self.padding_token]=self.padding_index
        self.word_to_index[self.unknown_token]=self.unknown_index

        self.index_to_word = {index+2:word for index,word in enumerate(top_k_words)}
        self.index_to_word[self.padding_index]=self.padding_token
        self.index_to_word[self.unknown_index]=self.unknown_token

        return self.top_k_words

    def in_encoder(self,s):
        return (s in self.word_to_index.keys())

    def encode(self,s: str)->List[int]:
        return [self.word_to_index[word] if self.in_encoder(word) else self.unknown_index for word in clean_string(s).split(" ")]

    def decode(self,list_of_indices: List[int])->str:
        return  " ".join([self.index_to_word[index] for index in list_of_indices])

    # calculates max_len and pads encodings to that length
    def pad_encodings(self,list_of_encodings)->List[List[int]]:
        encoding_lens = [len(encoding) for encoding in list_of_encodings]
        max_len = max(encoding_lens)
        return [[self.padding_index]*(max_len-encoding_len)+encoding for encoding,encoding_len in zip(list_of_encodings,encoding_lens)]


# Tensorizing and Dataset/Dataloader train/test class
def tensorize_dataset(padded_encodings,labels):
    x = torch.stack([torch.tensor(encoding,dtype=torch.long) for encoding in padded_encodings])
    y = torch.tensor(labels)
    return x,y


if __name__ == '__main__':
    print(clean_string(".a. a.  a.   a."))

    texts = ["the hi the",
         "the bye, a. A"]
    labels = [1,2]

    dat1 = Text_Classification_Data(texts,labels)

    enc1 = Text_Encoder()

    print(enc1.fit_top_k(dat1.get_text(),2))
    print([enc1.encode(s) for s in dat1.texts])
    print(enc1.decode([2,1,2]))
    print(enc1.pad_encodings([enc1.encode(s) for s in dat1.texts]))
    x,y = tensorize_dataset(enc1.pad_encodings([enc1.encode(s) for s in dat1.texts]),dat1.labels)
    print(x)
    print(y)


