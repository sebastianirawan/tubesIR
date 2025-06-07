import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt') # model tokenizer untuk memecah kalimat menjadi kata
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))# daftar stopword bahasa inggris
stemmer = PorterStemmer()# ubah kata ke bentuk dasar dng porter stemmer

def preprocess(text): # preprocess text, text yaitu dokumen dan query
    text = text.lower() # ubah jadi lower case
    tokens = nltk.word_tokenize(text) # teks diubah menjadi kata-kata
    tokens = [t for t in tokens if t.isalnum()] # hapus token yang non huruf atau angka
    tokens = [t for t in tokens if t not in stopWords]# hapus stopword
    tokens = [stemmer.stem(t) for t in tokens] # stemming
    return ' '.join(tokens) #join kata-kata kembali menjadi string
