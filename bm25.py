import os
import math
import nltk
from preprocessing import preprocess

class BM25:
    def __init__(self, docDir, k=1.5, b=0.75): # Parameter k dan b berdasarkan angka umum
        self.k = k # Parameter k
        self.b = b # Parameter b
        self.documents = [] # arr isi dokumen
        self.fileNames = [] # arr isi nama file
        self.raw_documents = [] # arr for original, raw document content
        self.term_freqs = [] # term frequencies per document
        self.doc_lengths = [] # length of each doc
        self.doc_freq = {} # how many docs contains each term
        self.N = 0 # total n of docs
        self.avg_doc_len = 0.0
        for file in os.listdir(docDir):
            if file.endswith('.txt'):
                with open(os.path.join(docDir, file), 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                    self.raw_documents.append(raw_content)
                    self.fileNames.append(file)
                    processed = preprocess(raw_content).split()

                    self.documents.append(processed)
                    self.doc_lengths.append(len(processed))
                    self.N += 1
                    
                    # Count term freq
                    tf = {}
                    for term in processed:
                        tf[term] = tf.get(term, 0) + 1
                    self.term_freqs.append(tf)

                    # Count doc freq
                    for term in set(processed):
                        self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        
        self.avg_doc_len = sum(self.doc_lengths) / self.N
    
    def compute_idf(self, term):
        df = self.doc_freq.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query_terms, doc_idx):
        score = 0.0
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        for term in query_terms:
            if term in doc_tf:
                f = doc_tf[term]
                idf = self.compute_idf(term)
                denom = f + self.k * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * ((f * (self.k + 1)) / denom)
        return score
    
    def search(self, query):
        query_terms = preprocess(query).split()
        result = []

        for i in range(self.N):
            score = self.score(query_terms, i)

            snippet = ""
            lines = self.raw_documents[i].splitlines()
            if len(lines) > 2:
                body = '\n'.join(lines[2:])
                sentences = nltk.sent_tokenize(body)
                if sentences:
                    snippet = sentences[0].replace('\n', ' ')

            result.append((self.fileNames[i], score, snippet))
        
        ranked = sorted(result, key=lambda x: x[1], reverse=True)
        return ranked[:30]

