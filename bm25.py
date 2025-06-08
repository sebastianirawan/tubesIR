import os
import math
import nltk
from preprocessing import preprocess

class BM25:
    def __init__(self, docDir, k=1.5, b=0.75): # Parameter k dan b berdasarkan angka umum
        self.k = k # Parameter k pengaruh frekuensi term
        self.b = b # Parameter b normalisasi panjang dokumen
        self.documents = [] # arr isi dokumen
        self.fileNames = [] # arr isi nama file
        self.raw_documents = [] # arr for raw document, sebelum preprocess
        self.term_freqs = [] # term frequencies per document
        self.doc_lengths = [] # length of each doc
        self.doc_freq = {} # how many docs contains each term
        self.N = 0 # total n of docs
        self.avg_doc_len = 0.0 # rata rata panjang dokumen
        
        # baca semua dokumen dalam folder
        for file in os.listdir(docDir):
            if file.endswith('.txt'): # proses file .txt
                with open(os.path.join(docDir, file), 'r', encoding='utf-8') as f: #baca isi doc
                    raw_content = f.read()
                    self.raw_documents.append(raw_content) # masukan isi raw doc
                    self.fileNames.append(file) # simpan nama file
                    processed = preprocess(raw_content).split() # preprocess raw doc

                    self.documents.append(processed) # masukan doc yang sudah preprocess
                    self.doc_lengths.append(len(processed)) # masukan panjang setiap doc
                    self.N += 1 # counter menghitung banyak doc
                    
                    # Count term freq
                    tf = {}
                    for term in processed:
                        tf[term] = tf.get(term, 0) + 1
                    self.term_freqs.append(tf)

                    # Count doc freq
                    for term in set(processed):
                        self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        
        # hitung rata-rata panjang doc
        self.avg_doc_len = sum(self.doc_lengths) / self.N
    
    def compute_idf(self, term):
        # hitung idf menggunakan rumus bm25
        df = self.doc_freq.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query_terms, doc_idx):
        score = 0.0
        doc_tf = self.term_freqs[doc_idx] # frek term pada doc
        doc_len = self.doc_lengths[doc_idx] # panjang doc

        # menghitung score
        for term in query_terms:
            if term in doc_tf:
                f = doc_tf[term] # frek term t pada doc
                idf = self.compute_idf(term)
                denom = f + self.k * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * ((f * (self.k + 1)) / denom)
        return score
    
    def search(self, query):
        query_terms = preprocess(query).split() # process query
        result = [] # list hasil filname, score, snippet

        # hitung score dari semua doc terhadap query
        for i in range(self.N):
            score = self.score(query_terms, i)

            snippet = ""
            lines = self.raw_documents[i].splitlines()

            # The document body starts after the title (line 0) and a blank line (line 1)
            if len(lines) > 2:
                # Join the actual body content back into a single string
                body = '\n'.join(lines[2:])
                # Tokenize the body into sentences and get the first one.
                sentences = nltk.sent_tokenize(body)
                if sentences:
                    snippet = sentences[0].replace('\n', ' ')

            result.append((self.fileNames[i], score, snippet)) #tambahkan ke list
        
        #urutkan hasil berdasarkan skor tertinggi
        ranked = sorted(result, key=lambda x: x[1], reverse=True)
        return ranked[:30] #kembalikan top 30 doc

