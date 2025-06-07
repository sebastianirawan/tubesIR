import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess

class VSM:
    def __init__(self, docDir): # init dengan path doc (semua dokumen)
        self.fileNames = [] # arr judul doc
        self.documents = [] # arr isi dokumen
        self.raw_documents = [] # arr for original, raw document content
        for file in os.listdir(docDir): # cek semua dokumen di corpus
            if file.endswith('.txt'): # jika file sesuai format (file txt)
                with open(os.path.join(docDir, file), 'r', encoding='utf-8') as f: #baca isi doc
                    raw_content = f.read()
                    self.raw_documents.append(raw_content) # store raw content
                    self.documents.append(preprocess(raw_content)) #preprocess isi doc, simpan ke array
                    self.fileNames.append(file)# simpan nama file (judul)
        self.vectorizer = TfidfVectorizer(sublinear_tf=True)# tf-idf vectorizer dengan sublinear tf scaling
        self.tfidfMatrix = self.vectorizer.fit_transform(self.documents) #matrix tf-idf dengan dimensi (jumlah dokumen x jumlah term dalam vocabulary), berbentuk sparse matrix: matrix yg hanya menyimpan non-zero value

    def search (self, query): # method search dgn param query
        queryVector = self.vectorizer.transform([preprocess(query)])# ubah query jadi vector, menghasilkan sparse matrix
        queryVectorMap = { # buat map dari vector query, key = index kata di vocabulary, value = skor tf-idf (karena sparse matrix, hanya menyimpan key yang ada value -> hanya index kata dari query dengan skornya)
            index: value
            for index, value in zip(queryVector.indices, queryVector.data)
        }

        result = [] # list tuple (tuple: judul doc, skornya, snippetnya)

        for i in range(self.tfidfMatrix.shape[0]): #iterasi tiap doc
            documentVector = self.tfidfMatrix[i] # buat doc jadi vector

            documentVectorMap = {# buat tiap vector doc menjadi map (seperti query)
                index: value
                for index, value in zip(documentVector.indices, documentVector.data)
            }

            dotProduct = 0.0 # hitung dot product vektor query dan dokumen
            for j in queryVectorMap: # untuk setiap term di query
                if j in documentVectorMap: # jika term tersebut ada di doc i
                    dotProduct += queryVectorMap[j] * documentVectorMap[j] # tambahkan hasil perkalian skor term di doc dan di query ke dot product

            # --- Corrected Snippet Extraction ---
            snippet = ""
            raw_doc = self.raw_documents[i]
            lines = raw_doc.splitlines()

            # The document body starts after the title (line 0) and a blank line (line 1)
            if len(lines) > 2:
                # Join the actual body content back into a single string
                document_body = '\n'.join(lines[2:])
                # Tokenize the body into sentences and get the first one.
                sentences = nltk.sent_tokenize(document_body)
                if sentences:
                    snippet = sentences[0].replace('\n', ' ')

            result.append((self.fileNames[i], dotProduct, snippet)) #tambahkan ke list

        rank = sorted(result, key=lambda x: x[1], reverse=True) #sort

        return self.retrieve(rank) #kembalikan top 30
    
    def retrieve (self, rank, threshold=0.09):
        if not rank:
            return []
        relevantDocs = [
            (filename, score, snippet)
            for filename, score, snippet in rank
            if score >= threshold
        ]
        return relevantDocs
