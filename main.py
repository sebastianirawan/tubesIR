import os
from vsm import VSM
from bm25 import BM25

def main():
    print('\n[Pilih model]\n1. VSM\n2. BM25')
    model = input() # input untuk pilih gunakan model apa
    print('Query: ', end="")
    query = input() # input query

    docPath = os.path.join(os.path.dirname(__file__), 'docs') # path folder dokumen

    if model == '1': #modle vsm dipilih
        vsm = VSM(docPath) # init dengan param path doc (semua dokumen)
        result = vsm.search(query) # search query, mengembalikan hasil ranking dokumen
        
        print("\n--- Search Results ---")
        if not result or result[0][1] == 0.0:
            print("No relevant documents found.")
        else:
            for filename, score, snippet in result: # unpack filename, score, and snippet
                if score > 0: # Only show documents with a relevance score
                    title = os.path.splitext(filename)[0] # menghilangkan .txt dari nama file (nama file = judul doc)
                    print(f"   Title: {title}")
                    print(f"   Score: {score:.3f}")
                    print(f"   Snippet: {snippet}\n")
    elif model == '2': # model bm25 dipilih
        bm25 = BM25(docPath) # init dengan param path doc (semua dokumen)
        result = bm25.search(query) # search query, mengembalikan hasil ranking dokumen

        print("\n--- Search Results ---")
        if not result or result[0][1] == 0.0:
            print("No relevant documents found.")
        else:
            for filename, score, snippet in result: # unparck filename, score, and snippet
                if score > 0: # show document with relevance score
                    title = os.path.splitext(filename)[0]
                    print(f"   Title: {title}")
                    print(f"   Score: {score:.3f}")
                    print(f"   Snippet: {snippet}\n")


if __name__ == '__main__':
    main()