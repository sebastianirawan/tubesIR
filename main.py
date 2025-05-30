import os
from vsm import VSM
# from bm25 import BM25

def main():
    print('\n[Pilih model]\n1. VSM\n2. BM25')
    model = input() # input untuk pilih gunakan model apa
    print('Query: ', end="")
    query = input() # input query

    docPath = os.path.join(os.path.dirname(__file__), 'docs') # path folder dokumen

    if model == '1': #modle vsm dipilih
        vsm = VSM(docPath) # init dengan param path doc (semua dokumen)
        result = vsm.search(query) # search query, mengembalikan hasil ranking dokumen
        print("\nHasil:")
        for filename, score in result: # print hasil: judul doc dan skornya
            title = os.path.splitext(filename)[0] # menghilangkan .txt dari nama file (nama file = judul doc)
            print(f"{title} - score: {score:.3f}")

if __name__ == '__main__':
    main()