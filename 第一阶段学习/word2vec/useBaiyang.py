import pickle
from sklearn.metrics.pairwise import cosine_similarity

def top_related(center_words,n=10):
    
    
    for center in center_words:
        word_to_ids = vocab.get(center, vocab["[UNK]"])
        emb_emb =  token_emb[word_to_ids].reshape(1,-1)
        res = cosine_similarity(emb_emb, token_emb)
        ids = res.argsort()[0][::-1][1:n+1]
        context = [list(vocab)[id] for id in ids]
        print(f"{center} : {context}")

    
    pass

if __name__ == "__main__":
    with open("data/baiyang.pt", "rb") as f:
        vocab, token_emb = pickle.load(f)

    center_words = ["白杨树", "平凡"]

    top_related(center_words)
    pass