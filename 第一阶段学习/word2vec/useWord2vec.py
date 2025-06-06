from sklearn.metrics.pairwise import cosine_similarity
import pickle

def similarities(tokenids_emb, vocab_dict,centers_list):

    for center_str in centers_list:
        centerId_int = vocab_dict[center_str]
        centerEmb_tensor = tokenids_emb[centerId_int].reshape(1,-1)
        similarity = cosine_similarity(centerEmb_tensor, tokenids_emb)
        ids = similarity.argsort()[0][::-1][1:11]
        similarity_words = [list(vocab_dict)[i] for i in ids]
        print(f"{center_str} : {similarity_words} ")
    

if __name__ == "__main__":
    file = "day12/data/baiyangshuCBOW.pt"
    with open(file, "rb") as f:
        tokenids_emb, vocab_dict = pickle.load(f)


    centers_list = ["白杨树", "哨兵"]

    similarities(tokenids_emb, vocab_dict,centers_list) 


    