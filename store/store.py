import json
import gzip
import pickle
from utils import formula as sim_formula
from embedding.indobert_em import embedder as id_bert_embedder
from embedding.openai_em import embedder as openai_embedder

class Store():
    def __init__(self, embedding):
        if embedding == "indobert":
            self.embedder = id_bert_embedder
        else:
            self.embedder = openai_embedder

    def process_and_store_embeddings(self, data, key="text", save_name: str="data.pickle.gz"):
        docs = []
        vecs = []

        with open(data, "r") as f:
            for line in f:
                d = json.loads(line)
                docs.append(d)
                v = self.embedder(d[key])
                vecs.append(v)

        # save documents and its vectors
        self.save_pickle(vecs, docs, save_name)

    @staticmethod
    def save_pickle(vectors, documents, storage_file):
        data = {"vectors": vectors, "documents": documents}
        with gzip.open(storage_file, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_pickle(storage_file):
        with gzip.open(storage_file, "rb") as f:
            data = pickle.load(f)
        vectors = data["vectors"]
        documents = data["documents"]
        return documents, vectors

    def query_vector(self, documents, vectors, query_text, top_k=5):
        query_vector = self.embedder(query_text)
        ranked_results, _ = sim_formula.hyper_svm_ranking_algorithm_sort(
            vectors, query_vector, top_k=top_k
        )
        return [documents[index] for index in ranked_results]