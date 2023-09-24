from store.store import Store
from config.env import DATA_PATH

print("starting the app...")

lang = "indobert"
file_name = "data-id.pickle.gz"

query = "Debat calon presiden memanas menjelang pemilihan umum."

em = Store(languange=lang)

# process and store embeddings
em.process_and_store_embeddings(DATA_PATH, key="text", save_name=file_name)

# load the pickle
d, v = em.load_pickle(file_name)

# query
results = em.query_vector(d, v, query, top_k=3)

print(results)