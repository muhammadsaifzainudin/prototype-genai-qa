from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import pickle
import os
from tqdm import tqdm
from glob import glob

model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)

index_name = "book_wookieepedia_mpnet"
url = f"http://localhost:9200"
db = ElasticVectorSearch(embedding=hf, elasticsearch_url=url, index_name=index_name)


bookFilePath = "starwars_small_sample_data.pickle"
files = glob(bookFilePath)
for fn in files:
    batchtext = []
    with open(fn,'rb') as f:
       part = pickle.load(f)
       for ix, (key, value) in tqdm(enumerate(part.items()), total=len(part)):
           paragraphs = value['paragraph']
           for p in paragraphs:
               batchtext.append(p)
       db.from_texts(batchtext,
                     embedding=hf,
                     elasticsearch_url=url,
                     index_name=index_name)