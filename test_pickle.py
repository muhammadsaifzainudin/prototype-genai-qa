from pathlib import Path
import pickle
from glob import glob

bookFilePath = "starwars_small_sample_data.pickle"
files = glob(bookFilePath)
for fn in files:
   with open(fn,'rb') as f:
       part = pickle.load(f)
       for key, value in part.items():
           title = value['title'].strip()
           print(key, value)


