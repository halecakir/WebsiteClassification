import pickle

from model import run

filename = "preprocessed.pickle"
with open(filename, 'rb') as target:
    data = pickle.load(target)

data  = [d for d in data if len(d[2]) > 10]
eliminated_categories = ["LOCALIZATION", "PERMISSIONS", "MEDIA", "CLOUD STORAGES", "FACE RECOGNITION"]
data = [d for d in data if d[1] in eliminated_categories]
run(data)
