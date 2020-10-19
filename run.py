from dataset import dataset


from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import demoji
from collections import Counter
import demoji
import torch
from torch import nn, optim
import numpy as np
from sklearn.model_selection import train_test_split

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(url, body):
    soup = BeautifulSoup(body, 'html.parser')
    if "github.com" in url:
            soup = soup.find("div", {"id": "readme"})
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return [t.strip() for t in visible_texts]

data = []
for d in dataset:
    url = d[2]
    html = urllib.request.urlopen(url).read()
    raw_sequences = text_from_html(url, html)
    seq = [d[0], d[1], " ".join(raw_sequences)]
    data.append(seq)

stop_words = stopwords.words("english")
lemmatizer=WordNetLemmatizer()
for d in data:
    #Remove emoloji
    d[2] = demoji.replace(d[2], repl="")
    #Convert lowercase
    d[2] = d[2].lower()
    #Remove numbers
    d[2] = re.sub(r"\d+", "", d[2])
    #Remove punctuation
    d[2] = d[2].translate(str.maketrans('', '', string.punctuation))
    #Remove whitespaces
    d[2] = d[2].strip()
    #Stop word elimination
    tokens = word_tokenize(d[2])
    tokens = [i for i in tokens if not i in stop_words]
    #Lemmatization
    tokens = [lemmatizer.lemmatize(i) for i in tokens]
    d[2] = tokens

with open("saved_data.txt", "w") as target:
    for d in data:
        target.write("{};;{};;{}\n".format(d[0], d[1], " ".join(d[2])))

class WebpageClassificationModel(nn.Module):
    def __init__(self, embeddings, n_features=10, hidden_size=100, n_classes=5, dropout_prob=0.5):
        super(WebpageClassificationModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))

        self.embed_to_hidden = nn.Linear(self.n_features * self.embed_size, self.hidden_size)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight.data)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight.data)
    
    def embedding_lookup(self, t):
        x = self.pretrained_embeddings(t)
        x = x.view(1, -1)
        return x
    
    def forward(self, t):
        lookup = self.embedding_lookup(t)
        embeddings = self.embed_to_hidden(lookup)
        relu = nn.ReLU()
        hidden = relu(embeddings)
        hidden = self.dropout(hidden)
        logits = self.hidden_to_logits(hidden)

        return logits


def load_and_preprocess_data(data, embedding_file, embed_size, most_common_n):
    c2i = {}
    w2i = {}
    w2i["NULL"] = 0
    for d in data:
        for w in d[2]:
            if w not in w2i:
                w2i[w] = len(w2i)
        if d[1] not in c2i:
            c2i[d[1]] = len(c2i)
    assert len(c2i) == 5
    word_vectors = {}
    for line in open(embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (len(w2i), emb_size)), dtype='float32')
    
    for token in w2i:
        i = w2i[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
    
    instances = []
    for d in data:
        c = Counter(d[2])
        indexes = []
        for w,_ in c.most_common(most_common_n):
            indexes.append(w2i[w])
        indexes += [w2i["NULL"]] * (most_common_n - len(indexes))
        instances.append([np.array(indexes), np.array([c2i[d[1]]])])
    return w2i, c2i, embeddings_matrix, instances

emb_size = 50
most_common_n_words = 10
embeding_file_path = "/Users/huseyinalecakir/Security/WebpageClassification/glove.6B/glove.6B.{}d.txt".format(emb_size)

w2i, c2i, embeddings_matrix, instances = load_and_preprocess_data(data, embeding_file_path, emb_size, most_common_n_words)



n_epochs = 10
lr = 0.0005
classifier = WebpageClassificationModel(embeddings_matrix)
optimizer = optim.Adam(classifier.parameters(), lr=lr)
loss_func = torch.nn.CrossEntropyLoss()

train_instances, test_instances = train_test_split(instances, test_size=0.33, random_state=42)

for epoch in range(n_epochs):
    print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
    loss_meter = AverageMeter()
    classifier.train()
    for i, (train_x, train_y) in enumerate(train_instances):
        optimizer.zero_grad()
        loss = 0. 
        train_x = torch.from_numpy(train_x).long()
        train_y = torch.from_numpy(train_y).long()
        logits = classifier(train_x)
        loss = loss_func(logits, train_y)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

classifier.eval()
for i, (test_x, test_y) in enumerate(test_instances):
    test_x = torch.from_numpy(test_x).long()
    pred = classifier(test_x)
    pred = pred.detach().numpy()
    pred = np.argmax(pred, 1)
    print("Prediction ", pred, " Target ", test_y)
