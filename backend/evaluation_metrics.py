import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import transformers

def cos_sim_tfidf(text, ground):
    # Convert the texts into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text, ground])

    return cosine_similarity(vectors)

def cos_sim_bert(text, ground):
    model = transformers.BertModel.from_pretrained('bert-base-uncased')

    encoding1 = model.encode(text, max_length=512)
    encoding2 = model.encode(ground, max_length=512)

    return numpy.dot(encoding1, encoding2) / (numpy.linalg.norm(encoding1) * numpy.linalg.norm(encoding2))

def iou_range(range1, range2):
    