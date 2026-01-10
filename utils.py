from time import strptime, localtime, mktime

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch
from torchviz import make_dot
import seaborn as sns


def timer(timer_started_at):
    timer_started_at = strptime(timer_started_at, '%y%m%d%H%M%S')
    timer_started_at = mktime(timer_started_at)
    now = mktime(localtime())
    timer_in_hours = (now - timer_started_at) / 3600
    
    return round(timer_in_hours, 2)

class TransformerEmbedder:

    def __init__(self, model_name="paraphrase-distilroberta-base-v2", device="cuda:1"):            
        self.model = SentenceTransformer(model_name,device=device)
        self.model.eval()      
    
    def emb_sentence(self, sentences, normalize=False):
        x = self.model.encode(sentences)

        if not isinstance(sentences, list):            
            if normalize:
                x = x / np.linalg.norm(x, axis=0, ord=2)# + 1e-8
            return x.reshape(1,-1)
        else:
            if normalize:
                x = x / np.linalg.norm(x, axis=0, ord=2)# + 1e-8
            return x


def filtering_prototypes(x_real, y_real, classes=101, prots_per_class=23):

    def chunks(lista, n):
        for i in range(0, len(lista), n):
            yield lista[i:i + n]

    def count_unique_elements(elements):
        x = set(elements)
        e = {}
        for i in x:
            e[i] = 0
        for i in elements:
            e[i] += 1
        max = -1
        v = -1
        for i in e.keys():
            if e[i] > max:
                max = e[i]
                v = i
        return v, e

    kmeans = KMeans(init="k-means++", n_clusters=classes, n_init=4, random_state=0)
    kmeans.fit(x_real)
    preds = kmeans.predict(x_real)      
    ck = list(chunks(preds, prots_per_class))
    i = 0
    y = []
    x = []
    for c in ck:
        v, _ = count_unique_elements(c)
        for e in c:
            if e == v:
                y.append(y_real[i])
                x.append(x_real[i,:])
            i = i + 1
    return np.asarray(x), np.asarray(y)


def print_confusion_matrix(y_test, y_pred, classes, w=24,h=16,d=70, show=False, save=True, absolute_values=True, file="results/cm.pdf", plot_name="", font_size=16):    
    
    classes = [c.replace("_"," ") for c in list(classes.keys())]
    
    np.set_printoptions(precision=3)
    plt.figure(figsize=(w, h), dpi=d)
    plt.rcParams.update({'font.size': font_size})
    data = {
        'Ocorreu': y_test,
        'Predito': y_pred
    }
    df = pd.DataFrame(data, columns=['Ocorreu','Predito'])
    if absolute_values:
        conf = pd.crosstab(df['Ocorreu'], df['Predito'], rownames=['Ocorreu'], colnames=['Predito'])
        sns_plot = sns.heatmap(conf, annot=True, fmt="d", annot_kws={"size":font_size}, cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes, cbar=False)
    else:
        conf = pd.crosstab(df['Ocorreu'], df['Predito'], rownames=['Ocorreu'], colnames=['Predito'], normalize=True)
        sns_plot = sns.heatmap(conf, annot=False, annot_kws={"size":font_size}, cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=90)
    plt.xlabel(plot_name)
    plt.ylabel("")
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    if show:
        plt.show()
    if save:
        fig = sns_plot.get_figure()
        fig.savefig(file, bbox_inches='tight')