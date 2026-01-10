import os

from objects import Objects
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from dataset import load_random_classes
from dataset import load_class_sentences
from dataset import load_trueze_classes
from utils import TransformerEmbedder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
from zsardataset import ZSARDataset
from zsardataset import zsar_collate
from tqdm import tqdm
from utils import filtering_prototypes
import nltk


def encode_sentence(sentence, sbert, jointmodel, device=1, joint=True):    
    sent_emb = jointmodel.sentence_embedding
    sent_emb.to(device)
    x = sbert.encode(sentence)
    if joint:
        x = torch.from_numpy(np.expand_dims(np.asarray(x),axis=0)).to(device)
        x = sent_emb(x)
        x = x.data.cpu().numpy()
    else:
        x = np.expand_dims(x, axis=0)
    return x

def embedding_class_descriptions(z_names, sentences, max_len, sbert, jointmodel, device, max_sentences, filter=False, joint=True, mean_prototypes=False):
    y_real = []
    x_real = []
    for i, label in enumerate(z_names):
        ss = [] + sentences[label]            
        ss = [" ".join(s.split(" ")[0:max_len]) for s in ss]
        emb = encode_sentence(ss, sbert, jointmodel, device, joint)
        if mean_prototypes:
            emb = np.expand_dims(np.mean(emb,axis=1),axis=0)
        for j in range(emb.shape[1]):
            y_real.append(i)
            x_real.append(np.expand_dims(emb[0,j,:],axis=0))
    y_real = np.asarray(y_real)
    x_real = np.concatenate(x_real)
    if filter:
        x_real, y_real = filtering_prototypes(x_real, y_real, classes=len(z_names), prots_per_class=max_sentences+3)
    return x_real, y_real


def run_experiment(max_senteces=20,
                   min_len=3,
                   max_len=300,
                   n_neigh=3,
                   joint_model_path=None,
                   mode="visual",
                   device_id=0,
                   dataset_files = "data/splits/ucf101_dataset.json",
                   classes_descriptions = "data/texts/ucf101_texts_gemini/",
                   truze_splits_file = "data/splits/ucf101_truezsl_splits.json",
                   random_splits = True,
                   number_random_classes = 101,
                   num_workers=8,
                   runs = 1,
                   text_embedder="paraphrase-distilroberta-base-v2",
                   video_dir="data/bit_features/ucf101_features_bit_152"):
    
    device=f"cuda:{device_id}"
    
    start = time.time()
    sbert = SentenceTransformer(text_embedder,device=device)
    sbert.eval()
    jointmodel = torch.load(joint_model_path, weights_only=False)#, map_location=lambda storage, loc: storage.cuda(device_id))
    jointmodel.eval()

        
    embedder = TransformerEmbedder(model_name=text_embedder,device=device)        
    
    print("Loading object module...")
    objects = Objects(device=device)

    print("Loading class descriptions...")
    sentences = load_class_sentences(classes_descriptions, embedder, objects, min_len, max_len, max_senteces) 

    accs = []
    for r in range(runs):
        run_start = time.time()  
        print(80*"*")
        print(f"Summary - {r+1} of {runs}")
        print(80*"*")

        if random_splits:
            z_names = load_random_classes(truze_splits_file, number_random_classes, dataset_files)
        else:
            z_names = load_trueze_classes(truze_splits_file, "testing", dataset_files)
        
        print("Embedding action sentences...")
        x_real, y_real = embedding_class_descriptions(z_names, sentences, max_len, sbert, jointmodel, device, max_senteces, filter=False, joint=True, mean_prototypes=False)
        print(f"\t...{x_real.shape[0]} semantic prototypes\n\t...{len(z_names)} classes")


        print("Learning classifier...")
        classifier = KNeighborsClassifier(n_neighbors=n_neigh, metric="cosine")
        classifier.fit(x_real, y_real)

        print("Embedding videos...")
        data = []
        for id, z_name in enumerate(list(z_names.keys())):
            for fid in z_names[z_name]:
                data.append([fid, id])
        
        zsar_dataset = ZSARDataset(data,
                                    sbert, 
                                    jointmodel,
                                    objects,
                                    video_dir=video_dir,
                                    input_len=16,
                                    device_id=device_id,
                                    max_text_len=max_len,
                                    imagenet_file="data/bit_model/imagenet21k_wordnet_lemmas.txt",
                                    mode=mode)
        zsar_loader = torch.utils.data.DataLoader(zsar_dataset, 
                                                batch_size=32, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                num_workers=num_workers, 
                                                pin_memory=True,
                                                collate_fn=zsar_collate)
        
        class_ids = []
        samples = []        
        for c_ids, sam in tqdm(zsar_loader):
            class_ids.append(c_ids)
            samples.append(sam)
        
        class_ids = np.concatenate(class_ids)
        samples = np.concatenate(samples)
        preds = classifier.predict(samples)
        
        count = 0
        for i in range(len(class_ids)):
            if class_ids[i] == preds[i]:
                count += 1
        print(count)
        print(len(class_ids))


        acc = 100*accuracy_score(class_ids, preds)
        print(acc)
        accs.append(acc)
        
        if len(z_names) <= 51:
            from utils import print_confusion_matrix
            print_confusion_matrix(class_ids,
                                preds, 
                                z_names, w=24,h=16,d=70, 
                                show=False, save=True, absolute_values=True, 
                                file="cm.pdf", plot_name="")
        print(f"Run time elapsed: {time.time()-run_start}")
    accs = np.asarray(accs)
    print("Final report...")
    print(f"{np.mean(accs)}\t{np.std(accs)}")    
    print(f"\nTime elapsed: {time.time()-start}")
    return np.mean(accs), np.std(accs)


if __name__ == '__main__':        
    
    torch.multiprocessing.set_start_method('spawn')    
    #nltk.download('punkt_tab')
    #nltk.download('wordnet')
    device_id = 1
    num_workers = 0
    dataset_name="ucf101"
    #dataset_name="kinetics400"


    if dataset_name == "ucf101":
        run_experiment(max_senteces=1,
                       min_len=3,
                       max_len=300,
                       n_neigh=1,
                       joint_model_path=f"tmp_models/bestmodel.pt", 
                       mode="visual", # visual, object or visual_object
                       device_id=device_id,                       
                       dataset_files = "data/datasets/splits/ucf101_dataset.json",
                       classes_descriptions = "data/texts/ucf_101_texts_gemini/",
                       truze_splits_file = "data/datasets/splits/ucf101_truezsl2.0_splits.json",
                       #truze_splits_file = "data/splits/ucf101_hard_cases_splits.json",
                       random_splits = True,
                       runs = 1,
                       number_random_classes = 101,
                       num_workers=num_workers,
                       text_embedder="paraphrase-distilroberta-base-v2",                                        
                       video_dir="data/bit_features/ucf101_features_bit_152")
    
    if dataset_name == "kinetics400":
        run_experiment(max_senteces=1,
                       min_len=3,
                       max_len=300,
                       n_neigh=1,
                       joint_model_path=f"tmp_models/bestmodel.pt", 
                       mode="visual_object", # visual, object or visual_object
                       device_id=device_id,
                       dataset_files = "data/datasets/splits/kinetics400_dataset.json",                       
                       classes_descriptions="data/texts/kinetics_400_texts_gemini/",
                       truze_splits_file = "data/datasets/splits/kinetics400_splits",
                       random_splits = True,
                       runs = 1,                       
                       number_random_classes = 400, 
                       num_workers=num_workers,
                       text_embedder="paraphrase-distilroberta-base-v2",                      
                       video_dir="data/bit_features/kinetics_features_bit")
