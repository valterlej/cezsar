import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import torch
import pickle
import random
import bit_pytorch.models as models
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import distance
from tqdm import tqdm
from nltk.corpus import wordnet


def get_word_net_definition(words):
    """Return a wordnet definition for a given set of words. 
    Follows 'Elaborative Rehearsal for Zero-Shot Action Recognition'

    Parameters
    ----------
    words: list
        lista de palavras a serem procuradas na wordnet
            
    Returns
    -------
    
    str
        a paragraph with a textual definition for all the input words
    """
    return_sentence = ""
    for word in words:
        result = wordnet.synsets(word)
        if not result:
            continue        
        sentence = ""
        for item in result:
            sentence += f"{item.definition()} . "
        return_sentence += sentence
    return return_sentence

def load_object_classes_and_descriptions(file):
    """Load object classes and their corresponding descriptions

    Parameters
    ----------
    file: str
        a file with ImageNet 21k lemmas (object labels)
            
    Returns
    -------
    
    list, list
        a list with all object class names
        a list with their corresponding descriptions from word net definitions
    """    
    object_classes = [line[:-1] for line in open(file,"r").readlines()]    
    obj_desc = []
    i = 0
    for o in tqdm(object_classes):
        obj_desc.append(o + " " +get_word_net_definition(o.replace(" ","").split(",")))
    return object_classes, obj_desc


def get_object_descriptions(bit_model,
                            vid_stack, 
                            obj_names, 
                            obj_descriptions, 
                            device="cuda:1", 
                            nobjs=3):
    try:
        x = vid_stack
        x = np.expand_dims(x, axis=-1)
        x = np.expand_dims(x, axis=-1)
        x = torch.from_numpy(x).to(device)
        logits = bit_model.head.conv(x)[...,0,0]
        logits = torch.mean(logits, 0, keepdim=True)
        soft = torch.nn.functional.softmax(logits, dim=1).data.cpu().numpy()       
        poslistdesc = []
        for _ in range(nobjs):
            pos = np.argmax(soft)
            soft[0,pos] = 0
            sent = [obj_names[pos].lower(), obj_descriptions[pos].lower()]
            poslistdesc.append(sent)
        return poslistdesc
    except Exception as e:
        return None

def pairing_data(acnet_files=["data/datasets/acnet_cap/train.json","data/datasets/acnet_cap/val_1.json","data/datasets/acnet_cap/val_2.json"],
                 acnet_data_file="tmp_data/paired_data.pkl",
                 save_file=True,
                 features_data_dir="data/bit_features/activitynetcaptions_features_bit",
                 device="cuda:1",                 
                 bit_model="BiT-M-R152x2",
                 bit_file_name="data/bit_model/BiT-M-R152x2.npz",
                 imagenet_file = "data/bit_model/imagenet21k_wordnet_lemmas.txt",
                 input_len=10,
                 n_positive_samples=3,
                 max_objects=20):
    """Load object classes and their corresponding descriptions

    Parameters
    ----------
    acnet_files: list
        a list containing the annotions from ActivityNet Captions dataset (train.json, val_1.json, val_2.json)
    acnet_data_file: str
        a file path where to save the pairs
    save_file: bool
        informs if it would save in hard disc
    features_data_dir: str
        a string containing the path for pre-computed features
    bit_model: str
        bit model name
    bit_file_name: str
        file name for the bit model
    imagenet_file: str
        path to the file containing the imagenet class names
    input_len: int
        size in seconds to consider from each video clip
    n_positive_samples: int
        number of positive random splits of input_len size to extract from each video clip
    max_objects: int
        number of most probable recognined objects in each positive sample
    
                
    Returns
    -------    
    dict
    a dictionary with acnet split as keys and a list of data for each split.
    Each element in this list has the structure: video_id, clip_start, clip_end, sentence, a list of object descriptions, seq_id
    """   
    bit_model = models.KNOWN_MODELS[f"{bit_model}"]()
    bit_model.load_from(np.load(bit_file_name))
    bit_model = bit_model.to(device)
    bit_model.eval()
    
    object_names, object_descriptions = load_object_classes_and_descriptions(imagenet_file)

    t_data = json.load(open(acnet_files[0], "r"))
    v1_data = json.load(open(acnet_files[1], "r"))
    v2_data = json.load(open(acnet_files[2], "r"))    
    data = {"train": t_data, "val1": v1_data, "val2": v2_data}
    acnet_data = {"train": [], "val1": [], "val2": []}
    c_loaded=0
    c_not_loaded=0
    for split in data.keys():
        idx = 0
        for yid in tqdm(data[split]):
            d = data[split][yid]
            timestamps = d["timestamps"]
            sentences = d["sentences"]
            for i in range(len(timestamps)):
                for _ in range(n_positive_samples):
                    try:                                        
                        vid_stack = np.load(features_data_dir+"/"+yid+".npy")                    
                        start = int(timestamps[i][0])
                        end = int(timestamps[i][1])
                        if start > end: start, end = end, start
                        if (end-input_len) > start: start= random.randint(start, (end-input_len))
                        end = start + input_len
                        vid_stack = vid_stack[start:end,:]
                        obj_descs = get_object_descriptions(bit_model, vid_stack, object_names, object_descriptions, device=device, nobjs=max_objects)
                        x = []
                        x.append(yid)
                        x.append(timestamps[i][0])
                        x.append(timestamps[i][1])
                        x.append(sentences[i].lower())
                        x.append(obj_descs)
                        x.append(idx)
                        idx = idx + 1                        
                        acnet_data[split].append(x)                         
                        c_loaded = c_loaded + 1 
                    except:
                        c_not_loaded = c_not_loaded + 1
    if save_file:        
        with open(acnet_data_file, "wb") as f:
            pickle.dump(acnet_data, f)    
        
    print(f"Loaded data from {c_loaded} video segments -- {100*c_loaded/(c_loaded+c_not_loaded)}%")
    print(f"Not loaded data from {c_not_loaded} video segments -- {100*c_not_loaded/(c_loaded+c_not_loaded)}%")
    count = 0
    for split in acnet_data.keys():
        count += len(acnet_data[split])
    print(f"Loaded {count} data pairs (video, positive_sentence)")  
    return acnet_data # [[video_id, caption, start, end, duration, caption_embedding, idx]]

def filtering_positive_descriptions(paired_data,
                                    min_len=5,
                                    max_len=20,
                                    ndescs=3,
                                    device="cuda:0",
                                    acnet_data_file="tmp_data/paired_filtered_data.pkl",
                                    save_file=True,
                                    encoder_name="paraphrase-distilroberta-base-v2"):

    
    sbert = SentenceTransformer(encoder_name,device=device)
    acnet_data = {"train": [], "val1": [], "val2": []}
    for split in paired_data.keys():
        idx = 0
        for paired in tqdm(paired_data[split]):
            positive = [paired[3]]
            pemb = sbert.encode(positive)
            object_descs = paired[4]
            object_sentences = []
            for obj_desc in object_descs:
                for desc in obj_desc:
                    obj_desc = desc.split(".")
                    object_sentences = object_sentences + obj_desc

            filtered = []
            for s in object_sentences:
                s = s.strip()
                s = s.split(" ")
                if len(s) > min_len:
                    s = s[0:max_len]
                    s = " ".join(s)
                    filtered.append(s)                
            filtered = list(set(filtered))
            obj_embs = sbert.encode(filtered)    

            top_k = min(ndescs, len(filtered))
            cos_scores = util.cos_sim(pemb, obj_embs)[0]
            top_results = torch.topk(cos_scores, k=top_k)    
            positive_object_sentences = []
            positive_object_embeddings = []
            for score, i in zip(top_results[0], top_results[1]):
                positive_object_sentences.append(filtered[i])    
                positive_object_embeddings.append(obj_embs[i].reshape(1,-1))
                        
            pos_embs = [pemb] + positive_object_embeddings
            sentences = [positive] + positive_object_sentences
            for sent, pemb in zip(sentences, pos_embs):
                x = []
                x.append(paired[0])
                x.append(paired[1])
                x.append(paired[2])
                x.append(sent)
                x.append(pemb)
                x.append(idx)
                idx = idx + 1                        
                acnet_data[split].append(x)
    if save_file:        
        with open(acnet_data_file, "wb") as f:
            pickle.dump(acnet_data, f)


def hardmining(acnet_data,
               num_negative_samples=10,
               max_sim_score=0.8,
               min_sim_score=0.0,
               acnet_data_file="tmp_data/data.pkl",
               save_file=True):
    
    
    new_acnet_data = []
    idx = 0        
    for _, d in enumerate(tqdm(acnet_data)):   
        try:
            for _ in range(num_negative_samples):
                data = []
                dist = -1
                while True:
                    p = random.randint(0,len(acnet_data)-1)
                    dist = distance.cosine(np.asarray(d[4]), np.asarray(acnet_data[p][4]))                    
                    if dist < max_sim_score and dist > min_sim_score:                                                
                        break                
                data.append(d[0]) # vid id
                data.append(d[1]) # vid timestamps - start
                data.append(d[2]) # vid timestamps - end
                data.append(d[4]) # positive sample 
                data.append(acnet_data[p][4]) # negative sample 
                data.append(idx)
                new_acnet_data.append(data)
                idx = idx + 1

        except Exception as e:
            print(e)
            import sys
            sys.exit()
            continue
    
    if save_file:        
        with open(acnet_data_file, "wb") as f:
            pickle.dump(new_acnet_data, f)
    print(f"Hard negative mining {len(new_acnet_data)} triplets (video, positive_sentence, negative_sentence)")
    return new_acnet_data

print("Pairing data")
paired_data = pairing_data(device="cuda:0",input_len=10, max_objects=5)
paired_data = pickle.load(open("tmp_data/paired_data.pkl", "rb"))

print("Filtering")
filtering_positive_descriptions(paired_data, save_file=True, acnet_data_file="tmp_data/paired_filtered_data.pkl", ndescs=2)
filtered_paired_data = pickle.load(open("tmp_data/paired_filtered_data.pkl","rb"))


train_data = hardmining(filtered_paired_data["train"]+filtered_paired_data["val1"]+filtered_paired_data["val2"], acnet_data_file="tmp_data/train_data.pkl", save_file=True, num_negative_samples=10)

print("Spliting")
random.shuffle(train_data)
t_data = train_data[:int(len(train_data)*0.7)]
v_data = train_data[int(len(train_data)*0.7):]
with open("tmp_data/train_data.pkl", "wb") as f:
    pickle.dump(t_data, f)
with open("tmp_data/val_data.pkl", "wb") as f:
    pickle.dump(v_data, f)