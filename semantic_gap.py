import os
import sys
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from dataset import load_trueze_classes, load_class_sentences
from objects import Objects
from utils import TransformerEmbedder
import time
from zsar import encode_sentence
from zsardataset import ZSARDataset
from zsardataset import zsar_collate
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from utils import filtering_prototypes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from utils import print_confusion_matrix
from zsar import embedding_class_descriptions
from sklearn.preprocessing import normalize

def plot_tsne_data(x, y, label_names, file_name="semantic_gap.pdf", show_legend=False, n_prots=None, w=14,h=14, perplexity=3):
    x = TSNE(n_components=2, init='random',verbose=1, perplexity=perplexity).fit_transform(x,y)#100
    labels = [i for i in label_names]
    y_names = [labels[i] for i in y]
    if n_prots is None:
        d = {'tsne-2d-one': x[:,0], 'tsne-2d-two': x[:,1]}
        df = pd.DataFrame(data=d)
        plt.figure(figsize=(h,w))
        ax = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=y_names,
            #palette=sns.color_palette("hls", len(labels)),
            data=df,
            #legend="full",
            #alpha=0.3
        )
    else:
        plt.figure(figsize=(w,h))
        #plt.legend(markerscale=2)
        ds = {'tsne-2d-one': x[:-n_prots,0], 'tsne-2d-two': x[:-n_prots,1]}
        dfs = pd.DataFrame(data=ds)        
        ax = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=y_names[:-n_prots],
            data=dfs,
            s=50,
        )
        dp = {'tsne-2d-one': x[-n_prots:,0], 'tsne-2d-two': x[-n_prots:,1]}
        dfp = pd.DataFrame(data=dp)
        ax = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=y_names[-n_prots:],
            data=dfp,
            marker="*",
            s=1500,
        )
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    if not show_legend:
        ax.get_legend().remove()
    plt.axis('off')
    plt.savefig(file_name, bbox_inches='tight')



def embedding_videos(z_names, sbert, jointmodel, video_dir, device_id, max_len, observer_files, mode, max_videos_per_class=20, objects=None):
    print(f"Embedding videos...{mode}")
    data = []
    for id, z_name in enumerate(list(z_names.keys())):
        for i, fid in enumerate(z_names[z_name]):
            if i == max_videos_per_class-1:
                break
            data.append([fid, id])
        
    zsar_dataset = ZSARDataset(data,
                               sbert, 
                               jointmodel,
                               objects=objects,
                               video_dir=video_dir,
                               input_len=15,
                               device_id=device_id,
                               max_text_len=max_len,
                               imagenet_file="data/pretrainedmodels/imagenet21k_wordnet_lemmas.txt",
                               observer_files=observer_files,
                               mode=mode)
    zsar_loader = torch.utils.data.DataLoader(zsar_dataset, 
                                              batch_size=32, 
                                              shuffle=False, 
                                              drop_last=False, 
                                              num_workers=0, 
                                              pin_memory=True,
                                              collate_fn=zsar_collate)
    class_ids = []
    samples = []        
    for c_ids, sam in tqdm(zsar_loader):
        class_ids.append(c_ids)
        samples.append(sam)
        
    class_ids = np.concatenate(class_ids)
    samples = np.concatenate(samples)
    return samples, class_ids


def embedding_videos_resnet(z_names, video_dir, max_videos_per_class=20):
    
    class_ids = []
    samples = []
    for id, z_name in enumerate(list(z_names.keys())):
        for i, fid in enumerate(z_names[z_name]):
            if i == max_videos_per_class-1:
                break            
            features_path = os.path.join(video_dir, fid+".npy")
            features = np.load(features_path)            
            vid_stack = features
            vid_stack = np.mean(vid_stack, axis=0).reshape(1,-1)
            class_ids.append(id)
            samples.append(vid_stack)             
    #class_ids = np.concatenate(class_ids)
    class_ids = np.asarray(class_ids)
    samples = np.concatenate(samples)
    return samples, class_ids



def foo(text_embedder="paraphrase-distilroberta-base-v2",
        min_len=3,
        max_len=300,
        device_id=0,
        max_sentences=20,
        max_videos=200,
        n_neigh=3,
        joint=True,
        joint_model_path=f"tmp_models/23_nov/bestmodel.pt",
        truze_splits_file = "data/splits/ucf101_selected_classes_sg.json",
        dataset_files = "data/splits/ucf101_dataset.json",
        classes_descriptions = "data/texts/ucf101_texts/",
        video_dir="/home/vlejunior/bitfeatures/ucf101/ucf101_features_bit_152",
        observer_files=["data/observers/bit/ucf101_resnet152_4096.json","data/observers/bit/ucf101_resnet_objectemb_768.json","data/observers/bit/ucf101_resnet152_objectemb_1268.json"],
        modes=["visual", "object", "captions", "visual_object", "visual_object_captions"]):
    
    if "captions_zsarcap" in modes and (len(modes) > 1 or joint == True):
        sys.exit("Invalid configuration.")        
    elif ("captions" in modes or "visual_object_captions" in modes) and joint == False:
        sys.exit("Invalid configuration.")

    start = time.time()
    device=f"cuda:{device_id}"
    sbert = SentenceTransformer(text_embedder,device=device)
    sbert.eval()
    jointmodel = torch.load(joint_model_path,weights_only=False)
    jointmodel.eval()
    embedder = TransformerEmbedder(model_name=text_embedder,device=device) 
    z_names = load_trueze_classes(truze_splits_file, "testing", dataset_files)

    print("Loading object module...")
    objects = Objects(device=device)    
    sentences = load_class_sentences(classes_descriptions, embedder, objects, min_len, max_len, max_sentences)



    """
    Semantic embedding
    """
    x_real, y_real = embedding_class_descriptions(z_names, sentences, max_len, sbert, jointmodel, device, max_sentences, filter=False, joint=joint, mean_prototypes=False)
    print(f"\t...{x_real.shape[0]} semantic prototypes\n\t...{len(z_names)} classes")
    name = "cezsar"
    if "captions_zsarcap" in modes:
        name = "zsarcap"    
    plot_tsne_data(x_real, y_real, z_names, file_name=f"plots/prototypes_{name}.pdf", w=12,h=10)    


    """
    Learning a classifier
    """
    classifier = KNeighborsClassifier(n_neighbors=n_neigh, metric="cosine")
    print(x_real.shape)
    classifier.fit(x_real, y_real)
    

    """
    Visual embedding
    """
    for i, m in enumerate(modes):
        print(f"Mode: {m}")
        samples, class_ids = embedding_videos(z_names, sbert, jointmodel, video_dir, device_id, max_len, observer_files, m, max_videos_per_class=max_videos, objects=objects)                
        samples = normalize(samples,norm='l2')
        print(samples.shape)
        preds = classifier.predict(samples)       
        acc = 100*accuracy_score(class_ids, preds)        
        print(acc)
        print_confusion_matrix(class_ids, preds, z_names,font_size=32, w=20,h=16,d=70, show=False, save=True, absolute_values=True, file=f"plots/{i}_confmatrix_{m}.pdf", plot_name="")
        # group prototypes
        samples = np.concatenate([samples, x_real])
        samples = normalize(samples,norm='l2')
        class_ids = np.concatenate([class_ids,y_real])                
        plot_tsne_data(samples, class_ids, z_names, file_name=f"plots/{i}_semantic_gap_{m}.pdf",n_prots=len(z_names)*max_sentences,show_legend=False, w=13,h=13, perplexity=50) 
    
    
    
    print("ResNet features loading")
    samples, class_ids = embedding_videos_resnet(z_names, video_dir, max_videos)
    plot_tsne_data(samples, class_ids, z_names, file_name=f"plots/{i}_resnetemb.pdf",show_legend=False, w=13,h=13, perplexity=50) 


    print(f"Time elapsed {time.time()-start}")

if __name__ == '__main__':
    #foo(max_sentences=10, modes=["visual", "object", "captions", "visual_object", "visual_object_captions"])
    """
    foo(max_sentences=17,
        max_videos=200, 
        modes=["visual", "visual_object_captions"],
        truze_splits_file = "data/splits/ucf101_selected_classes_sg_1.json",
        joint=True)
    """
    """    
    foo(max_sentences=20,
        max_videos=2000,
        device_id=0,
        n_neigh=3,
        #modes=["captions_zsarcap"],
        #joint=False,
        #modes=["visual","captions","visual_object_captions"],        
        modes=["visual_object_captions"],#,"visual_object_captions"],
        joint=True,
        #truze_splits_file = "data/splits/ucf101_truezsl2.0_splits.json",
        #truze_splits_file = "data/splits/ucf101_selected_classes_sg_2.json",
        truze_splits_file = "data/splits/ucf101_selected_classes_sg_4.json"
        )
    """

    
    foo(max_sentences=1,
        max_videos=2000,
        device_id=0,
        n_neigh=1,
        joint_model_path=f"old_scripts/tmp_models/23_nov/bestmodel.pt",        
        classes_descriptions = "data/texts/ucf_101_texts_gemini/",
        #modes=["captions_zsarcap"],
        #joint=False,
        #modes=["visual","captions","visual_object_captions"],        
        modes=["visual_object"],#,"visual_object_captions"],
        joint=True,
        video_dir="/media/valterlej/dados/bitfeatures/ucf101/ucf101_features_bit_152",
        #truze_splits_file = "data/splits/ucf101_truezsl2.0_splits.json",
        #truze_splits_file = "data/splits/ucf101_selected_classes_sg_2.json",
        truze_splits_file = "data/splits/ucf101_selected_classes_sg_1.json"
    )


    '''
    foo(max_sentences=10,
        max_videos=2000,
        device_id=0,
        n_neigh=1,
        joint_model_path=f"old_scripts/tmp_models/23_nov/bestmodel.pt",        
        classes_descriptions = "data/texts/ucf101_texts/",
        modes=["captions_zsarcap"],
        #joint=False,
        #modes=["visual","captions","visual_object_captions"],        
        joint=False,
        video_dir="/media/valterlej/dados/bitfeatures/ucf101/ucf101_features_bit_152",
        #truze_splits_file = "data/splits/ucf101_truezsl2.0_splits.json",
        #truze_splits_file = "data/splits/ucf101_selected_classes_sg_2.json",
        truze_splits_file = "data/splits/ucf101_selected_classes_sg_1.json"
    )
    '''