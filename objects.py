from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import bit_pytorch.models as models
import numpy as np
from nltk.corpus import wordnet
from itertools import chain



class Objects:
    
    def __init__(self, device="cuda:0", 
                 sentence_transformer_model="paraphrase-distilroberta-base-v2", 
                 bit_model="BiT-M-R152x2"):
        self.device = device
        self.sentence_transformer_model = sentence_transformer_model
        self.bit_model = bit_model
        self.imagenet_file = "data/bit_model/imagenet21k_wordnet_lemmas.txt"
        self.object_names, self.object_descriptions = self.load_object_classes_and_descriptions()
        self.embedding_model = SentenceTransformer(self.sentence_transformer_model, device=self.device)
        self.object_embeddings = self.embedding_model.encode(self.object_descriptions, normalize_embeddings=False, show_progress_bar=True, batch_size=128)
        self.obj_model = models.KNOWN_MODELS[self.bit_model]()
        self.obj_model.load_from(np.load("data/bit_model/BiT-M-R152x2.npz"))
        self.obj_model = self.obj_model.to(self.device)
        self.obj_model.eval()

    def _get_word_net_definition(self, words):
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

    def load_object_classes_and_descriptions(self):
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
        object_classes = [line[:-1] for line in open(self.imagenet_file,"r").readlines()]    
        obj_desc = []
        for o in object_classes:
            obj_desc.append(o.replace("_"," ") + " " +self._get_word_net_definition(o.replace(" ","").split(",")))
        return object_classes, obj_desc

    def _get_object_descriptions_from_video(self, vid_stack):
        try:
            x = vid_stack
            x = np.expand_dims(x, axis=-1)
            x = np.expand_dims(x, axis=-1)
            x = torch.from_numpy(x).to(self.device)
            logits = self.obj_model.head.conv(x)[...,0,0]
            soft = torch.nn.functional.softmax(logits, dim=1).data.cpu().numpy()        
            k = 3
            # top-k indexes (not sorted)
            part = np.argpartition(soft, -k, axis=1)[:, -k:]
            # sort k objects for each line in ascent order
            rows = np.arange(soft.shape[0])[:, None]
            order = np.argsort(soft[rows, part], axis=1)[:, ::-1]
            topk_idxs = part[rows, order]            
            unique_idx = list(set(list(chain(*topk_idxs))))

            return unique_idx
        except Exception as e:
            return []

    def get_object_sentences(self, vid, timestamp, sentence, max_len_vid, num_objects, include_sentence=True):
        try:
            stack = np.load(f"data/bit_features/activitynetcaptions_features_bit/{vid}.npy")
            sentence_embeding = self.embedding_model.encode(sentence, normalize_embeddings=False)
            vid_features = stack[int(timestamp[0]):int(timestamp[1]),:]
            n_frames, dim = vid_features.shape            
            if n_frames >= max_len_vid:
                vid_features = vid_features[0:max_len_vid,:]
            else:
                n = max_len_vid - n_frames
                n = np.zeros((n,dim), dtype="float32")
                vid_features = np.concatenate([vid_features, n])

            obj_ids = self._get_object_descriptions_from_video(vid_features)
            unique_embeddings = [self.object_embeddings[uidx] for uidx in obj_ids]
            unique_embeddings = np.asarray(unique_embeddings)
                
            similarities = self.embedding_model.similarity(sentence_embeding, unique_embeddings).cpu().detach().numpy()
                
            a = np.where((similarities > 0.01) & (similarities < 0.99), similarities, 0)
            part = np.argpartition(a, -num_objects, axis=1)[:, -num_objects:]
            rows = np.arange(a.shape[0])[:, None]
            order = np.argsort(a[rows, part], axis=1)[:, ::-1]
            topk_idxs = part[rows, order]

            sel_objs = [obj_ids[idx] for idx in topk_idxs[0]]
            sel_sents = [self.object_descriptions[sel_obj] for sel_obj in sel_objs]
            
            sentence = "" if not include_sentence else sentence
            sentence = " " + " ".join(sel_sents)
            return sentence
        except Exception as e:
            return sentence

    def get_names_and_descriptions(self):
        return self.object_names, self.object_descriptions