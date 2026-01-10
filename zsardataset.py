import torch
import os
import random
import numpy as np
from dataset import load_object_classes_and_descriptions
import bit_pytorch.models as models
import json

class ZSARDataset(torch.utils.data.Dataset):
  
    def __init__(self,                
                data,
                sbert, 
                jointmodel,
                objects = None,
                video_dir="/home/vlejunior/bitfeatures/ucf101/ucf101_features_bit_152",
                input_len=20,
                device_id=1,
                max_text_len=20,
                imagenet_file="data/bit_model/imagenet21k_wordnet_lemmas.txt",                
                mode="visual"):
        super().__init__()
        
        self.video_dir = video_dir
        self.data = data # [[vid_id, class_id], [vid_id, class_id]]
        self.input_len = input_len 
        self.max_text_len = max_text_len
        self.mode = mode
        self.sbert = sbert
        self.jointmodel = jointmodel
        self.device=f"cuda:{device_id}"

        self.vid_embedder = jointmodel.visual_embedding
        self.vid_embedder.to(self.device)
        self.vid_embedder.eval()     
        self.sentence_embedder = jointmodel.sentence_embedding
        self.sentence_embedder.to(self.device)
        self.sentence_embedder.eval()


        self.objAugmentation = objects
        
        self.object_names, self.object_descriptions = self.objAugmentation.get_names_and_descriptions()
        model = models.KNOWN_MODELS["BiT-M-R152x2"]()
        model.load_from(np.load("data/bit_model/BiT-M-R152x2.npz"))
        self.model = model.to(self.device)
        model.eval()
        

    def load_logit_data(self, vid_stack, obj_names, obj_descriptions, device="cuda:1", nobjs=3):
        try:
            x = vid_stack
            x = np.expand_dims(x, axis=-1)
            x = np.expand_dims(x, axis=-1)
            x = torch.from_numpy(x).to(device)
            logits = self.model.head.conv(x)[...,0,0]
            logits = torch.mean(logits, 0, keepdim=True) ### investigar melhor esse método        
            soft = torch.nn.functional.softmax(logits, dim=1).data.cpu().numpy()   
            poslistdesc = []
            for _ in range(nobjs):
                pos = np.argmax(soft)
                soft[0,pos] = 0
                poslistdesc.append(obj_names[pos] + " " + obj_descriptions[pos])
            sent = " ".join(poslistdesc)
            return sent
        except Exception as e:
            return None

    def encode_sentences(self, sentences, sbert, device=1, joint=True):    
            sentences = [" ".join(sentences)]
            x = sbert.encode(sentences)
            x = torch.from_numpy(np.asarray(x)).to(device)
            if joint:
                x = self.sentence_embedder(x)
                x = x.data.cpu().numpy()  
            else:
                x = x.data.cpu().numpy()
            return np.mean(x,axis=0)


    def encode_video(self, file_id, vstack_bit, sbert, device=1, mode="visual"):
                          
        if mode == "visual":            
            x = torch.from_numpy(np.expand_dims(vstack_bit, axis=0)).to(device)
            x = self.vid_embedder(x)
            x = x.data.cpu().numpy()
                            
        if mode == "object":

            obj_desc = self.load_logit_data(vstack_bit, self.object_names, self.object_descriptions, device=device)
            obj_desc = obj_desc.split(".")
            obj_desc = [" ".join(obj.split(" ")[0:self.max_text_len]) for obj in obj_desc]
            x = self.encode_sentences(obj_desc, sbert, device).reshape(1,-1)            
                
        if mode == "visual_object":
            x = torch.from_numpy(np.expand_dims(vstack_bit, axis=0)).to(device)
            x = self.vid_embedder(x)
            x = x.data.cpu().numpy()
            obj_desc = self.load_logit_data(vstack_bit, self.object_names, self.object_descriptions, device=device)
            obj_desc = obj_desc.split(".")
            obj_desc = [" ".join(obj.split(" ")[0:self.max_text_len]) for obj in obj_desc]                        
            y = self.encode_sentences(obj_desc, sbert, device).reshape(1,-1)         
            x = (0.8*x + 0.2*y) # equation 9 - paper

        return x


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        try:
            d = self.data[idx]
            video_id, class_id = d
            vid_dir = self.video_dir
            features_path = os.path.join(vid_dir, video_id+".npy")
            vstack_bit = np.load(features_path)    
            if vstack_bit.shape[0] < self.input_len:
                n_frames, dim = vstack_bit.shape            
                n = self.input_len - n_frames
                n = np.zeros((n,dim), dtype="float32")
                vstack_bit = np.concatenate([vstack_bit, n])                
            elif vstack_bit.shape[0] > self.input_len:
                vstack_bit = vstack_bit[0:self.input_len,:]
            
            vemb = self.encode_video(video_id, vstack_bit, self.sbert, self.device, mode=self.mode)           
            return vemb, class_id

        except Exception as e:
            print(e)
            return None, None
        
def zsar_collate(batch):        
    xl = []
    yl = []
    for x, y in batch:
        if x is not None and y is not None:
            xl.append(x)
            yl.append(y)
    return np.asarray(yl), np.concatenate(xl) # class_ids, samples
        