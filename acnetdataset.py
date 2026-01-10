import torch
import os
import numpy as np

class AcnetCapDataset(torch.utils.data.Dataset):
  
    def __init__(self,                
                 data,
                 video_dir="data/bitfeatures/activitynetcaptions_features_bit",
                 input_len=20):
        super().__init__()
        
        self.video_dir = video_dir
        self.data = data
        self.input_len = input_len 
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        try:
            d = self.data[idx]
            video_id, start, end, positive_caption_embedding, negative_caption_embedding, idx = d
                       
            vid_dir = self.video_dir
            
            features_path = os.path.join(vid_dir, video_id+".npy")
            features = np.load(features_path)            
            
            vid_features = features[int(start):int(end),:]
            n_frames, dim = vid_features.shape            

            if n_frames >= self.input_len:
                vid_features = vid_features[0:self.input_len,:]
            else:
                n = self.input_len - n_frames
                n = np.zeros((n,dim), dtype="float32")
                vid_features = np.concatenate([vid_features, n])                
            
            positive_caption_embedding = positive_caption_embedding.reshape(1,-1)
            negative_caption_embedding = negative_caption_embedding.reshape(1,-1)           

            return vid_features, positive_caption_embedding, negative_caption_embedding

        except Exception as e:
            return None, None, None
        
def acnet_collate(batch):        
    xl = []
    yl = []
    zl = []
    for x, y, z in batch:
        if x is not None and y is not None and z is not None:
            xl.append(np.expand_dims(np.asarray(x), axis=0))
            yl.append(np.asarray(y))
            zl.append(np.asarray(z))
    xl = np.concatenate(xl)
    yl = np.concatenate(yl)
    zl = np.concatenate(zl)
    return torch.from_numpy(xl), torch.from_numpy(yl), torch.from_numpy(zl)
        