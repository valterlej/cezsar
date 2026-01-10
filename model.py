import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sentence_transformers import SentenceTransformer

class VisualEmbedding(nn.Module):

    def __init__(self,
                 vis_emb_dim=4096,
                 d_model=512,
                 joint_emb_dim=128,
                 drop_rate=0.10,                 
                 num_layers=1,
                 nhead=2,
                 max_len=20):        
        super(VisualEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.visual_dim_reducing = nn.Linear(vis_emb_dim,d_model)
        self.dropout1 = nn.Dropout(drop_rate)
        self.transf_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers)        
        self.dropout2 = nn.Dropout(drop_rate)
        self.video_joint_embedding = nn.Linear(d_model,joint_emb_dim)     

    def forward(self, video):
        vid = self.visual_dim_reducing(F.relu(video)) #Should I do apply F.relu()?
        vid = self.dropout1(vid)
        vid = self.transf_encoder(vid)
        vid = self.dropout2(vid)        
        vid = self.video_joint_embedding(F.relu(vid)) #Should I do apply F.relu()?
        vid = torch.mean(vid,axis=1) 
        return vid

class SentenceEmbedding(nn.Module):
    def __init__(self,
                 sent_emb_dim,
                 joint_emb_dim,
                 drop_rate=0.15,
                 encoder_name="paraphrase-distilroberta-base-v2",
                 encoder_freeze=True,
                 device="cuda:1"):
        super(SentenceEmbedding, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.sentence_joint_embedding = nn.Linear(sent_emb_dim, joint_emb_dim)

    def forward(self, sentence):       
        txt = self.dropout(sentence)
        txt = self.sentence_joint_embedding(txt)
        return txt

class JointEmbeddingModel(nn.Module):
    
    def __init__(self,                  
                 vis_emb_dim=4096,
                 d_model=1024,
                 joint_emb_dim=500,
                 sent_emb_dim=768,
                 drop_rate=0.15):

        super(JointEmbeddingModel, self).__init__()
        self.visual_embedding = VisualEmbedding(vis_emb_dim, d_model, joint_emb_dim, drop_rate)
        self.sentence_embedding = SentenceEmbedding(sent_emb_dim, joint_emb_dim, drop_rate)
        self.norm_vid = nn.LayerNorm(joint_emb_dim)
        self.norm_sent = nn.LayerNorm(joint_emb_dim)

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        
    def forward(self, video, sentence):        
        vid = self.norm_vid(self.visual_embedding(video))
        txt = self.norm_sent(self.sentence_embedding(sentence))
        return vid, txt