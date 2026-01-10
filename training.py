import warnings

warnings.filterwarnings("ignore")
import pickle
import random
import torch
from tqdm import tqdm
from time import localtime, strftime
from acnetdataset import AcnetCapDataset
from acnetdataset import acnet_collate
from model import JointEmbeddingModel
from utils import timer
import matplotlib.pyplot as plt


def training(training_triplet_data,
             validation_triplet_data,
             num_epochs=20,
             device_id=1,
             early_stop_after=5,             
             model_dir="tmp_models/",
             features_dir="data/bit_features/activitynetcaptions_features_bit",             
             vis_emb_dim=4096,
             joint_emb_dim=128,
             sent_emb_dim=768,
             drop_rate=0.15,
             batch_size=256,
             input_len=20,
             num_workers=8,
             lr=1e-4,
             wd=0.01):

    device = f"cuda:{device_id}"    
    curr_time = strftime('%y%m%d%H%M%S', localtime())    
    
    
    jembmodel = JointEmbeddingModel(input_vid_len=input_len,
                                    input_vid_feat_dim=vis_emb_dim,
                                    input_sent_feat_dim=sent_emb_dim,
                                    output_dim=joint_emb_dim,
                                    dropout_rate=drop_rate)

    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    optimizer = torch.optim.AdamW(jembmodel.parameters(), 
                                lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=wd)
        
    jembmodel.to(torch.device(device), non_blocking=True) 
    param_num = sum(p.numel() for p in jembmodel.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')

    history = {}
    history['train_loss'] = []
    history['val_loss'] = []
    best_metric = 1000000
    num_epoch_best_metric_unchanged = 0
    for epoch in range(0,num_epochs):                

        print(f"**** Epoch {epoch+1} ****")
        print(f'The best metrict was unchanged for {num_epoch_best_metric_unchanged} epochs.')
        print(f'Expected early stop @ {epoch+early_stop_after-num_epoch_best_metric_unchanged}')
        print(f'Current timer: {timer(curr_time)}')
        if num_epoch_best_metric_unchanged == early_stop_after:
            break                      

        acnet_training_dataset = AcnetCapDataset(training_triplet_data,
                                            input_len=input_len,
                                            video_dir=features_dir)    


        acnet_validation_dataset = AcnetCapDataset(validation_triplet_data,
                                                input_len=input_len,
                                                video_dir=features_dir)

        train_loader = torch.utils.data.DataLoader(acnet_training_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                drop_last=False,
                                                num_workers=num_workers, 
                                                pin_memory=True,
                                                collate_fn=acnet_collate)

        val_loader   = torch.utils.data.DataLoader(acnet_validation_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                num_workers=num_workers, 
                                                pin_memory=True,
                                                collate_fn=acnet_collate)

        # train loop
        train_losses = 0
        for vid, positive_txt, negative_txt in tqdm(train_loader):    
            vid = vid.to(device)
            positive_txt = positive_txt.to(device)
            negative_txt = negative_txt.to(device)
            optimizer.zero_grad(set_to_none=True)
            vid_emb_pos, positive_sent_emb = jembmodel(vid, positive_txt)
            vid_emb_neg, negative_sent_emb = jembmodel(vid, negative_txt)            
            vid_emb = vid_emb_neg
            loss_iter = criterion(vid_emb, positive_sent_emb, negative_sent_emb)
            loss_iter.backward()
            optimizer.step()
            train_losses += loss_iter.item()
        train_loss_total_norm = train_losses / len(train_loader)
        print(train_loss_total_norm)
        history['train_loss'].append(train_loss_total_norm)

        # val loop
        val_losses = 0
        for vid, positive_txt, negative_txt in tqdm(val_loader):
            vid = vid.to(device)
            positive_txt = positive_txt.to(device)
            negative_txt = negative_txt.to(device)
            with torch.no_grad():
                vid_emb_pos, positive_sent_emb = jembmodel(vid, positive_txt)
                vid_emb_neg, negative_sent_emb = jembmodel(vid, negative_txt)
                vid_emb = vid_emb_neg
            loss_iter = criterion(vid_emb,positive_sent_emb, negative_sent_emb)
            val_losses += loss_iter.item()

        val_loss_total_norm = val_losses / len(val_loader)
        print(val_loss_total_norm)
        history['val_loss'].append(val_loss_total_norm)
        torch.save(jembmodel, f"{model_dir}/lastmodel.pt")

        if val_loss_total_norm < best_metric:
            best_metric = val_loss_total_norm
            num_epoch_best_metric_unchanged = 0
            torch.save(jembmodel, f"{model_dir}/bestmodel.pt")
        else:
            num_epoch_best_metric_unchanged += 1

        loss = history["train_loss"]
        val_loss = history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.clf()
        plt.plot(epochs, loss, "r--", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.legend()
        plt.savefig("training.png")

    print(f'{curr_time}')
    print(f'best_metric: {best_metric}')


if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')     

    train_triplet = pickle.load(open("tmp_data/train_data.pkl","rb"))
    val_triplet = pickle.load(open("tmp_data/val_data.pkl","rb"))
    random.shuffle(train_triplet)
    
    training(train_triplet,
             val_triplet,
             num_epochs=200,  
             device_id=1,
             early_stop_after=20,
             model_dir="tmp_models/",
             features_dir="data/bit_features/activitynetcaptions_features_bit",
             vis_emb_dim=4096,
             joint_emb_dim=128,
             drop_rate=0.20,
             batch_size=128,             
             input_len=16,
             num_workers=8,
             lr=1e-4, 
             wd=1e-5)