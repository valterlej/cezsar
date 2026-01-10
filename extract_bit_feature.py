#!/usr/bin/bash
import os
import time
import argparse
import jsonlines
import json
import shutil
import glob

import cv2
from PIL import Image

from progressbar import ProgressBar
from utils import process_file_name

import numpy as np
import decord

import torch
import torchvision as tv
import torch.multiprocessing as mp

import bit_pytorch.models as models


def load_video(file):
  frames = []
  cap = cv2.VideoCapture(file)    
  if (cap.isOpened()== False):  
      print("Error opening video file: "+file) 
   
  while(cap.isOpened()):             
      ret, frame = cap.read() 
      if ret == True:                     
          frames.append(frame)
      else:  
          break   
  cap.release()    
  return np.asarray(frames)


class VideoDataset(torch.utils.data.Dataset):
  def __init__(self, video_dir, video_list, num_frames_per_video, tmp_dir, frame_rate=25):
    super().__init__()
    self.video_dir = video_dir
    self.video_list = video_list
    self.num_frames_per_video = num_frames_per_video
    self.tmp_dir = tmp_dir
    self.frame_rate = frame_rate

    self.transform = tv.transforms.Compose([
      tv.transforms.Resize(256),
      tv.transforms.CenterCrop(224),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

  def __len__(self):
    return len(self.video_list)

  def __getitem__(self, idx):
    
    try:

      videoname = process_file_name(os.path.join(self.video_dir, self.video_list[idx]+".mp4"))
      tmp_videoname = os.path.join(self.tmp_dir, self.video_list[idx]+".mp4")
      ffmpeg_tmp_videoname = process_file_name(tmp_videoname)
      cmd = f"ffmpeg -hide_banner -loglevel panic -y -i {videoname} -filter:v fps=fps={self.frame_rate} {ffmpeg_tmp_videoname}"
      os.system(cmd)
      vid = load_video(tmp_videoname)
      nframes = vid.shape[0]
      
      if nframes <= self.num_frames_per_video:
        idxs = np.arange(0, nframes).astype(np.int32)
      else:
        idxs = np.linspace(0, nframes-1, nframes // self.num_frames_per_video) #### one frame at each num_frames_per_video
        idxs = np.round(idxs).astype(np.int32)
        
      images = []
      for k in idxs:
        frame = Image.fromarray(vid[k])
        frame = self.transform(frame)
        images.append(frame)
      images = torch.stack(images, 0)

      name = self.video_list[idx]

      try:
          os.remove(tmp_videoname)
      except OSError as e:
          pass


      return name, images
    except Exception as e:
      print(self.video_list[idx])
      print(e)
      return "", []


def extract_image_features(proc_id, log_queue, video_list, device, args):
    print(f"Process{proc_id} starts to extract features using GPU{device}")

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:%d" % device)
    torch.set_grad_enabled(False)

    print(f"\tProcess{proc_id}: loading model from {args.model}.npz")
    model = models.KNOWN_MODELS[args.model]()
    model.load_from(np.load(
      os.path.join(args.model_dir, f"{args.model}.npz")))
    model = model.to(device)
    model.eval()

    video_dataset = VideoDataset(
      args.video_dir, video_list, args.num_frames_per_video, args.tmp_dir, args.frame_rate)    
  
    video_loader = torch.utils.data.DataLoader(
      video_dataset, batch_size=1, shuffle=False, 
      drop_last=False, num_workers=args.num_data_workers, pin_memory=True)
  
    output_dir = args.output_dir
    output_feat_dir = args.output_feat_dir
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    for name, images in video_loader:

      if name == "" or len(images) == 0:
        continue  
    
      name = name[0]
      images = images[0]

      if os.path.exists(os.path.join(output_dir, name+'.npy')):
        log_queue.put(name)
        continue
      

      chunck_size = 20
      chuncks = torch.split(images, chunck_size)
      logits_list = []
      features_list = []
      for chunck in chuncks:
        images = chunck.to(device)
        x = model.root(images)      
        x = model.body(x)
        x = model.head.gn(x)
        x = model.head.relu(x)      
        fts = model.head.avg(x)
        features = fts      
        logits = model.head.conv(fts)[...,0,0]
        logits_list.append(logits)
        features_list.append(features)
    
      logits = torch.cat(logits_list, 0)  
      features = torch.cat(features_list, 0)

      logits = torch.mean(logits, 0, keepdim=True).data.cpu().numpy()
    
      with open(os.path.join(output_dir, name+'.npy'), 'wb') as outf:
        np.save(outf, logits)
    
      features = features.reshape(-1,4096)

      with open(os.path.join(output_feat_dir, name+'.npy'), 'wb') as outf:
        np.save(outf, features.data.cpu().numpy())

      log_queue.put(name)

    log_queue.put(None)

  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_dir')
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--tmp_dir', default="data/tmp")
    parser.add_argument('--frame_rate', type=int, default=25)
        
    parser.add_argument('--num_frames_per_video', type=int, default=25)
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--output_feat_dir', required=True)

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_data_workers', type=int, default=2)
    args = parser.parse_args()

    video_list, todo_video_list = [], []


    if args.video_dir[:-1]!="/":
        args.video_dir = args.video_dir+"/"

    files = glob.glob(args.video_dir+"*.mp4")

    for item in files:
          item = item.split("/")[-1][:-4]
          video_list.append(item)
          if not os.path.exists(os.path.join(args.output_dir, item+'.npy')):
            todo_video_list.append(item)    
    print('total videos: %d, todo videos: %d' % (len(video_list), len(todo_video_list)))

    if len(todo_video_list) > 0:
      mp.set_start_method('spawn')
      log_queue = mp.Queue()

      num_workers = min(len(todo_video_list), args.num_workers)
      avg_videos_per_worker = len(todo_video_list) // num_workers
      num_gpus = 1#torch.cuda.device_count()
      assert num_gpus > 0, 'No GPU available'

      processes = []
      for i in range(num_workers):
        sidx = avg_videos_per_worker * i
        eidx = None if i == num_workers - 1 else sidx + avg_videos_per_worker
        device = i % num_gpus

        process = mp.Process(
          target=extract_image_features, args=(i, log_queue, todo_video_list[sidx: eidx], device, args)
        )
        process.start()
        processes.append(process)

      progress_bar = ProgressBar(max_value=len(todo_video_list))
      progress_bar.start()

      num_finished_workers, num_finished_files = 0, 0
      while num_finished_workers < num_workers:
        res = log_queue.get()
        if res is None:
          num_finished_workers += 1
        else:
          num_finished_files += 1
          progress_bar.update(num_finished_files)

      progress_bar.finish()

      for i in range(num_workers):
        processes[i].join()


if __name__ == "__main__":
  main()  

### command example
# python extract_bit_feature_new.py --model BiT-M-R152x2 --model_dir data/models/ --video_dir /media/valter/Arquivos/activitynetcaptions/ --output_dir /media/valter/Arquivos/activitynetcaptions_bit/ --output_feat_dir /media/valter/Arquivos/activitynetcaptions_features_bit/ --num_workers 2 --num_frames_per_video 25