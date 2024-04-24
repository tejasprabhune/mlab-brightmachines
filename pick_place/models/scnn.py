import torch
import torchvision
import joblib
import cv2
import numpy as np
from sklearn.preprocessing import normalize

from transformers import ViTImageProcessor, ViTModel

class SiameseCNN(torch.nn.Module):
    def __init__(
        self,
        features="resnet",
        in_channels=2000,
        orb_in_channel = 100,
        out_channels=3,
        kmeans_model_path = '../../notebooks/kmeans_feature.pkl',
    ):
        super().__init__()
        self.features = features
        print(self.features)

        print(features)
        if self.features == 'resnet':
            self.resnet = torchvision.models.resnet18(pretrained=True)
            in_channels_resnet = in_channels
        elif self.features == 'orb':
            in_channels_resnet = orb_in_channel * 2
            # Ensure kmeans model is loaded
            #KMeans model path must be provided for ORB features
            self.kmeans = joblib.load(kmeans_model_path) if kmeans_model_path else None
            assert self.kmeans is not None
        elif self.features == 'resnetorb':
            in_channels_resnet = orb_in_channel * 2 + in_channels
            self.resnet = torchvision.models.resnet18(pretrained=True)
            self.kmeans = joblib.load(kmeans_model_path) if kmeans_model_path else None
            assert self.kmeans is not None
        elif self.features == 'akaze':
            self.akaze = cv2.AKAZE_create()
            self.akaze_enc = torch.nn.Linear(in_features=61, out_features=1)
            self.akaze_enc2 = torch.nn.Linear(in_features=40000, out_features=1000)
            in_channels_resnet = 2000
        elif self.features == 'resnetakaze':
            print("here")
            self.akaze = cv2.AKAZE_create()
            self.akaze_enc = torch.nn.Linear(in_features=61, out_features=1)
            self.akaze_enc2 = torch.nn.Linear(in_features=40000, out_features=1000)
            self.resnet = torchvision.models.resnet18(pretrained=True)
            in_channels_resnet = in_channels * 2
        elif self.features == "dino":
            print("here")
            self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
            self.dino = ViTModel.from_pretrained('facebook/dino-vitb8').to(0)
            self.dino_enc = torch.nn.Linear(in_features=768, out_features=1)
            #self.dino_enc2 = torch.nn.Linear(in_features=785, out_features=100)
            self.resnet = torchvision.models.resnet18(pretrained=True)
            in_channels_resnet = 1570 + in_channels
        elif self.features == "resnetdino":
            self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
            self.dino = ViTModel.from_pretrained('facebook/dino-vitb8').to(0)
            self.dino_enc = torch.nn.Linear(in_features=768, out_features=1)
            #self.dino_enc2 = torch.nn.Linear(in_features=785, out_features=100)
            in_channels_resnet = 1570

        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels_resnet, out_features=in_channels),
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.Linear(in_features=in_channels, out_features=out_channels)
            )
    
    def forward(self, x, ref=None):
        if self.features == 'resnet':
            x_emb = self.resnet(x)
            ref_emb = self.resnet(ref)
            full_emb = torch.cat([x_emb, ref_emb], dim=1)
        elif self.features == 'orb':
            full_embs = []
            for i in range(x.shape[0]):
                x_emb = self.orb_emb(x[i, 0:1, :, :]).to(0)
                ref_emb = self.orb_emb(ref[i, 0:1, :, :]).to(0)
                full_embs.append(torch.cat([x_emb, ref_emb], dim=1))
            full_emb = torch.cat(full_embs, dim=0)
        elif self.features == 'resnetorb':
            orb_x, orb_ref = self.apply_batch_grayscale_transform(x, ref, self.orb_emb)
            x_emb = self.resnet(x)
            ref_emb = self.resnet(ref)
            full_emb = torch.cat([x_emb, orb_x, ref_emb, orb_ref], dim=1)
        elif self.features == "akaze":
            akaze_x, akaze_ref = self.apply_batch_grayscale_transform(x, ref, self.akaze.detectAndCompute)
            full_emb = torch.cat([akaze_x, akaze_ref], dim=1)
        elif self.features == "resnetakaze":
            akaze_x, akaze_ref = self.apply_batch_grayscale_transform(x, ref, self.akaze.detectAndCompute)
            x_emb = self.resnet(x)
            ref_emb = self.resnet(ref)
            full_emb = torch.cat([x_emb, akaze_x, ref_emb, akaze_ref], dim=1)
        elif self.features == "dino":
            x_emb = self.processor(x, return_tensors="pt").to(0)
            ref_emb = self.processor(ref, return_tensors="pt").to(0)
            x_emb = self.dino(**x_emb)
            ref_emb = self.dino(**ref_emb)
            x_emb = x_emb.last_hidden_state
            ref_emb = ref_emb.last_hidden_state
            x_emb = self.dino_enc(x_emb)
            ref_emb = self.dino_enc(ref_emb)
            full_emb = torch.cat([x_emb, ref_emb], dim=1).to(0)
            full_emb = full_emb.squeeze(2)
        elif self.features == "resnetdino":
            x_emb = self.processor(x, return_tensors="pt").to(0)
            ref_emb = self.processor(ref, return_tensors="pt").to(0)
            x_emb = self.dino(**x_emb)
            ref_emb = self.dino(**ref_emb)
            x_emb = x_emb.last_hidden_state
            ref_emb = ref_emb.last_hidden_state
            x_emb = self.dino_enc(x_emb)
            ref_emb = self.dino_enc(ref_emb)
            x_emb = x_emb.squeeze(2)
            ref_emb = ref_emb.squeeze(2)

            resnet_x = self.resnet(x)
            resnet_ref = self.resnet(ref)
            full_emb = torch.cat([resnet_x, x_emb, resnet_ref, ref_emb], dim=1).to(0)
        else:
            raise ValueError("Invalid feature extraction method.")
        
        y_hat = self.fcn(full_emb)
        
        return y_hat
    
    def apply_batch_grayscale_transform(self, x, ref, fn):
        x = x.detach().cpu().numpy()
        ref = ref.detach().cpu().numpy()
        x_embs = []
        ref_embs = []
        for i in range(x.shape[0]):
            x_emb = torch.tensor(fn(x[i, 0, :, :], None)[1][:40000]).to(0)
            x_emb = self.akaze_enc(x_emb.float())
            x_emb = x_emb.transpose(0, 1)
            x_emb = self.akaze_enc2(x_emb)
            ref_emb = torch.tensor(fn(ref[i, 0, :, :], None)[1][:40000]).to(0)
            ref_emb = self.akaze_enc(ref_emb.float())
            ref_emb = ref_emb.transpose(0, 1)
            ref_emb = self.akaze_enc2(ref_emb)
            x_embs.append(x_emb)
            ref_embs.append(ref_emb)
        return torch.cat(x_embs, dim=0), torch.cat(ref_embs, dim=0)

    
    def orb_emb(self, x):
            # Assuming x is a tensor in the shape of (N, C, H, W)
            x_np = x.squeeze().detach().cpu().numpy()  #
            # (C, H, W) to (H, W, C)
            x_np = np.moveaxis(x_np, 0, -1)  
            #gray_image = cv2.cvtColor(x_np, cv2.COLOR_BGR2GRAY)
            gray_image = x_np
            orb = cv2.ORB_create()
            #print(gray_image.shape)
            gray_image = gray_image.astype(np.uint8)
            _, descriptors = orb.compute(gray_image, orb.detect(gray_image, None))
            #print(descriptors.shape, self.kmeans)
            vector_sample = self.visual_bag_word(descriptors, self.kmeans)
            return torch.tensor(vector_sample, dtype=torch.float).unsqueeze(0)
    
    def visual_bag_word(self, descriptors, kmeans):
        labels = kmeans.predict(descriptors)
        vector = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))[0]
        vector = normalize(vector[:, np.newaxis], axis=0).ravel()
        return vector

    def loss(y, y_hat):
        w = 0.99
        t = y[:2]
        t_hat = y_hat[:2]
        q = y[2]
        q_hat = y_hat[2]
        criterion = torch.nn.MSELoss()
        RMSY = torch.sqrt(criterion(y, y_hat))
        RMSR = torch.sqrt(criterion(q, q_hat))
        return w * RMSY + (1 - w) * RMSR        