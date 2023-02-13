import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import csv

import pointnet2.data.data_utils as d_utils
from pointnet2.data.MuckpileDataLoader import MuckpileData
from pointnet2.data.ModelNet40Loader import ModelNet40Cls


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


lr_clip = 1e-5
bnm_clip = 1e-2


class PointNet2FragmentationSSG(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
      
        
        self.SA_modules.append(
            PointnetSAModule(
                npoint=25000,
                radius=0.05,
                nsample=64, #upper limit for number of points in ball
                mlp=[0, 16, 16, 32],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )   
    
       
        self.SA_modules.append(
            PointnetSAModule(
                npoint=10000,
                radius=0.1,
                nsample=64,
                mlp=[32, 32, 32, 64],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        
        self.SA_modules.append(
            PointnetSAModule(
                npoint=5000,
                radius=0.15,
                nsample=64,
                mlp=[64, 64, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        
        self.SA_modules.append(
            PointnetSAModule(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.3,
                nsample=64,
                mlp=[256, 256, 256, 512],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        
        
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.4,
                nsample=64,
                mlp=[512, 512, 512, 1024],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
   
       
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[1024, 1024, 1024, 2048], use_xyz=self.hparams["model.use_xyz"]
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 8), #8は粒度分布の数
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

    
    def training_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        
        '''
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)
        '''
        
      
        if logits.ndim == 1:
            labels = lebals.reshape(1, labels.size)
            logits = logits.reshape(1, logits.size)
        batch_size = logits.shape[0]
        
        softmax = nn.Softmax(dim=1)
        #loss = -torch.sum(labels * torch.log(softmax(logits + 1e-7))) / batch_size
        KLDivLoss = nn.KLDivLoss(reduction='sum')
        loss = KLDivLoss(softmax(logits + 1e-7).log(), labels + 1e-7)
        '''
        print(batch_size)
        print(logits)
        print(labels)
        print(loss)
        '''
        #print(softmax(logits + 1e-7))
        
        #accの代わりにヒストグラムの一致率みたいなのにするのもあり？
        with torch.no_grad():
            hist_logits = logits / torch.sum(logits, axis=0)
            hist_labels = labels / torch.sum(labels, axis=0)
            acc = (torch.sqrt(hist_logits * hist_labels)).float().mean() 
            #バタチャリア係数
            #一致率はバッチの平均をとっていることに注意

        #logとしてlossとaccを辞書にして保持
        log = dict(train_loss=loss, train_acc=acc)
        

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    
    def validation_step(self, batch, batch_idx):
        pc, labels = batch

        
        logits = self.forward(pc)
    
        if logits.ndim == 1:
            labels = lebals.reshape(1, labels.size)
            logits = logits.reshape(1, logits.size)
        batch_size = logits.shape[0]
    
        softmax = nn.Softmax(dim=1)
        #loss = -torch.sum(labels * torch.log(softmax(logits + 1e-7))) / batch_size
        KLDivLoss = nn.KLDivLoss(reduction='sum')
        #loss = -KLDivLoss(labels + 1e-7, softmax(logits + 1e-7))
        loss = KLDivLoss(softmax(logits + 1e-7).log(), labels + 1e-7)

        
        hist_logits = logits / torch.sum(logits, dim=0)
        hist_labels = labels / torch.sum(labels, dim=0)
        acc = (torch.sqrt(hist_logits * hist_labels)).float().mean() #バタチャリア係数
        
        '''
        logits = self.forward(pc)
        loss = F.cross_entropy(logits, labels)
        acc = (torch.argmax(logits, dim=1) == labels).float().mean()
        '''
        print(labels)
        print(softmax(logits))
        print('\n')
        


        with open('result.csv','a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(labels.tolist())
            writer.writerow(softmax(logits).tolist())
            writer.writerow("\n")
        f.close()
        
        return dict(val_loss=loss, val_acc=acc)

    
    def validation_end(self, outputs):

        reduced_outputs = {}
        if not outputs:
            return reduced_outputs 

        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    
    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["optimizer.lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["optimizer.decay_step"]
                )
            ),
            lr_clip / self.hparams["optimizer.lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["optimizer.bn_momentum"]
            * self.hparams["optimizer.bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["optimizer.decay_step"]
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["optimizer.lr"],
            weight_decay=self.hparams["optimizer.weight_decay"],
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def prepare_data(self):
        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudRotatePerturbation(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter(),
                d_utils.PointcloudRandomInputDropout(),
            ]
        )

        self.train_dset = MuckpileData(
            self.hparams["num_points"], transforms=train_transforms, train=True
        )
        self.val_dset = MuckpileData(
            self.hparams["num_points"], transforms=None, train=False
        )

    def _build_dataloader(self, dset, mode):
        return DataLoader(
            dset,
            batch_size=self.hparams["batch_size"],
            shuffle=mode == "train",
            num_workers=4,
            pin_memory=True,
            drop_last=mode == "train",
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
