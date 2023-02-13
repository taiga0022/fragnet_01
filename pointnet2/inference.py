import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
import pointnet2_ops.pointnet2_modules
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import csv
import pointnet2.data.data_utils as d_utils
from pointnet2.data.MuckpileDataLoader import MuckpileData
from pointnet2.data.ModelNet40Loader import ModelNet40Cls
import os
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class PointNet2FragmentationSSG(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.Global_modules = nn.ModuleList()
        
        self.SA_modules.append(
            PointnetSAModule(
                npoint=num_point,
                radius=0.15,
                nsample=1024, #upper limit for number of points in ball
                #mlp=[0, 64, 128, 1024, 2048, 5000], # 大きい用
                mlp=[0, 32, 64, 128, 256, 1024], # r=0.15用
                #mlp=[0, 32, 64, 128], # 小さい用
                use_xyz=self.hparams["model.use_xyz"],
            )
        )   
        

        self.sharedMLP_layer =  nn.Sequential(
            # nn.Linear(5000, 2048),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, FRAG_DIM),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(64, FRAG_DIM),
            # nn.ReLU(),
        )

        self.softmax = nn.Softmax(dim=1)

        self.Global_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0,
                nsample=1, #upper limit for number of points in ball
                mlp=[0, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )   

        self.Global_modules.append(
            PointnetSAModule(
                mlp=[128, 512], use_xyz=self.hparams["model.use_xyz"]
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, FRAG_DIM*FRAG_DIM),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        # feature (B,N,C)を(B,C,N)に転置
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

        '''Local Module'''
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        
        new_features = (features.transpose(1, 2).contiguous()).reshape(pointcloud.size()[0]*num_point, -1)
        # features(B,C,N')を(B*N',C)に

        scores = self.sharedMLP_layer(new_features) 
        # shared_MLP

        new_scores = (scores.reshape(pointcloud.size()[0], num_point, FRAG_DIM)).transpose(1, 2).contiguous()
        # scores(B*N',C')を(B,C',N')に

        result = torch.sum(self.softmax(new_scores), 2) / num_point
        #result = self.softmax(torch.sum(new_scores, 2))
        # softmaxで正規化したscoreをsum state:(B,8,1)




        
        '''全体形状考慮'''
        '''Global Mudule'''
        xyz2, features2 = self._break_up_pc(pointcloud)
        
        for module in self.Global_modules:
            xyz2, features2 = module(xyz2, features2)

        global_mat = self.fc_layer(features2.squeeze(-1)).reshape(pointcloud.size()[0], FRAG_DIM, FRAG_DIM)

        '''aggregate Local and Global'''
        global_score = torch.bmm(global_mat, result.unsqueeze(-1))
        # (B,8,8)*(B,8,1)
    
        result = self.softmax(global_score.squeeze(-1))
        




        return result



    
    def training_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        if logits.ndim == 1:
            labels = labels.reshape(1, labels.size)
            logits = logits.reshape(1, logits.size)
        batch_size = logits.shape[0]
        
        softmax = nn.Softmax(dim=1)
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        loss = KLDivLoss((logits + 1e-7).log(), labels + 1e-7)
        
        #print(logits.shape) (B,8)
        #print(labels.shape) (B,8)

        with torch.no_grad():
            dis = torch.sum((torch.sqrt(logits * labels)).float(), 1) 
            acc = (1 / (dis+1)).mean()
            #バタチャリア係数
            #一致率はバッチの平均をとっていることに注意

            peak_acc = (torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).float().mean()
            #ピーク正解率

       
        #logとしてlossとaccを辞書にして保持
        log = dict(train_loss=loss, train_acc=dis.mean(), train_peak_acc=peak_acc)
        

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=dis.mean()))

    
    
    #in:データbatch out:モデル通した結果のloss
    def validation_step(self, batch, batch_idx):
        pc, labels = batch
        
        logits = self.forward(pc)
        if logits.ndim == 1:
            labels = labels.reshape(1, labels.size)
            logits = logits.reshape(1, logits.size)
        batch_size = logits.shape[0]

        softmax = nn.Softmax(dim=1)
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        loss = KLDivLoss((logits + 1e-7).log(), labels + 1e-7)

        dis = torch.sum((torch.sqrt(logits * labels)).float(), 1) 
        acc = (1 / (dis+1)).mean()
        #バタチャリア係数
        #一致率はバッチの平均をとっていることに注意

        peak_acc = (torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).float().mean()
        #ピーク正解率
        

        with open('result.csv','a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(labels.tolist())
            writer.writerow(softmax(logits).tolist())
            writer.writerow("\n")
        f.close()

      
        return dict(val_loss=loss, val_acc=dis.mean(), val_peak_acc=peak_acc)

    
    # エポックごとのvalidationデータに対する処理
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

    # 最適化手法・スケジューラの設定
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

    #データの準備（ダウンロード等）
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

    #学習用data_loader
    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    #val用data_loader
    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")



def prepare_data(self):
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

#テスト用data_loader
def test_dataloader(self):
    return self._build_dataloader(self.val_dset, mode="val")


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)




@hydra.main("config/config.yaml")
def main(cfg):
    checkpoint = "../Mthesis_test/frag-ssg/epoch=1991-val_loss=0.14-val_acc=0.955.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2FragmentationSSG()
    model.to(device)
    model.eval()
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict["state_dict"])
    print(model)

    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
    )
    trainer.test()

if __name__ == "__main__":
    main()
