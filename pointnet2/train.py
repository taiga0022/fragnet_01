import os
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


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


@hydra.main("config/config.yaml") #config読み込み？
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stop_callback = pl.callbacks.EarlyStopping(patience=1000) #学習を途中で止める
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",#モニターは何を見るかを指定　これはロスを監視
        mode="min",#ロスが小さいものを記録　精度だったらmaxに変更して記録したほうがいい
        save_top_k=2,#２つを記録
        filepath=os.path.join(
            cfg.task_model.name, "{epoch}-{val_loss:.2f}-{val_acc:.3f}"
        ),
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.distrib_backend,
    )


    # '''学習済モデル読み込み'''
    # checkpoint = "../thesis_J_moto/frag-ssg/epoch=837-val_loss=0.16-val_acc=0.949.ckpt" #学習済みモデル（ckptファイル）読み込み
    # state_dict = torch.load(checkpoint, map_location=device)
    # model.load_state_dict(state_dict["state_dict"])


    trainer.fit(model) #学習させる


if __name__ == "__main__":
    main()
