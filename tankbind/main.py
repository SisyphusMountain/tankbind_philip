import sys; sys.path.append("/fs/pool/pool-marsot/")
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
from tankbind_philip.TankBind.tankbind.model import TankBind
from tankbind_philip.TankBind.tankbind.data import get_data, TankBindDataLoader
from torch.utils.data import RandomSampler
import wandb



@hydra.main(version_base=None,
            config_path="/fs/pool/pool-marsot/tankbind_philip/TankBind/tankbind/configs")
def main(cfg):
    torch.set_float32_matmul_precision("high")
    # seeding
    L.pytorch.seed_everything(cfg.seed, workers=True)
    # logging

    wandb_logger = WandbLogger(project="TankBind",
                               save_dir=cfg.logging.save_dir,
                               )
    wandb_logger.log_hyperparams(cfg)
    # data
    train, train_after_warm_up, valid, test, all_pocket_test, info = get_data(addNoise=5.0, num_examples=cfg.data.num_examples, use_esm_embeddings=cfg.model.use_esm)
    # sampler = RandomSampler(train, replacement=True, num_samples=cfg.training.num_samples)
    # train_loader = TankBindDataLoader(train, batch_size=cfg.training.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler, pin_memory=False, num_workers=cfg.compute.num_workers)
    # sampler2 = RandomSampler(train_after_warm_up, replacement=True, num_samples=cfg.training.num_samples)
    train_after_warm_up_loader = TankBindDataLoader(train_after_warm_up, batch_size=cfg.training.batch_size, follow_batch=['x', 'compound_pair'], pin_memory=True, num_workers=cfg.compute.num_workers)
    valid_loader = TankBindDataLoader(valid, batch_size=cfg.training.batch_size, follow_batch=['x', 'compound_pair'], shuffle=cfg.debug.shuffle_val_dataset, pin_memory=False, num_workers=cfg.compute.num_workers)
    # losses
    def criterion(y_pred, y):
        if y_pred.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        else:
            return torch.nn.functional.mse_loss(y_pred, y)

    def my_affinity_criterion(y_pred, y, mask, decoy_gap=1.0):
        affinity_loss = torch.zeros(y_pred.shape).to(y_pred.device)
        affinity_loss[mask] = (((y_pred - y)**2)[mask])
        affinity_loss[~mask] = (((y_pred - (y - decoy_gap)).relu())**2)[~mask]
        return affinity_loss.mean()
    affinity_criterion = my_affinity_criterion
    checkpoint_callback = ModelCheckpoint(save_top_k=-1,
                                          auto_insert_metric_name=True,
                                          )
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    # model
    model = TankBind(criterion=criterion,
                              affinity_criterion=affinity_criterion,
                              constant_affinity_coeff=cfg.training.constant_affinity_coeff,
                              lr=cfg.training.lr,
                              scheduler=cfg.training.scheduler,
                              warmup_epochs=cfg.training.warmup_epochs,
                              max_epochs=cfg.training.max_epochs_scheduler,
                              use_distogram_loss=cfg.training.use_distogram_loss,
                              distogram_coefficient=cfg.training.distogram_coefficient,
                              n_trigonometry_module_stack=cfg.model.n_trigonometry_module_stack,
                              use_esm=cfg.model.use_esm,
                              )
    # training
    trainer = L.Trainer(max_epochs=cfg.training.max_epochs,
                         logger=wandb_logger,
                         log_every_n_steps=cfg.logging.log_every_n_steps,
                         accelerator="auto",
                         gradient_clip_val=cfg.training.gradient_clip_val,
                         check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
                         precision='32',
                         callbacks=[checkpoint_callback, lr_callback],
                         )
    trainer.fit(model, train_after_warm_up_loader, valid_loader)
    wandb.finish()
if __name__ == "__main__":
    main()