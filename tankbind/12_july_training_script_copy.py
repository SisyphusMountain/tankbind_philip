import sys; sys.path.append("/fs/pool/pool-marsot/")
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import hydra
from tankbind_philip.TankBind.tankbind.model import TankBind
from tankbind_philip.TankBind.tankbind.data import get_data, TankBindDataLoader
from torch.utils.data import RandomSampler


@hydra.main(version_base=None,
            config_path="/fs/pool/pool-marsot/tankbind_philip/TankBind/tankbind/configs")
def main(cfg):
    # logging
    wandb_logger = WandbLogger(project="TankBind",
                               save_dir=cfg.logging.save_dir,
                               )
    wandb_logger.log_hyperparams(cfg)
    # data
    train, train_after_warm_up, valid, test, all_pocket_test, info = get_data(addNoise=5.0, num_examples=cfg.data.num_examples)
    sampler = RandomSampler(train, replacement=True, num_samples=cfg.training.num_samples)
    train_loader = TankBindDataLoader(train, batch_size=cfg.training.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler, pin_memory=False, num_workers=cfg.compute.num_workers)
    sampler2 = RandomSampler(train_after_warm_up, replacement=True, num_samples=cfg.training.num_samples)
    train_after_warm_up_loader = TankBindDataLoader(train_after_warm_up, batch_size=cfg.training.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler2, pin_memory=False, num_workers=cfg.compute.num_workers)
    valid_loader = TankBindDataLoader(valid, batch_size=cfg.training.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=cfg.compute.num_workers)
    # losses
    criterion = torch.nn.MSELoss()

    def my_affinity_criterion(y_pred, y, mask, decoy_gap=1.0):
        affinity_loss = torch.zeros(y_pred.shape).to(y_pred.device)
        affinity_loss[mask] = (((y_pred - y)**2)[mask])
        affinity_loss[~mask] = (((y_pred - (y - decoy_gap)).relu())**2)[~mask]
        return affinity_loss.mean()
    affinity_criterion = my_affinity_criterion
    constant_affinity_coeff = 0.01
    # model
    model = TankBind(criterion=criterion,
                              affinity_criterion=affinity_criterion,
                              constant_affinity_coeff=constant_affinity_coeff,
                              )
    # training
    trainer = L.Trainer(max_epochs=cfg.training.max_epochs,
                         logger=wandb_logger,
                         log_every_n_steps=cfg.logging.log_every_n_steps,
                         accelerator="auto",
                         gradient_clip_val=cfg.training.gradient_clip_val,
                         check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
                         )
    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    main()