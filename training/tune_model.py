import optuna
import optunahub
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from sklearn.model_selection import KFold
import torch
from typing import List, Tuple
from pathlib import Path
from optuna.storages import RDBStorage
from optuna.storages import fail_stale_trials
import argparse
from model.lightning_model import SymmetryTransformer
from data.symmetry_dataset import SymmetryDataset
from model.pointNetpp_pretrained import get_model

"""
To run hyperparameter optimization:
    python3 tune_model.py --backbone pointnetpp/pointnet/None
Hyperparameters to tune can be adjusted in the objective() function.
"""

from configs.config import (
    RANDOM_STATE,       # int
    NUM_WORKERS,        # int
    NUM_FOLDS,          # int
    NUM_TRIALS,         # int
    LOG_DIR,            # str or Path
    CKPT_DIR,           # str or Path
    EARLY_STOP_PATIENCE,# int
    DATASET_PATH,        # str or Path
    NUM_LAYERS,         # int
)

# --------------------------------------------------------------------------- #
#                               load frozen backbone                          #
# --------------------------------------------------------------------------- #
def make_backbone():
    m = get_model(13, input_features=9)
    m.eval()
    return m


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def build_dataloaders(
    dataset: SymmetryDataset,
    indices: Tuple[List[int], List[int]],
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train / val DataLoaders for the given indices."""
    train_idx, val_idx = indices

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler   = torch.utils.data.SubsetRandomSampler(val_idx)

    collate_fn = SymmetryTransformer.custom_collate_fn

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_callbacks(trial_id: int, fold_id: int) -> List[pl.callbacks.Callback]:
    """Return early-stopping + checkpoint callbacks for one fold."""
    ckpt = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename=f"trial_{trial_id}_fold_{fold_id}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    es = EarlyStopping(monitor="val_loss", mode="min", patience=EARLY_STOP_PATIENCE)
    return [es, ckpt]


# --------------------------------------------------------------------------- #
#                               Optuna objective                              #
# --------------------------------------------------------------------------- #
def objective(trial: optuna.Trial) -> float:

    pl.seed_everything(RANDOM_STATE, workers=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 60 + f"\nStarting trial {trial.number}")

    # ---- sample hyper-parameters (keep the same style) -------------------------
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        epochs=trial.suggest_categorical("epochs", [10]),
        weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-4, log=True),
        hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128]),   # keep fixed (same framework)
        num_preds=trial.suggest_categorical("num_preds", [3]),      # keep fixed
        direction_loss_weight=trial.suggest_categorical("direction_loss_weight", [5.0, 10.0, 100.0]),  # keep fixed
        num_layers=trial.suggest_categorical("num_layers", [4, 8]),
        conf_loss_weight=trial.suggest_categorical("conf_loss_weight", [1.0]),
    )
    print(f"Trial {trial.number} parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # ---- dataset ---------------------------------------------------------------
    dataset_path = "pointNetpp_embeddings.npz"
    dataset = SymmetryDataset(dataset_path, ret_features=(args.backbone=="pointnetpp"))
    n = len(dataset)
    indices_all = list(range(n))

    # ---- cross-validation (same structure, now KFold) --------------------------
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices_all), start=1):
        print("-" * 60 + f"\nFold {fold}/{NUM_FOLDS}")

        # trainer & loaders --------------------------------------------------------
        train_loader, val_loader = build_dataloaders(dataset, (train_idx, val_idx), params["batch_size"])
        callbacks = build_callbacks(trial.number, fold)
        tb_logger = TensorBoardLogger(str(LOG_DIR), name=f"trial_{trial.number}_fold_{fold}")

        gpu_cnt = torch.cuda.device_count()
        if gpu_cnt == 0:
            print("No GPUs available, exiting trial early...")
            # Return a large loss so Optuna treats it as bad
            return float("inf")

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=tb_logger,
            callbacks=callbacks,
            max_epochs=params["epochs"],
            enable_checkpointing=True,
            default_root_dir=str(CKPT_DIR),
            enable_progress_bar=False,
            deterministic=True,
        )

        # model -------------------------------------------------------------------
        model = SymmetryTransformer(
            hidden_dim=params["hidden_dim"],
            nheads=4,
            num_encoder_layers=params["num_layers"],
            num_decoder_layers=params["num_layers"],
            input_dim=2,
            num_preds=params["num_preds"],
            learning_rate=params["learning_rate"],
            output_dim=4,
            direction_loss_weight=params["direction_loss_weight"],
            conf_loss_weight=params["conf_loss_weight"],
            dataset_path=dataset_path,
            backbone_model=args.backbone,
        )

        print(f"Training fold {fold}...")
        trainer.fit(model, train_loader, val_loader)

        # metric -------------------------------------------------------------------
        # use the logged val_loss; checkpoint best score is also val_loss (min)
        val_loss = trainer.callback_metrics["val_loss"].item() if "val_loss" in trainer.callback_metrics else float("inf")
        checkpoint = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)][0]
        best_val_loss = checkpoint.best_model_score.item() if checkpoint.best_model_score is not None else val_loss
        fold_losses.append(val_loss)
        print(f"Fold {fold} completed with validation loss: {val_loss:.6f}, best (ckpt) val_loss: {best_val_loss:.6f}")

        # cleanup
        del model, train_loader, val_loader, trainer
        torch.cuda.empty_cache()

    avg_loss = sum(fold_losses) / len(fold_losses)
    print(f"\nTrial {trial.number} completed with average validation loss: {avg_loss:.6f}")
    trial.set_user_attr('avg_val_loss', avg_loss)

    # IMPORTANT: we now minimize loss
    return avg_loss


# --------------------------------------------------------------------------- #
#                                   Driver                                    #
# --------------------------------------------------------------------------- #
def run_hyperparameter_optimization() -> None:

    STUDY_DB = Path("symmetry_hpo.sqlite").absolute()

    if not STUDY_DB.exists():
        print(f"Creating new study database at {STUDY_DB}")
    else:
        print(f"Using existing study database at {STUDY_DB}")

    storage = RDBStorage(
        url=f"sqlite:///{STUDY_DB}",
        engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=60,
        grace_period=120,
    )

    sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler(
        seed=RANDOM_STATE,
        constraints_func=None
    )

    study = optuna.create_study(
        direction="minimize",
        study_name="symmetry_hpo",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    fail_stale_trials(study)

    # print a light table of columns that exist for this problem
    df = study.trials_dataframe(attrs=("number", "state", "params", "user_attrs", "value"))
    cols = [c for c in df.columns if any(x in c for x in [
        "number", "state", "value", "params_learning_rate",
        "params_batch_size", "params_epochs", "params_weight_decay", "params_direction_loss_weight",
        "params_hidden_dim", "params_num_preds", "params_num_layers", "params_conf_loss_weight",
    ])]
    if not df.empty and cols:
        print(df[cols].to_string(index=False))

    try:
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        to_run = max(0, NUM_TRIALS - completed)
        study.optimize(objective, n_trials=to_run)

        best = study.best_trial
        print("=" * 60 + "\nBest trial")
        print(f"avg val_loss: {best.value:.6f}")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        out_file = Path("hyperparameter_optimization_results.csv")
        study.trials_dataframe().to_csv(out_file)
        print(f"Results written to {out_file.absolute()}")

    except Exception as exc:
        print(f"\nError during hyperparameter optimization: {str(exc)}")
        raise exc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="pointnetpp", choices=["None", "pointnet", "pointnetpp"], help="Which backbone to use")
    args = ap.parse_args()
    run_hyperparameter_optimization()
    
    
    
    


