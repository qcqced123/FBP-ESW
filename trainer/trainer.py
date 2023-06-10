import gc
import math
import dataset_class.dataclass as dataset_class
import model.metric as model_metric
import model.metric_learning as metric_learning
import model.model as model_arch
from torch.utils.data import DataLoader
from dataset_class.data_preprocessing import *
from utils.helper import *
from trainer.trainer_utils import *
from model.metric import *
from tqdm.auto import tqdm


class NERTrainer:
    """
    Trainer Class for NER Task Pipeline (Stage 1)
    This class have 4 function:
        1) make_batch: make some input object related to batch (dataloader, dataframe)
        2) model_setting: make some input object related to model (model, criterion, optimizer, scheduler)
        3) train_fn: implement train stage per epoch
        4) valid_fn: implement valid stage per epoch
    Args:
        cfg: configuration.CFG
        generator: torch.Generator
    """
    def __init__(self, cfg: configuration.CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.df = load_data('./dataset_class/data_folder/final_converted_train_df.csv')
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self, fold: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, pd.DataFrame]:
        """ Make Batch Dataset for main train loop """
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)

        # Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, valid)
        # my_collate = MiniBatchCollate

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            # collate_fn=my_collate,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            # collate_fn=my_collate,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
        """ set train & validation options for main train loop """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))

        model.to(self.cfg.device)

        criterion = getattr(metric_learning, self.cfg.loss_fn)(self.cfg.reduction)
        val_metrics = getattr(model_metric, self.cfg.metrics)()
        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)
        return model, criterion, val_metrics, optimizer, lr_scheduler

    # Train Function
    def train_fn(self, loader_train, model, criterion, optimizer, lr_scheduler):
        """ Training Function """
        torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()
        model.train()
        for step, (inputs, labels) in enumerate(tqdm(loader_train)):  # Maybe need to append
            optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # input_ids, attention_mask, token_type_ids to GPU
            labels = labels.to(self.cfg.device)  # labels to GPU
            batch_size = self.cfg.batch_size
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                pred = model(inputs)
                loss = criterion(pred, labels)
            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach(), batch_size)

            if self.cfg.clipping_grad and ((step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1):
                scaler.unscale_(optimizer)  # unscale 풀어 보기
                torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

            gc.collect()

        train_loss = losses.avg.detach().cpu().numpy()
        return train_loss

    # Validation Function
    def valid_fn(self, loader_valid, model, val_metrics) -> float:
        """ Validation Functions """
        metrics = AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(loader_valid)):  # Maybe need to append
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)  # prompt to GPU

                val_batch_size = inputs.shape[0]
                labels = labels.to(self.cfg.device)  # labels to GPU
                val_pred = model(inputs)
                val_metric = val_metrics(cell_features, ranks)
                metrics.update(val_metric.detach(), 1)

        metric = metrics.avg.detach().cpu().numpy()
        gc.collect()
        return metric
