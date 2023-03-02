import copy
import jsonargparse
import pytorch_lightning as pl


class GumbelTempScheduler(pl.callbacks.Callback):

    def __init__(self, init_temp, final_temp):
        self.init_temp = init_temp
        self.final_temp = final_temp

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        current_step = pl_module.global_step
        max_steps = trainer.max_steps
        pl_module.gumbel_temp = (1. - current_step / max_steps) * \
            (self.init_temp - self.final_temp) + self.final_temp
        assert pl_module.gumbel_temp >= self.final_temp


class EntropyWeightScheduler(pl.callbacks.Callback):

    def __init__(self, init_val, final_val):
        self.init_val = init_val
        self.final_val = final_val

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        current_step = pl_module.global_step
        max_steps = trainer.max_steps
        pl_module.entropy_weight = (
            1. - current_step / max_steps) * (self.init_val - self.final_val) + self.final_val


class LightningArgumentParser(jsonargparse.ArgumentParser):
    """
    Extension of jsonargparse.ArgumentParser that lets us parse datamodule, model and training args.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_datamodule_args(self, datamodule_obj: pl.LightningDataModule):
        """Add arguments from datamodule_obj to the parser
        Args:
            datamodule_obj (pl.LightningDataModule): Any LightningDataModule subclass
        """
        skip = {'num_duplicates'}
        self.add_method_arguments(
            datamodule_obj, '__init__', 'datamodule', as_group=True, skip=skip)

    def add_model_args(self, model_obj: pl.LightningModule):
        """Add arguments from model_obj to the parser
        Args:
            model_obj (pl.LightningModule): Any LightningModule subclass
        """
        skip = {'tasks', 'metrics_dict', 'task_channel_mapping',
                'edge_pos_weight', 'normals_centroids'}
        self.add_class_arguments(model_obj, 'model', as_group=True, skip=skip)

    def add_trainer_args(self):
        """Add Lightning's Trainer args to the parser.
        Args:
        """
        skip = {'logger', 'callbacks', 'max_steps',
                'max_epochs', 'automatic_optimization'}
        self.add_class_arguments(
            pl.Trainer, 'trainer', as_group=True, skip=skip)


def get_num_duplicates(gpus, num_nodes):
    # parse `gpus` arg as in pl
    gpu_ids = pl.utilities.device_parser.parse_gpu_ids(gpus)
    num_gpus = 0 if gpu_ids is None else len(gpu_ids)
    return max(1, num_gpus * num_nodes)


def get_exp_name(args):
    cargs = copy.deepcopy(args)
    ll = []
    ll.append(cargs.datamodule.dataset_name)
    ll.append(cargs.model.model_head)
    ll.append(cargs.model.model_backbone)
    ll.append(cargs.datamodule.tasks.replace(',', '-'))
    return '_'.join(ll)
