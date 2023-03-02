import os
import jsonargparse
import pytorch_lightning as pl

from data.pl_data import DataModule
from model.search_pl_model import MultiTaskSearchModel
import utils_main


def main(args):

    pl.seed_everything(args.random_seed)

    # parse this in advance to adjust batch_size accordingly
    num_duplicates = utils_main.get_num_duplicates(
        args.trainer.gpus, args.trainer.num_nodes)

    dm = DataModule(num_duplicates=num_duplicates,
                    **vars(args.datamodule))
    dm.prepare_data()
    dm.setup()

    model = MultiTaskSearchModel(tasks=dm.tasks,
                                 task_channel_mapping=dm.task_channel_mapping,
                                 metrics_dict=dm.metrics_dict,
                                 edge_pos_weight=dm.edge_pos_weight,
                                 normals_centroids=dm.normals_centroids,
                                 **vars(args.model))

    logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(
        os.getcwd(), 'lightning_logs'), name=utils_main.get_exp_name(args))
    gumbel_scheduler = utils_main.GumbelTempScheduler(1.0, 0.05)
    # ~25% of iterations the entropy weight is negative
    entropy_scheduler = utils_main.EntropyWeightScheduler(-0.02, 0.06)
    trainer = pl.Trainer(logger=logger,
                         callbacks=[gumbel_scheduler, entropy_scheduler],
                         max_steps=model.iterations,
                         max_epochs=model.iterations,  # just so it's never reached
                         **vars(args.trainer))

    trainer.fit(model, datamodule=dm)
    trainer.test(model, test_dataloaders=dm.val_dataloader())


if __name__ == '__main__':
    parser = utils_main.LightningArgumentParser()
    parser.add_argument('--cfg', action=jsonargparse.ActionConfigFile, help='path to config file')
    parser.add_argument('--random_seed', default=0, type=int, help='random seed')
    parser.add_datamodule_args(DataModule)
    parser.add_model_args(MultiTaskSearchModel)
    parser.add_trainer_args()
    args = parser.parse_args()
    main(args)
