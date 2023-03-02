import os
import jsonargparse
import pytorch_lightning as pl
import pytorch_lightning.loggers

from data.pl_data import DataModule
from model.pl_model import MultiTaskModel
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

    model = MultiTaskModel(tasks=dm.tasks,
                           task_channel_mapping=dm.task_channel_mapping,
                           metrics_dict=dm.metrics_dict,
                           edge_pos_weight=dm.edge_pos_weight,
                           normals_centroids=dm.normals_centroids,
                           **vars(args.model))

    logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(
        os.getcwd(), 'logger'), name=utils_main.get_exp_name(args))

    print("The logger directory is : {}".format(logger.log_dir))

    trainer = pl.Trainer(logger=logger,
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
    parser.add_model_args(MultiTaskModel)
    parser.add_trainer_args()
    args = parser.parse_args()
    main(args)
