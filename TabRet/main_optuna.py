import logging

import git
import hydra
from omegaconf import OmegaConf

import trainer
from trainer.optuna import OptunaSearch

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="optuna")
def main(config):
    print(f"gpu id: {config.gpuid}")

    multi_node = False
    world_size, world_rank, local_rank = 1, 0, config.gpuid
    logger.info("Single process training")

    if world_rank == 0:
        repo = git.Repo(path=__file__, search_parent_directories=True)
        commit_id = repo.head.object.hexsha
        logger.info(f"git-commit: {commit_id}")

    additional_conifig = {
        "multi_node": multi_node,
        "world_size": world_size,
        "world_rank": world_rank,
        "local_rank": local_rank,
    }
    print(additional_conifig)
    OmegaConf.set_struct(config, False)
    for k, v in additional_conifig.items():
        config[k] = v
    OmegaConf.set_struct(config, False)

    Trainer = getattr(trainer, config.fine_conf.trainer)
    os = OptunaSearch(Trainer, config)
    os.run(config.use_storage)


if __name__ == "__main__":
    # warnings.simplefilter("ignore")
    main()
