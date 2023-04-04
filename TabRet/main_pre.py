import logging

import git
import hydra
from omegaconf import OmegaConf

import trainer

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="pre")
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

    OmegaConf.set_struct(config, False)
    for k, v in additional_conifig.items():
        config[k] = v
    OmegaConf.set_struct(config, False)

    Trainer = getattr(trainer, config.pre_conf.trainer)
    t = Trainer(config)
    t.training()


if __name__ == "__main__":
    # warnings.simplefilter("ignore")
    main()
