import logging

import git
import hydra
from omegaconf import OmegaConf

import trainer

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="classic")
def main(config):
    logger.info("Single process training")

    repo = git.Repo(path=__file__, search_parent_directories=True)
    commit_id = repo.head.object.hexsha
    logger.info(f"git-commit: {commit_id}")

    additional_conifig = {
        "multi_node": False,
        "world_size": 1,
        "world_rank": 0,
        "local_rank": 0,
    }
    OmegaConf.set_struct(config, False)
    for k, v in additional_conifig.items():
        config[k] = v
    OmegaConf.set_struct(config, False)

    Trainer = getattr(trainer, config.trainer)
    t = Trainer(config)
    t.training()
    t.print_evaluate()


if __name__ == "__main__":
    # warnings.simplefilter("ignore")
    main()
