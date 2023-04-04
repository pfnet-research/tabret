import logging
from copy import copy

logger = logging.getLogger(__name__)


def get_diff_columns(datamodule):
    continuous_columns = copy(datamodule.continuous_columns)
    cat_cardinality_dict = copy(datamodule.cat_cardinality_dict)
    if hasattr(datamodule, "pre_continuous_columns"):
        continuous_columns = [col for col in continuous_columns if col not in datamodule.pre_continuous_columns]
        cat_cardinality_dict = {
            col: item for col, item in cat_cardinality_dict.items() if col not in datamodule.pre_cat_cardinality_dict
        }

    diff_columns = continuous_columns + list(cat_cardinality_dict.keys())
    logger.info(f"diff columns are {diff_columns}")
    return diff_columns, continuous_columns, cat_cardinality_dict
