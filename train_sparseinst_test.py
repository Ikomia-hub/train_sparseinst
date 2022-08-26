import logging
from ikomia.core import task, ParamMap
from ikomia.utils.tests import run_for_test

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::train sparseinst =====")
    input_dataset = t.getInput(0)
    params = task.get_parameters(t)
    params["max_iter"] = 10
    params["batch_size"] = 1
    params["eval_period"] = 5
    params["split"] = 0.5
    task.set_parameters(t, params)
    input_dataset.load(data_dict["datasets"]["instance_segmentation"]["dataset_wgisd"])
    yield run_for_test(t)