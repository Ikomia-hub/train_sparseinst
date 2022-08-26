# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from train_sparseinst import update_path
from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
import copy
from ikomia.dnn import datasetio, dnntrain
from train_sparseinst.ik_utils import setup_cfg, register_datasets, Trainer, gdrive_download, model_zoo
from detectron2.config import CfgNode
from argparse import Namespace
import os
from distutils.util import strtobool
from datetime import datetime
from ikomia.core import config as ikcfg
# Your imports below


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainSparseinstParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["model_name"] = "sparse_inst_r50_giam_aug"
        self.cfg["batch_size"] = 2
        self.cfg["max_iter"] = 4000
        self.cfg["eval_period"] = 50
        self.cfg["split"] = 0.9
        self.cfg["conf_thr"] = 0.5
        self.cfg["expert_mode"] = False
        self.cfg["output_folder"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        self.cfg["custom_cfg"] = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["max_iter"] = int(param_map["max_iter"])
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["split"] = float(param_map["split"])
        self.cfg["conf_thr"] = float(param_map["conf_thr"])
        self.cfg["expert_mode"] = strtobool(param_map["expert_mode"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["custom_cfg"] = param_map["custom_cfg"]


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainSparseinst(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Variable to check if the training must be stopped by user
        self.output_folder = None
        self.stop_train = False
        self.advancement = None
        self.iters_done = None
        self.iters_todo = None

        # Create parameters class
        if param is None:
            self.setParam(TrainSparseinstParam())
        else:
            self.setParam(copy.deepcopy(param))

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        input = self.getInput(0)
        # Get parameters :
        param = self.getParam()
        plugin_folder = os.path.dirname(os.path.abspath(__file__))

        dataset = input.data
        register_datasets(dataset, param.cfg["split"], True)

        # Current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")

        # Output directory
        self.output_folder = os.path.join(param.cfg["output_folder"], str_datetime)

        # Tensorboard
        tb_logdir = os.path.join(ikcfg.main_cfg["tensorboard"]["log_uri"], str_datetime)

        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        if not param.cfg["expert_mode"]:
            model_folder = os.path.join(plugin_folder, "models")
            if not os.path.isdir(model_folder):
                os.mkdir(model_folder)
            model_weights = os.path.join(model_folder, param.cfg["model_name"]+".pth")
            if not os.path.isfile(model_weights):
                gdrive_download(model_zoo[param.cfg["model_name"]], model_weights)
            args = Namespace()
            args.opts = ["MODEL.WEIGHTS", model_weights]
            args.confidence_threshold = param.cfg["conf_thr"]
            args.input = ""
            args.config_file = os.path.join(plugin_folder, "configs", param.cfg["model_name"]+".yaml")
            cfg = setup_cfg(args, param)
        else:
            if os.path.isfile(param.cfg["custom_cfg"]):
                with open(param.cfg["custom_cfg"], 'r') as f:
                    cfg = CfgNode.load_cfg(f.read())
            else:
                print("File {} doesn't exist".format(param.cfg["custom_cfg"]))
                self.endTaskRun()
                return

        self.advancement = 0
        self.iters_done = 0
        self.iters_todo = cfg.SOLVER.MAX_ITER
        cfg.OUTPUT_DIR = self.output_folder
        with open(os.path.join(self.output_folder, str_datetime+"_config.yaml"), 'w') as f:
            f.write(cfg.dump())

        cfg.freeze()
        trainer = Trainer(cfg, tb_logdir, self.get_stop, self.log_metrics, self.update_progress)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train = True

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 100

    def update_progress(self):
        self.iters_done += 1
        steps = range(self.advancement, int(100 * self.iters_done / self.iters_todo))
        for step in steps:
            self.emitStepProgress()
            self.advancement += 1

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainSparseinstFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_sparseinst"
        self.info.shortDescription = "Train Sparseinst instance segmentation models"
        self.info.description = "Train Sparseinst instance segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.iconPath = "icons/sparseinst.png"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Cheng, Tianheng and Wang, Xinggang and Chen, Shaoyu and Zhang, Wenqiang and Zhang, " \
                            "Qian and Huang, Chang and Zhang, Zhaoxiang and Liu, Wenyu "
        self.info.article = "Sparse Instance Activation for Real-Time Instance Segmentation"
        self.info.journal = "Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)"
        self.info.year = 2022
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = "https://github.com/hustvl/SparseInst#readme"
        # Code source repository
        self.info.repository = "https://github.com/hustvl/SparseInst"
        # Keywords used for search
        self.info.keywords = "train, sparse, instance, segmentation, real-time, detectron2"

    def create(self, param=None):
        # Create process object
        return TrainSparseinst(self.info.name, param)
