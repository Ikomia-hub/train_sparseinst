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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_sparseinst.train_sparseinst_process import TrainSparseinstParam
from train_sparseinst.ik_utils import model_zoo

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class TrainSparseinstWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainSparseinstParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        self.combo_model = pyqtutils.append_combo(self.gridLayout, "Model name")
        for model in model_zoo:
            self.combo_model.addItem(model)
        self.combo_model.setCurrentText(self.parameters.cfg["model_name"])
        self.spin_max_iter = pyqtutils.append_spin(self.gridLayout, "Max iteration", self.parameters.cfg["max_iter"])
        self.spin_batch_size = pyqtutils.append_spin(self.gridLayout, "Batch size", self.parameters.cfg["batch_size"])
        self.spin_eval_period = pyqtutils.append_spin(self.gridLayout, "Evaluation period",
                                                      self.parameters.cfg["eval_period"])
        self.double_split = pyqtutils.append_double_spin(self.gridLayout, "Split train/test ratio",
                                                         self.parameters.cfg["split"], min=0, max=1, step=0.01,
                                                         decimals=2)
        self.double_conf_thr = pyqtutils.append_double_spin(self.gridLayout, "Confidence threshold",
                                                            self.parameters.cfg["conf_thr"], min=0, max=1, step=0.01,
                                                            decimals=2)
        self.browse_output_folder = pyqtutils.append_browse_file(self.gridLayout, label="Output folder",
                                                                 path=self.parameters.cfg["output_folder"],
                                                                 tooltip="Select folder",
                                                                 mode=QFileDialog.Directory)
        self.check_expert_mode = pyqtutils.append_check(self.gridLayout, "Expert mode",
                                                        self.parameters.cfg["expert_mode"])
        self.check_expert_mode.stateChanged.connect(self.on_check_expert_mode)
        self.browse_custom_cfg = pyqtutils.append_browse_file(self.gridLayout, path=self.parameters.cfg["custom_cfg"],
                                                              label="Custom cfg",
                                                              tooltip="Select file",
                                                              mode=QFileDialog.ExistingFile)
        self.on_check_expert_mode(self.check_expert_mode.isChecked())
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_check_expert_mode(self, b):
        expert = self.check_expert_mode.isChecked()
        self.combo_model.setVisible(not expert)
        self.spin_max_iter.setVisible(not expert)
        self.spin_batch_size.setVisible(not expert)
        self.spin_eval_period.setVisible(not expert)
        self.double_split.setVisible(not expert)
        self.double_conf_thr.setVisible(not expert)

        self.browse_custom_cfg.setVisible(expert)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.cfg["model_name"] = self.combo_model.currentText()
        self.parameters.cfg["expert_mode"] = self.check_expert_mode.isChecked()
        self.parameters.cfg["max_iter"] = self.spin_max_iter.value()
        self.parameters.cfg["batch_size"] = self.spin_batch_size.value()
        self.parameters.cfg["eval_period"] = self.spin_eval_period.value()
        self.parameters.cfg["split"] = self.double_split.value()
        self.parameters.cfg["conf_thr"] = self.double_conf_thr.value()
        self.parameters.cfg["output_folder"] = self.browse_output_folder.path
        self.parameters.cfg["custom_cfg"] = self.browse_custom_cfg.path
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainSparseinstWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_sparseinst"

    def create(self, param):
        # Create widget object
        return TrainSparseinstWidget(param, None)
