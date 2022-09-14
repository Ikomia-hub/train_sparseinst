from detectron2.structures import BoxMode
from train_sparseinst.sparseinst import add_sparse_inst_config, COCOMaskEvaluator
import random
import copy
import pycocotools.mask as mask_util
from detectron2.structures import polygons_to_bitmask
import yaml
import numpy as np
import os
import itertools
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from detectron2.engine.hooks import BestCheckpointer
import requests
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.events import TensorboardXWriter
import logging

model_zoo = {"sparse_inst_r50vd_base": "1fjPFy35X2iJu3tYwVdAq4Bel82PfH5kx",
             "sparse_inst_r50_giam": "1pXU7Dsa1L7nUiLU9ULG2F6Pl5m5NEguL",
             "sparse_inst_r50_giam_soft": "1doterrG89SjmLxDyU8IhLYRGxVH69sR2",
             "sparse_inst_r50_giam_aug": "1MK8rO3qtA7vN9KVSBdp0VvZHCNq8-bvz",
             "sparse_inst_r50_dcn_giam_aug": "1qxdLRRHbIWEwRYn-NPPeCCk6fhBjc946",
             "sparse_inst_r50vd_giam_aug": "1dlamg7ych_BdWpPUCuiBXbwE0SXpsfGx",
             "sparse_inst_r50vd_dcn_giam_aug": "1clYPdCNrDNZLbmlAEJ7wjsrOLn1igOpT",
             "sparse_inst_r101_giam": "1EZZck-UNfom652iyDhdaGYbxS0MrO__z",
             "sparse_inst_r101_dcn_giam": "1shkFvyBmDlWRxl1ActD6VfZJTJYBGBjv",
             "sparse_inst_pvt_b1_giam": "13l9JgTz3sF6j3vSVHOOhAYJnCf-QuNe_",
             "sparse_inst_pvt_b2_li_giam": "1DFxQnFg_UL6kmMoNC4StUKo79RXVHyNF"}


def gdrive_download(file_id, dst_path):
    print("Downloading model")
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': file_id, "confirm": "t"}, stream=True)
    save_response_content(response, dst_path)
    print("Model downloaded")


def save_response_content(response, dst_path):
    CHUNK_SIZE = 32768
    with open(dst_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class Trainer(DefaultTrainer):

    def __init__(self, cfg, tb_dir, stop_train, log_metrics, update_progress):
        self.tensorboard_dir = tb_dir
        super().__init__(cfg)
        self.stop_train = stop_train
        self.log_metrics = log_metrics
        self.update_progress = update_progress

    def train(self):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.iter = self.start_iter

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    if self.stop_train():
                        break
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    self.update_progress()
                    if self.iter % 20 == 0:
                        self.log_metrics({name: value[0] for name, value in self.storage.latest().items()},
                                         step= self.iter)
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
                with open(os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth"), "w") as f:
                    f.write("")
                self.checkpointer.save("model_final")

    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(BestCheckpointer(eval_period=self.cfg.TEST.EVAL_PERIOD,
                                    checkpointer=self.checkpointer,
                                    val_metric="segm/AP",
                                    mode="max",
                                    file_prefix="model_best"))
        return ret

    def build_writers(self):
        return [TensorboardXWriter(self.tensorboard_dir)]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOMaskEvaluator(dataset_name, ("segm",), True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            # for transformer
            if "patch_embed" in key or "cls_token" in key:
                weight_decay = 0.0
            if "norm" in key:
                weight_decay = 0.0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full  model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, amsgrad=cfg.SOLVER.AMSGRAD
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.SPARSE_INST.DATASET_MAPPER == "SparseInstDatasetMapper":
            from train_sparseinst.sparseinst import SparseInstDatasetMapper
            mapper = SparseInstDatasetMapper(cfg, is_train=True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


def polygon_to_rle(polygon: list, shape=(520, 704)):
    '''
    polygon: a list of [x1, y1, x2, y2,....]
    shape: shape of bitmask
    Return: RLE type of mask
    '''
    mask = polygons_to_bitmask([np.asarray(polygon)], shape[0],
                               shape[1])  # add 0.25 can keep the pixels before and after the conversion unchanged
    rle = mask_util.encode(np.asfortranarray(mask))
    return rle


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


def setup_cfg(args, param):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.DATASETS.TRAIN = ("train_inst_seg_dataset",)
    cfg.DATASETS.TEST = ("test_inst_seg_dataset",)
    cfg.CLASS_NAMES = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.CLASS_NAMES)
    cfg.MODEL.RETINANET.NUM_CLASSES = len(cfg.CLASS_NAMES)
    cfg.MODEL.SPARSE_INST.NUM_CLASSES = len(cfg.CLASS_NAMES)
    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = len(cfg.CLASS_NAMES)
    cfg.MODEL.SEM_SEG_HEAD = len(cfg.CLASS_NAMES)
    old_batch_size = cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.IMS_PER_BATCH = param.cfg["batch_size"]
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * cfg.SOLVER.IMS_PER_BATCH / old_batch_size
    cfg.SOLVER.MAX_ITER = param.cfg["max_iter"]
    cfg.SOLVER.STEPS = (int(0.8 * cfg.SOLVER.MAX_ITER), int(0.9 * cfg.SOLVER.MAX_ITER))
    cfg.SOLVER.WARMUP_ITERS = int(0.1 * cfg.SOLVER.MAX_ITER)
    cfg.TEST.EVAL_PERIOD = min(param.cfg["max_iter"], param.cfg["eval_period"])
    cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.INPUT.CROP.ENABLED = False
    return cfg


def polygon_2_bbox(polygon):
    x = min(polygon[::2])
    y = min(polygon[1::2])
    w = max(polygon[::2]) - x
    h = max(polygon[1::2]) - y
    return [int(x), int(y), int(w), int(h)]


def rgb2mask(img, color2index):
    W = np.power(256, [[0], [1], [2]])
    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)
    for i, c in enumerate(values):
        try:
            mask[img_id == c] = color2index[tuple(img[img_id == c][0])]
        except:
            pass

    return mask


def register_dataset(dataset_name, images, metadata):
    DatasetCatalog.register(dataset_name, lambda: images)
    MetadataCatalog.get(dataset_name).thing_classes = [v for k, v in metadata["category_names"].items()]
    MetadataCatalog.get(dataset_name).ignore_label = 0
    MetadataCatalog.get(dataset_name).evaluator_type = "coco"


def to_list(list_or_ndarray):
    if isinstance(list_or_ndarray, list):
        if len(list_or_ndarray) > 0:
            return [to_list(e) for e in list_or_ndarray]
        else:
            return list_or_ndarray
    elif isinstance(list_or_ndarray, np.ndarray):
        return list_or_ndarray.tolist()


def register_datasets(data, split, had_bckgnd_class):
    data = copy.deepcopy(data)
    class_offset = 0 if had_bckgnd_class else 1

    for i, sample in enumerate(data["images"]):
        if "file_name" not in sample.keys():
            sample["file_name"] = sample.pop("filename")
            if "category_colors" in data["metadata"]:
                sample["category_colors"] = {color: i for i, color in enumerate(data["metadata"]["category_colors"])}
            sample["image_id"] = i

        for anno in sample["annotations"]:
            anno["bbox_mode"] = BoxMode.XYWH_ABS

            if "segmentation_poly" in anno:
                seg_poly = anno.pop("segmentation_poly")
                anno["segmentation"] = seg_poly
                anno["bbox"] = anno["bbox"]
                anno["category_id"] += class_offset

    random.seed(10)
    random.shuffle(data["images"])
    split_id = int(len(data["images"]) * split)
    train_imgs = data["images"][:split_id]
    test_imgs = data["images"][split_id:]
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    register_dataset("train_inst_seg_dataset", train_imgs, data["metadata"])
    register_dataset("test_inst_seg_dataset", test_imgs, data["metadata"])


def load_cwfid_dataset(folder_path):
    data = {"images": [], "metadata": {}}
    data["metadata"]["category_colors"] = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]
    data["metadata"]["category_names"] = {0: "background", 1: "crop", 2: "weed"}

    annotations = []
    labels = []
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("image.png"):
                images.append(os.path.join(root, file))

    for img in images:
        label = img.replace("images" + os.sep, "annotations" + os.sep)
        label = label.replace("image.png", "annotation.png")
        annot = label.replace(".png", ".yaml")
        labels.append(label)
        annotations.append(annot)

    id = 0
    for img, mask, annotation in zip(images, labels, annotations):
        id += 1
        record = {}
        record["id"] = id
        record["filename"] = img
        record["semantic_seg_masks_file"] = mask
        record["annotations"] = []
        record["height"] = 966
        record["width"] = 1296
        with open(annotation) as file:
            annotation_data = yaml.load(file, Loader=yaml.FullLoader)
            for e in annotation_data["annotation"]:
                polygon_annot = {}
                polygon_annot["category_id"] = list(data["metadata"]["category_names"].values()).index(e["type"])
                x = e["points"]["x"]
                y = e["points"]["y"]
                if type(x) == list:
                    points = np.zeros(2 * len(x))

                    points[0::2] = x
                    points[1::2] = y
                    polygon_annot["segmentation_poly"] = [points]
                    record["annotations"].append(polygon_annot)

        data["images"].append(record)
    return data
