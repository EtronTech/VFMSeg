from xmuda.models.xmuda_arch import Net2DSeg, Net3DSeg
from xmuda.models.xmuda_fusion_arch import Net2D3DFusionSeg
from xmuda.models.metric import SegIoU
from xmuda.models.vfm_fusion import LogitsFusion


def build_model_2d(cfg,model_name = ' '):
    model = Net2DSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES,
                     backbone_2d=cfg.MODEL_2D.TYPE,
                     backbone_2d_kwargs=cfg.MODEL_2D[cfg.MODEL_2D.TYPE],
                     dual_head=cfg.MODEL_2D.DUAL_HEAD
                     )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_2d' + model_name)
    return model, train_metric


def build_model_3d(cfg,model_name = ' '):
    model = Net3DSeg(num_classes=cfg.MODEL_3D.NUM_CLASSES,
                     backbone_3d=cfg.MODEL_3D.TYPE,
                     backbone_3d_kwargs=cfg.MODEL_3D[cfg.MODEL_3D.TYPE],
                     dual_head=cfg.MODEL_3D.DUAL_HEAD
                     )
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='seg_iou_3d' + model_name)
    return model, train_metric


def build_model_fuse(cfg):
    assert cfg.MODEL_2D.NUM_CLASSES == cfg.MODEL_3D.NUM_CLASSES
    model = Net2D3DFusionSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES,
                             backbone_2d=cfg.MODEL_2D.TYPE,
                             backbone_2d_kwargs=cfg.MODEL_2D[cfg.MODEL_2D.TYPE],
                             backbone_3d=cfg.MODEL_3D.TYPE,
                             backbone_3d_kwargs=cfg.MODEL_3D[cfg.MODEL_3D.TYPE],
                             dual_head=cfg.MODEL_3D.DUAL_HEAD
                             )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_3d')
    return model, train_metric


def build_model_vfm_fusion(cfg,device,model_name = ' '):
    assert cfg.MODEL_2D.NUM_CLASSES == cfg.MODEL_3D.NUM_CLASSES
    model = LogitsFusion(device,
                         num_classes=cfg.MODEL_2D.NUM_CLASSES,
                         )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_vfm_fusion' + model_name)
    return model, train_metric
