# %%
import torch
import torch.nn as nn
import cv2
import numpy as np

from mobilenetv3magicv3 import get_pose_net as create_model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
      pass
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      pass
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    # scores, inds, clses, ys, xs = _topk_rong(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections

def filter_and_restore(rets):
    dets = []
    rets = rets.detach().numpy().squeeze()
    for det in rets:
        x1,y1,x2,y2,score,classid = det
        x1 = int(float(x1) * DOWN_SCALE * RESOTRE_SCALE)
        y1 = int(float(y1) * DOWN_SCALE * RESOTRE_SCALE)
        x2 = int(float(x2) * DOWN_SCALE * RESOTRE_SCALE)
        y2 = int(float(y2) * DOWN_SCALE * RESOTRE_SCALE)
        score = float(score)
        classid = int(classid)
        class_name = CLASS_NAMES[classid]
        if score > PER_CLASS_THRESHOLD[class_name]:
            # print(x1,y1,x2,y2,class_name,score)
            dets.append((x1,y1,x2,y2,class_name,score))
    return dets

def _vis(boxes, img):
    vis_img = np.copy(img)
    for each in boxes:
        x1,y1,x2,y2,class_name,score = each
        print(x1,y1,x2,y2,class_name,score)
        cv2.rectangle(vis_img, (x1,y1), (x2,y2), (255,0,0), 5)
    cv2.imwrite('./vis_res.jpeg', vis_img)

def main(im_path):
    origin_image = cv2.imread(im_path)
    assert origin_image.shape == (ORIGIN_HEIGHT, ORIGIN_WIDTH, 3)
    input_image = cv2.resize(origin_image, (INPUT_WIDTH, INPUT_HEIGHT))
    input_data = ((input_image / 255. - MEAN) / STD).astype(np.float32)
    input_data = input_data.transpose(2, 0, 1).reshape(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    input_data = torch.from_numpy(input_data)
    output = MODEL(input_data)[-1]
    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']
    dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=100)
    dets = filter_and_restore(dets)
    if VIS:
        _vis(dets, origin_image)
    return dets

if __name__ == "__main__":
    INPUT_HEIGHT, INPUT_WIDTH = 288, 384
    ORIGIN_HEIGHT, ORIGIN_WIDTH = 960, 1280
    RESOTRE_SCALE = float(ORIGIN_HEIGHT) / INPUT_HEIGHT
    MEAN = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(1, 1, 3)
    STD = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(1, 1, 3)
    DOWN_SCALE = 4.0
    VIS = True
    CLASS_NAMES = ['wire','pet feces','shoe'
                ,'bar stool a','fan','power strip','dock(ruby)','dock(rubys+tanosv)'
                ,'bar stool b','scale','clothing item','cleaning robot','fan b'
                ,'door mark a','door mark b','wheel','door mark c','flat base'
                ,'whole fan','whole fan b','whole bar stool a','whole bar stool b'
                ,'fake poop a','dust pan','folding chair','laundry basket'
                ,'handheld cleaner','sock', 'fake poop b']
    PER_CLASS_THRESHOLD = {'wire': 0.22
                            ,'pet feces': 0.08
                            ,'shoe': 0.27
                            ,'bar stool a': 0.17
                            ,'fan': 0.33
                            ,'power strip': 0.21
                            ,'dock(ruby)': 0.3
                            ,'dock(rubys+tanosv)': 0.22
                            ,'bar stool b': 0.2
                            ,'scale': 0.21
                            ,'clothing item': 0.3
                            ,'cleaning robot': 0.25
                            ,'fan b': 0.1
                            ,'door mark a': 0.19
                            ,'door mark b': 0.22
                            ,'wheel': 0.2
                            ,'door mark c': 0.3
                            ,'flat base': 0.25
                            ,'whole fan': 0.2
                            ,'whole fan b': 0.12
                            ,'whole bar stool a': 0.22
                            ,'whole bar stool b': 0.3
                            ,'fake poop a': 0.12
                            ,'dust pan': 0.11
                            ,'folding chair': 0.14
                            ,'laundry basket': 0.15
                            ,'handheld cleaner': 0.08
                            ,'sock': 0.22
                            ,'fake poop b': 0.13}
    MODEL = create_model(None, {'hm': 29, 'wh': 2, 'reg': 2}, None)
    MODEL = load_model(MODEL, './ctdet_288x384_20200806.pth')
    MODEL.eval()
    main("./StereoVision_L_52855983_1_0_0_16018_D_Shoe_-6683_-652.jpeg")