import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)
#     print("box1 size", box1.size())
#     print("box2 size", box2.size())
#     print("N", N)
#     print("M", M)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


# +
class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.mse = nn.MSELoss(reduction="sum")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x / self.S - 0.5 * w
        y1 = y / self.S - 0.5 * h
        x2 = x / self.S + 0.5 * w
        y2 = y / self.S + 0.5 * h
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 5)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        best_ious = torch.zeros(box_target.shape[0], device=self.device)
        best_boxes = torch.zeros_like(box_target, device=self.device)
        for box_pred in pred_box_list:
#             print("box_pred isze", box_pred.size())
            iou = compute_iou(self.xywh2xyxy(box_pred[:, :4]), self.xywh2xyxy(box_target[:, :4]))
            iou = torch.diagonal(iou, 0)
            mask = iou > best_ious
            best_ious[mask] = iou[mask]
            best_boxes[mask, :] = box_pred[mask, :]
            
        best_ious = best_ious.view(-1, 1)
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        classes_pred = classes_pred[has_object_map]
#         print("Classes_pred_size", classes_pred.size())
        classes_target = classes_target[has_object_map]
        class_loss = self.mse(classes_pred, classes_target)
        return class_loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        no_object_mask = ~has_object_map
        loss = 0.0
        for pred_boxes in pred_boxes_list:
            noobj_pred_boxes = pred_boxes[no_object_mask]
#             print("noobj_pred_boxes.size()", noobj_pred_boxes.size())
            noobj_pred_boxes_C = noobj_pred_boxes[:, 4]
            loss += self.mse(noobj_pred_boxes_C, torch.zeros_like(noobj_pred_boxes_C))
        loss *= self.l_noobj
        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
#         print(box_pred_conf.size())
#         print(box_target_conf.size())
        return self.mse(box_pred_conf, box_target_conf)

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        reg_loss = 0.0
        pred_xy = box_pred_response[:, :2]
        pred_wh = torch.sqrt(box_pred_response[:, 2:])
        target_xy = box_target_response[:, :2]
        target_wh = torch.sqrt(box_target_response[:, 2:])
        
        reg_loss += self.mse(pred_xy, target_xy)
        reg_loss += self.mse(pred_wh, target_wh)
        reg_loss *= self.l_coord
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0
        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_cls = pred_tensor[:, :, :, self.B * 5:]
        pred_boxes_list = []
        for i in range(self.B):
            start_idx = i * 5
            end_idx = start_idx + 5
            pred_box = pred_tensor[:, :, :, start_idx:end_idx]
#             print("Pred_box size:", pred_box.size())
            pred_boxes_list.append(pred_box)


        # compcute classification loss
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)


        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)


        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        for i, pred_boxes in enumerate(pred_boxes_list):
            # tensor size (N, S, S, 5)
#             print("pred_boxes before filtering", pred_boxes.size())
            pred_boxes_list[i] = pred_boxes[has_object_map].view(-1, 5)
        
        target_boxes = torch.cat([target_boxes, torch.ones(target_boxes.size(0), target_boxes.size(1), target_boxes.size(2), 1, device=self.device)], dim=-1)
        
#         print("target_boxes before filtering", target_boxes.size())
        target_boxes = target_boxes[has_object_map].view(-1, 5)
        
#         print("target_boxes after filtering", target_boxes.size())

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)
    
        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss = self.get_regression_loss(best_boxes, target_boxes)
#         print("box isze", best_boxes.size())
        # compute contain_object_loss
        containing_obj_loss = self.get_contain_conf_loss(best_boxes[:, 4].view(-1, 1), best_ious)

        cls_loss /= N
        no_obj_loss /= N
        reg_loss /= N
        containing_obj_loss /= N
        
        # compute final loss
        total_loss = cls_loss + no_obj_loss + reg_loss + containing_obj_loss


        # construct return loss_dict
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=containing_obj_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss,
        )
        return loss_dict

# +
# s = 0
# mse = nn.MSELoss(reduction="sum")
# a = torch.tensor([1,2,3,4,5,6], dtype=float).to('cuda')
# s +=mse(a, torch.zeros_like(a))
# e = a[1]
# print(a[:4])
# print(s)

# +
# a = torch.tensor([1,2,3,4,5,6])
# b = torch.tensor([True, False, True, False, True, False])
# c = ~b
# print(c)
# print(a.unsqueeze(0))

# +
# a = torch.tensor([[True, False],
#                   [False, True],
#                   [False, True],
#                   [False, True]])
# b = torch.tensor([ [[1, 4, 3],[3, 5, 7]]
#                   ,[[4, 8, 5],[6, 6, 7]]
#                   ,[[1, 8, 3],[3, 9, 7]]
#                   ,[[1, 2, 3],[5, 1, 7]]])
# print(a.size())
# print(b.size())
# c = b[a]
# print(c.size())

# +
# N = 5
# S = 2
# target_boxes = torch.randn(N, S, S, 4, device="cuda")

# print(target_boxes.size())
# target_boxes = torch.cat([target_boxes, torch.ones(target_boxes.size(0), target_boxes.size(1), target_boxes.size(2), 1, device="cuda")], dim=-1)
# print(target_boxes.size())
# a = torch.randn(5, 3)
# b = a[:, 1].view(-1, 1)
# print(a)
# print(b.size())
# -

"""
Parameters:
boxes: (N,4) representing by x,y,w,h

Returns:
boxes: (N,4) representing by x1,y1,x2,y2

if for a Box b the coordinates are represented by [x, y, w, h] then
x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
Note: Over here initially x, y are the center of the box and w,h are width and height.
"""
### CODE ###
# Your code here
# N = 3
# boxes = torch.randn(N, 4)
# S = 5
# x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
# x1 = x / S - 0.5 * w
# y1 = y / S - 0.5 * h
# x2 = x / S + 0.5 * w
# y2 = y / S + 0.5 * h
# boxes = torch.stack([x1, y1, x2, y2], dim=1)
# print(boxes.size())
# print(boxes)


