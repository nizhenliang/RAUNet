import numpy as np
from torch import nn
import torch
import tqdm
import math

def val_multi(model: nn.Module, criterion, valid_loader, num_classes,batch_size,device_ids):
    with torch.no_grad():
        model.eval()
        losses = []
        confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.uint32)
        dt_size = len(valid_loader.dataset)
        tq = tqdm.tqdm(total=math.ceil(dt_size / batch_size))
        tq.set_description('Test')
        for inputs, targets in valid_loader:
            inputs = inputs.cuda(device_ids[0])
            targets = targets.long()
            targets = targets.cuda(device_ids[0])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            target_classes = targets.data.cpu().numpy()
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, num_classes)
            tq.set_postfix(loss='{0:.3f}'.format(np.mean(losses)))
            tq.update(1)
        tq.close()
        confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
        valid_loss = np.mean(losses)  # type: float
        ious = {'iou_{}'.format(cls + 1): iou
                for cls, iou in enumerate(calculate_iou(confusion_matrix))}

        dices = {'dice_{}'.format(cls + 1): dice
                 for cls, dice in enumerate(calculate_dice(confusion_matrix))}

        average_iou = np.mean(list(ious.values()))
        average_dices = np.mean(list(dices.values()))

        print('Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss, average_iou, average_dices))


        return average_dices, average_iou


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices
