import math
import argparse

import torch
from icecream import ic
from sklearn.metrics import confusion_matrix

from models import GPlusD
from dataloader import load_data_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(net, loaders):
    eval_result = list()
    ltotal = list()
    lcorrect = list()
    pred_result = list()
    true_result = list()
    for load in loaders:
        total = 0
        correct = 0
        pred_list = list()
        true_list = list()
        for data in load:
            inputs = data[0]
            labels = data[1]
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_list += list(predicted.cpu().numpy())
            true_list += list(labels.cpu().numpy())
        acc = correct / total
        eval_result.append(acc)
        lcorrect.append(correct)
        ltotal.append(total)
        pred_result.append(pred_list)
        true_result.append(true_list)
    return (lcorrect, ltotal), pred_result, true_result


def mcc(data):
    pos_count = data[0][0]
    neg_count = data[0][1]

    tol_pos_count = data[1][0]
    tol_neg_count = data[1][1]

    TP = pos_count
    FN = tol_pos_count - pos_count
    TN = neg_count
    FP = tol_neg_count - neg_count

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return precision, recall, MCC


def test(data_path, pretrain, ker=None):
    if ker is None:
        ker = [27, 14, 7]

    dataloader = load_data_test(data_path, device=device)

    net = GPlusD(ker)
    net.to(device)

    net.load_state_dict(torch.load(pretrain))

    net.eval()
    eval_data, results, true_results = evaluate(net, [dataloader])
    
    cm = confusion_matrix(true_results[0], results[0])

    return results, cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
    )
    parser.add_argument("-w", "--weight", type=str)
    args = parser.parse_args()

    output, cm = test(args.data, args.weight)
    
    ic(cm)

    with open("infer_results.txt", "w") as f:
        ic("Save the results to infer_results.txt")
        for out in output[0]:
            f.write(str(out) + "\n")