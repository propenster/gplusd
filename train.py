import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from icecream import ic

from dataloader import load_dataset
from models import GPlusD
from test import evaluate, mcc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(data_path, pretrain=None, outputdir="test", training=True, ker=None, epoch_num=1000):
    ker = ker or [27, 14, 7]

    # Create experiment folders
    exp_folder = Path(f"./output/{outputdir}")
    exp_folder.mkdir(parents=True, exist_ok=True)

    ic("Data loading")
    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = load_dataset(data_path, device=device)

    net = GPlusD(ker).to(device)
    if pretrain:
        net.load_state_dict(torch.load(pretrain))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    best_mcc, best_precision, best_recall = 0, 0, 0
    patience_counter = 0
    patience_limit = 10  # Stops after 10 evaluation cycles without MCC improvement

    if training:
        ic("Start training")
        for epoch in range(epoch_num):
            net.train()
            
            for (batch_pos, batch_neg) in zip(train_pos, train_neg):
                inputs = torch.cat((batch_pos[0], batch_neg[0]), dim=0)
                labels = torch.cat((batch_pos[1], batch_neg[1]), dim=0).long()

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluation cycle every 10 epochs
            if epoch % 10 == 0:
                torch.save(net.state_dict(), exp_folder / f"epoch_{epoch}.pth")
                
                net.eval()
                with torch.no_grad():
                    eval_data, _ = evaluate(net, [val_pos, val_neg])
                    precision, recall, MCC = mcc(eval_data)

                ic(epoch, outputdir, precision, recall, MCC)

                # Save best models
                if precision > best_precision:
                    best_precision = precision
                    ic("Update best precision")
                    torch.save(net.state_dict(), exp_folder / "best_precision.pth")
                    
                if recall > best_recall:
                    best_recall = recall
                    ic("Update best recall")
                    torch.save(net.state_dict(), exp_folder / "best_recall.pth")
                    
                if MCC > best_mcc:
                    best_mcc = MCC
                    ic("Update best MCC")
                    torch.save(net.state_dict(), exp_folder / "best_mcc.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience_limit:
                    ic("Early stopping triggered")
                    break

    # Final Testing Phase
    ic("Start testing phase")
    net.load_state_dict(torch.load(exp_folder / "best_mcc.pth"))
    net.eval()
    
    with torch.no_grad():
        eval_data, _ = evaluate(net, [test_pos, test_neg])
        precision, recall, MCC = mcc(eval_data)

    ic(precision, recall, MCC)

    # Save test logs
    with open(exp_folder / "log.txt", "w") as f:
        f.write(f"Test precision: {precision}\nTest recall: {recall}\nTest MCC : {MCC}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True, help="path to dataset(txt file)")
    parser.add_argument("-o", "--outputdir", type=str, default="test", help="name of folder to save output in ./output")
    parser.add_argument("-w", "--weight", type=str, help="Path to pre-train")
    parser.add_argument("--test", action="store_true", help="Add this flag to do test only")
    args = parser.parse_args()

    train(
        data_path=args.data, 
        pretrain=args.weight, 
        outputdir=args.outputdir, 
        training=not args.test
    )