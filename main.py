import argparse
import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from preprocess import get_data
from models import CONFIG, RNN_LSTM, RNN_GRU, TextCNN, MLP, Transformer, Bert

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class TextClassifier:
    def __init__(self, config, model_type, learning_rate, batch_size, max_length):
        self.config = config
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length

        if self.model_type == "RNN_LSTM":
            self.model = RNN_LSTM(self.config).to(DEVICE)
        elif self.model_type == "RNN_GRU":
            self.model = RNN_GRU(self.config).to(DEVICE)
        elif self.model_type == "TextCNN":
            self.model = TextCNN(self.config).to(DEVICE)
        elif self.model_type == "MLP":
            self.model = MLP(self.config).to(DEVICE)
        elif self.model_type == "Transformer":
            self.model = Transformer(self.config).to(DEVICE)
        elif self.model_type == "Bert":
            self.model = Bert(self.config).to(DEVICE)
        else:
            raise ValueError(
                "Invalid model type. Choose from: RNN_LSTM, RNN_GRU, TextCNN, MLP, Transformer, Bert"
            )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=5)

    def train(self, train_dataloader):
        self.model.train()
        train_loss, train_acc = 0.0, 0.0
        count, correct = 0, 0
        full_true, full_pred = [], []

        for x, y in train_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            correct += (output.argmax(1) == y).float().sum().item()
            count += len(x)
            full_true.extend(y.cpu().numpy().tolist())
            full_pred.extend(output.argmax(1).cpu().numpy().tolist())

        train_loss *= self.batch_size
        train_loss /= len(train_dataloader.dataset)
        train_acc = correct / count
        self.scheduler.step()
        f1 = f1_score(np.array(full_true), np.array(full_pred), average="binary")
        return train_loss, train_acc, f1

    def validate_and_test(self, dataloader):
        self.model.eval()
        loss, acc = 0.0, 0.0
        count, correct = 0, 0
        full_true, full_pred = [], []

        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = self.model(x)
            loss_val = self.criterion(output, y)
            loss += loss_val.item()
            correct += (output.argmax(1) == y).float().sum().item()
            count += len(x)
            full_true.extend(y.cpu().numpy().tolist())
            full_pred.extend(output.argmax(1).cpu().numpy().tolist())

        loss *= self.batch_size
        loss /= len(dataloader.dataset)
        acc = correct / count
        f1 = f1_score(np.array(full_true), np.array(full_pred), average="binary")
        return loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "-m", "--max-length", type=int, default=120, help="Maximum sentence length"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "-n",
        "--model-type",
        type=str,
        default="TextCNN",
        help="Model type (RNN_LSTM, RNN_GRU, TextCNN, MLP)",
    )
    args = parser.parse_args()

    config = CONFIG()
    train_dataloader, val_dataloader, test_dataloader = get_data(
        args.max_length, args.batch_size
    )

    classifier = TextClassifier(
        config, args.model_type, args.learning_rate, args.batch_size, args.max_length
    )

    wandb.init(
        project=f"Text Classifier",
        name=f"{args.model_type}",
        entity="642579041",
        group=args.model_type,
        tags=[
            f"lr_{args.learning_rate}",
            f"bs_{args.batch_size}",
            f"ep_{args.epochs}",
            f"ml_{args.max_length}",
        ],
    )
    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model_type": args.model_type,
    }

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss, train_acc, train_f1 = classifier.train(train_dataloader)
        val_loss, val_acc, val_f1 = classifier.validate_and_test(val_dataloader)
        test_loss, test_acc, test_f1 = classifier.validate_and_test(test_dataloader)

        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "epoch": epoch,
                "model_type": args.model_type,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
            }
        )

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
