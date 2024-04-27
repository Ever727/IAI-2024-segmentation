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

model_classes = {
    "RNN_LSTM": RNN_LSTM,
    "RNN_GRU": RNN_GRU,
    "TextCNN": TextCNN,
    "MLP": MLP,
    "Transformer": Transformer,
    "Bert": Bert,
}


class TextClassifier:
    def __init__(self, config, model_type, learning_rate, batch_size, max_length):
        self.config = config
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length

        if self.model_type in model_classes:
            self.model = model_classes[self.model_type](self.config).to(DEVICE)
        else:
            raise ValueError(
                f"Invalid model type: {self.model_type}. Choose from: {', '.join(model_classes.keys())}"
            )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=5)

    def train(self, train_dataloader):
        self.model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_count = 0
        true_labels, predicted_labels = [], []

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
            total_train_correct += (
                (outputs.argmax(dim=1) == labels).float().sum().item()
            )
            total_train_count += len(inputs)
            true_labels.extend(labels.cpu().numpy().tolist())
            predicted_labels.extend(outputs.argmax(dim=1).cpu().numpy().tolist())

        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        train_accuracy = total_train_correct / total_train_count
        self.scheduler.step()
        train_f1 = f1_score(
            np.array(true_labels), np.array(predicted_labels), average="binary"
        )
        return avg_train_loss, train_accuracy, train_f1

    def validate_and_test(self, dataloader):
        self.model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_count = 0
        true_labels, predicted_labels = [], []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_val_loss += loss.item()
                total_val_correct += (
                    (outputs.argmax(dim=1) == labels).float().sum().item()
                )
                total_val_count += len(inputs)
                true_labels.extend(labels.cpu().numpy().tolist())
                predicted_labels.extend(outputs.argmax(dim=1).cpu().numpy().tolist())

        avg_val_loss = total_val_loss / len(dataloader.dataset)
        val_accuracy = total_val_correct / total_val_count
        val_f1 = f1_score(
            np.array(true_labels), np.array(predicted_labels), average="binary"
        )
        return avg_val_loss, val_accuracy, val_f1


def main():
    parser = argparse.ArgumentParser(description="Text Classification Model")
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs",
    )
    parser.add_argument(
        "-m",
        "--max-length",
        type=int,
        default=64,
        help="Maximum sentence length",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "-n",
        "--model-type",
        type=str,
        default="TextCNN",
        choices=["RNN_LSTM", "RNN_GRU", "TextCNN", "MLP", "Transformer", "Bert"],
        help="Model type",
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
