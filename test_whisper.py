import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Third party imports
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model
from src.utils.config import (
    ExperimentParams,
    load_experiment_params,
)


class MLP(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def load_data(embeddings, labels_path):
    labels = torch.tensor(np.load(labels_path), dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(embeddings, labels), batch_size=32, shuffle=True)


def train(model, train_dataloader, test_dataloader, epochs=10, lr=1e-4, device="cpu"):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        train_acc = evaluate(model, train_dataloader, device)
        test_acc = evaluate(model, test_dataloader, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}"
        )


def evaluate(model, dataloader, device="cpu"):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    train_embeddings_path = "data/train_embeddings.npy"
    train_label2_path = "data/train_age_labels.npy"
    train_label1_path = "data/train_gender_labels.npy"
    test_embeddings_path = "data/test_embeddings.npy"
    test_label2_path = "data/test_age_labels.npy"
    test_label1_path = "data/test_gender_labels.npy"
    encoder_model_weights_path = (
        "exps/training/2025-02-12_20-43-08/encoder_weights/model_2.pt"
    )

    experiment_params: ExperimentParams = load_experiment_params(config)
    # Initialize encoder model
    encoder_model: Encoder = create_encoder_model(
        model_name=experiment_params.encoder_params.encoder_model_name,
        model_params=experiment_params.encoder_params.encoder_model_params,
    )
    # Load encoder model weights
    encoder_model.load_state_dict(torch.load(encoder_model_weights_path))

    train_embeddings = torch.tensor(np.load(train_embeddings_path), dtype=torch.float32)
    test_embeddings = torch.tensor(np.load(test_embeddings_path), dtype=torch.float32)
    encoder_model = encoder_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model = encoder_model.to(device)

    # with torch.no_grad():
    #     train_embeddings = train_embeddings.to(device)
    #     test_embeddings = test_embeddings.to(device)
    #     train_embeddings: torch.Tensor = encoder_model(train_embeddings)
    #     test_embeddings: torch.Tensor = encoder_model(test_embeddings)

    train_embeddings = train_embeddings.detach().cpu()
    test_embeddings = test_embeddings.detach().cpu()

    train_loader1 = load_data(train_embeddings, train_label1_path)
    train_loader2 = load_data(train_embeddings, train_label2_path)
    test_loader1 = load_data(test_embeddings, test_label1_path)
    test_loader2 = load_data(test_embeddings, test_label2_path)

    model1 = MLP()  # input_size=64, hidden_size=32, output_size=1)
    model2 = MLP()  # input_size=64, hidden_size=32, output_size=1)

    print("Training model for label 1...")
    train(model1, train_loader1, test_loader1)
    print("Training model for label 2...")
    train(model2, train_loader2, test_loader2)

    acc1 = evaluate(model1, test_loader1)
    acc2 = evaluate(model2, test_loader2)

    print(f"Final Accuracy for Label 1 Model: {acc1:.4f}")
    print(f"Final Accuracy for Label 2 Model: {acc2:.4f}")

    torch.save(model1.state_dict(), "mlp_label1.pth")
    torch.save(model2.state_dict(), "mlp_label2.pth")


if __name__ == "__main__":
    main()
