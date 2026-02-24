from core.trainer import train_toy_model


if __name__ == "__main__":
    # Train the model!
    trained_state, trained_model = train_toy_model(
        num_features=20, hidden_dim=5, sparsity=0.99
    )
