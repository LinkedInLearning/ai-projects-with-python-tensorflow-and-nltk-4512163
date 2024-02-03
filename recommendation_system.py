# Import necessary libraries
import torch
import torchrec
from torchrec.datasets import MovieLens

# Load the MovieLens dataset
dataset = MovieLens()

# Split the dataset into training and test sets
train_dataset, test_dataset = torchrec.datasets.train_test_split(dataset)

# Define the collaborative filtering model
class CollaborativeFilteringModel(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        return (user_embedding * item_embedding).sum(1)

# Initialize the model
model = CollaborativeFilteringModel(dataset.num_users, dataset.num_items, embedding_dim=10)

# Define the loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for user_ids, item_ids, ratings in train_dataset:
        # Forward pass
        predicted_ratings = model(user_ids, item_ids)
        loss = loss_fn(predicted_ratings, ratings)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    total_loss = 0
    for user_ids, item_ids, ratings in test_dataset:
        predicted_ratings = model(user_ids, item_ids)
        loss = loss_fn(predicted_ratings, ratings)
        total_loss += loss.item()

print(f"Test Loss: {total_loss / len(test_dataset)}")