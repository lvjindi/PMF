import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser("PMF")
parser.add_argument('--num_feature', type=int, default=5, help='the num of latent feature (D)')
parser.add_argument('--lam_u', type=float, default=0.05, help='the value of lambda U')
parser.add_argument('--lam_v', type=float, default=0.05, help='the value of lambda V')
parser.add_argument('--epochs', type=int, default=10000, help='the num of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
args = parser.parse_args()

# loss function
class PMFLoss(torch.nn.Module):
    def __init__(self, lam_u=0.3, lam_v=0.3):
        super().__init__()
        self.lam_u = lam_u
        self.lam_v = lam_v

    def forward(self, matrix, u_features, v_features):
        non_zero_mask = (matrix != -1).type(torch.FloatTensor)
        # g(U.V^T), g = sigmoid
        predicted = torch.sigmoid(torch.mm(u_features, v_features.t()))
        # root mean squared error (RMSE)
        diff = (matrix - predicted) ** 2
        prediction_error = torch.sum(diff * non_zero_mask)

        u_regularization = self.lam_u * torch.sum(u_features.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(v_features.norm(dim=1))

        return prediction_error + u_regularization + v_regularization


def main():
    ratings = pd.read_csv('./data/ml-latest-small/ratings.csv')
    rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    n_users, n_movies = rating_matrix.shape

    # Scaling ratings to between 0 and 1, this helps our model by constraining predictions
    min_rating, max_rating = ratings['rating'].min(), ratings['rating'].max()
    # map the ratings 1, ..., K to the interval [0, 1] using the function t(x) = (x - 1)/ (K - 1)
    rating_matrix = (rating_matrix - min_rating) / (max_rating - min_rating)

    # Replacing missing ratings with -1 so we can filter them out later
    rating_matrix[rating_matrix.isnull()] = -1
    rating_matrix = torch.FloatTensor(rating_matrix.values)

    # initialize the U and V feature matrices, and require their gradient to update
    latent_vectors = args.num_feature
    user_features = torch.randn(n_users, latent_vectors, requires_grad=True)
    user_features.data.mul_(0.01)
    movie_features = torch.randn(n_movies, latent_vectors, requires_grad=True)
    movie_features.data.mul_(0.01)

    criterion = PMFLoss(lam_u=args.lam_u, lam_v=args.lam_v)

    optimizer = torch.optim.Adam([user_features, movie_features], lr=args.lr)

    train(args.epochs, criterion, optimizer, user_features, movie_features, rating_matrix)
    # validate the model
    infer(4, rating_matrix, user_features, movie_features, max_rating, min_rating)


def train(epochs, criterion, optimizer, user_features, movie_features, rating_matrix):
    for step, epoch in enumerate(range(args.epochs)):
        optimizer.zero_grad()
        loss = criterion(rating_matrix, user_features, movie_features)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}, Loss {loss:.3f}")


def infer(user_idx, rating_matrix, user_features, movie_features, max_rating, min_rating):
    user_idx = user_idx
    user_ratings = rating_matrix[user_idx, :]
    true_ratings = user_ratings != -1
    predictions = torch.sigmoid(torch.mm(user_features[user_idx, :].view(1, -1), movie_features.t()))
    predicted_ratings = (predictions.squeeze()[true_ratings] * (max_rating - min_rating) + min_rating).round()
    actual_ratings = (user_ratings[true_ratings] * (max_rating - min_rating) + min_rating).round()

    print('User_id:', user_idx)
    print("Predictions: \n", predicted_ratings)
    print("Truth: \n", actual_ratings)


if __name__ == '__main__':
    main()
