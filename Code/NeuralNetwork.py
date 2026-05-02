# Neural network basketball win/loss classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_auc_score, roc_curve


# Model activation setting
activations = {}
activations['hidden'] = 'tanh'   # Can be 'relu','tanh', or 'sigmoid'

nn_sizes = {}

# Training hyperparameters
opts = {}
opts['batch_size'] = 50
opts['learning_rate'] = 0.001
opts['epochs'] = 50

# Load basketball game data
project_root = Path(__file__).resolve().parents[1]
data = pd.read_csv(project_root / 'Data' / 'Basketball.csv')

data['games_played_before_date'] = data['wins_before_date'] + data['losses_before_date']
data['opponent_games_played_before_date'] = data['opponent_wins_before_date'] + data['opponent_losses_before_date']

data['win_pct_before_date'] = np.where(
    data['games_played_before_date'] > 0,
    data['wins_before_date'] / data['games_played_before_date'],
    0
)

data['opponent_win_pct_before_date'] = np.where(
    data['opponent_games_played_before_date'] > 0,
    data['opponent_wins_before_date'] / data['opponent_games_played_before_date'],
    0
)
data['win'] = np.where(data['Team_Points'] > data['Opp_Points'], 1, 0)

features = ['Team', 'Opponent', 'games_played_before_date', 'opponent_games_played_before_date', 'win_pct_before_date', 'opponent_win_pct_before_date', 'Team_SOS', 'Opponent_SOS']
X = data[features]
y = data['win'].values
X = pd.get_dummies(X, columns=['Team', 'Opponent'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

m, n_features = X_train.shape
n_classes = 2

X = X_train.to_numpy(dtype=float).T

# One-hot labels
Y = np.zeros((n_classes, m))
for i in range(m):
    Y[int(y_train[i]), i] = 1

nn_sizes['input'] = n_features
nn_sizes['hidden'] = 16
nn_sizes['output'] = n_classes


opts['verbose'] = True
opts['seed'] = 42


def train_NN(X, Y, nn_sizes, activations, opts):

    if 'seed' in opts:
        np.random.seed(opts['seed'])

    n_features, m = X.shape
    n_out, mY = Y.shape

    if m != mY:
        raise ValueError('X and Y must have the same number of samples.')
    
    if nn_sizes['input'] != n_features:
        raise ValueError('nn_sizes input must match X.')

    if nn_sizes['output'] != n_out:
        raise ValueError('nn_sizes.output must match size(Y,1).')

    if 'hidden' not in activations:
        raise ValueError('activations.hidden must be specified.')
    

    params = initialize_nn_parameters(nn_sizes)

    history = {}
    history['loss'] = np.zeros((opts['epochs'],))

    for epoch in range(opts['epochs']):
        perm = np.random.permutation(m)
        X_shuf = X[:, perm]
        Y_shuf = Y[:, perm]

        num_batches = int(np.ceil(m / opts['batch_size']))
        epoch_loss = 0.0

        for b in range(num_batches):
            idx_start = b * opts['batch_size']
            idx_end = min((b + 1) * opts['batch_size'], m)
            batch_idx = np.arange(idx_start, idx_end)

            Xb = X_shuf[:, batch_idx]
            Yb = Y_shuf[:, batch_idx]

            cache = forward_pass_nn(Xb, params, nn_sizes, activations)
            batch_loss = compute_cross_entropy_loss(cache['A2'], Yb)
            epoch_loss = epoch_loss + batch_loss

            grads = backward_pass_nn(Xb, Yb, params, cache, nn_sizes, activations)

            params['W1'] = params['W1'] - opts['learning_rate'] * grads['dW1']
            params['b1'] = params['b1'] - opts['learning_rate'] * grads['db1']

            params['W2'] = params['W2'] - opts['learning_rate'] * grads['dW2']
            params['b2'] = params['b2'] - opts['learning_rate'] * grads['db2']


        history['loss'][epoch] = epoch_loss / num_batches

        if opts['verbose'] and (((epoch + 1) % max(1, opts['epochs'] // 20) == 0) or (epoch == 0)):
            print(f"Epoch {epoch + 1}/{opts['epochs']}, Loss = {history['loss'][epoch]:.6f}")

    return params, history

def initialize_nn_parameters(nn_sizes):

    n_input = nn_sizes['input']
    n_hidden = nn_sizes['hidden']
    n_out = nn_sizes['output']

    params = {}

    # Hidden layer weights: [n_hidden, n_input]
    limit1 = np.sqrt(6 / (n_input + n_hidden))
    params['W1'] = -limit1 + 2 * limit1 * np.random.rand(n_hidden, n_input)
    params['b1'] = np.zeros((n_hidden, 1))

    # Output layer weights: [n_out, n_hidden]
    limit2 = np.sqrt(6 / (n_hidden + n_out))
    params['W2'] = -limit2 + 2 * limit2 * np.random.rand(n_out, n_hidden)
    params['b2'] = np.zeros((n_out, 1))

    return params


def forward_pass_nn(X, params, nn_sizes, activations):
    # X: [n_features, m]

    Z1 = params['W1'] @ X + params['b1']
    A1 = apply_activation(Z1, activations['hidden'])

    Z2 = params['W2'] @ A1 + params['b2']
    A2 = softmax(Z2)

    cache = {}
    cache['Z1'] = Z1
    cache['A1'] = A1
    cache['Z2'] = Z2
    cache['A2'] = A2

    return cache

def backward_pass_nn(X, Y, params, cache, nn_sizes, activations):

    n_features, m = X.shape

    grads = {}

    # Output layer
    dZ2 = cache['A2'] - Y
    grads['dW2'] = (1 / m) * (dZ2 @ cache['A1'].T)
    grads['db2'] = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer
    dA1 = params['W2'].T @ dZ2
    dZ1 = dA1 * activation_derivative_from_z(cache['Z1'], activations['hidden'])

    grads['dW1'] = (1 / m) * (dZ1 @ X.T)
    grads['db1'] = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return grads



def apply_activation(Z, activation_name):

    name = activation_name.lower()

    if name == 'relu':
        A = np.maximum(0, Z)

    elif name == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))

    elif name == 'tanh':
        A = np.tanh(Z)

    else:
        raise ValueError(f'Unknown activation: {activation_name}')

    return A


def activation_derivative_from_z(Z, activation_name):

    name = activation_name.lower()

    if name == 'relu':
        dA = (Z > 0).astype(float)

    elif name == 'sigmoid':
        S = 1 / (1 + np.exp(-Z))
        dA = S * (1 - S)

    elif name == 'tanh':
        T = np.tanh(Z)
        dA = 1 - T**2

    else:
        raise ValueError(f'Unknown activation: {activation_name}')

    return dA

def softmax(Z):
    # Numerically stable softmax applied columnwise

    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    P = expZ / np.sum(expZ, axis=0, keepdims=True)
    return P


def compute_cross_entropy_loss(Yhat, Y):
    # Cross-entropy for one-hot targets

    eps_val = 1e-12
    Yhat_clipped = np.maximum(Yhat, eps_val)

    m = Y.shape[1]
    loss = -(1 / m) * np.sum(Y * np.log(Yhat_clipped))
    return loss


def predict_NN(X, params, nn_sizes, activations):
    cache = forward_pass_nn(X, params, nn_sizes, activations)
    class_probs = cache['A2']
    predicted_labels = np.argmax(class_probs, axis=0)
    return predicted_labels, class_probs


params, history = train_NN(X, Y, nn_sizes, activations, opts)

X_test = X_test.to_numpy(dtype=float).T
predicted_labels, class_probs = predict_NN(X_test, params, nn_sizes, activations)

test_confusion_matrix = confusion_matrix(y_test, predicted_labels, labels=[0, 1])
test_accuracy = accuracy_score(y_test, predicted_labels)
test_f1 = f1_score(y_test, predicted_labels)
test_auc = roc_auc_score(y_test, class_probs[1, :])

print("Test Confusion Matrix:")
print(test_confusion_matrix)
print("Test Accuracy:", test_accuracy)
print("Test F1 Score:", test_f1)
print("Test ROC AUC:", test_auc)

# ROC curve using the neural network's predicted win probabilities
fpr, tpr, _ = roc_curve(y_test, class_probs[1, :], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network Test ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('nn_roc_curve.png')

# South Carolina vs Connecticut example
new_row = pd.DataFrame({
    'Team': ['South Carolina'],
    'Opponent': ['Connecticut'],
    'games_played_before_date': [39],
    'opponent_games_played_before_date': [39],
    'win_pct_before_date': [0.97435],
    'opponent_win_pct_before_date': [0.92307],
    'Team_SOS': [0.5909],
    'Opponent_SOS': [0.6243]
})

new_row = pd.get_dummies(new_row, columns=['Team', 'Opponent'], drop_first=True)
new_row = new_row.reindex(columns=X_train.columns, fill_value=0)

new_row_nn = new_row.to_numpy(dtype=float).T
pred, probs = predict_NN(new_row_nn, params, nn_sizes, activations)
print(f"NN predicts: {'Win' if pred[0] == 1 else 'Loss'}")
print(f"Win probability: {probs[1,0]:.4f}")

# Feature importance via first layer weight magnitude
# W1 is shape [n_hidden, n_features], so take the mean absolute value across hidden neurons
feature_names_nn = list(X_train.columns)

w1_importance = np.mean(np.abs(params['W1']), axis=0)

# Plot the top 15 because the team/opponent dummy variables create many features
indices_nn = np.argsort(w1_importance)[::-1][:15]

plt.figure(figsize=(10, 6))
plt.bar(range(15), w1_importance[indices_nn])
plt.xticks(range(15), 
           [feature_names_nn[i] for i in indices_nn], 
           rotation=45, ha='right')
plt.title('Neural Network Feature Importances (Input Weight Magnitude)')
plt.tight_layout()
plt.savefig('nn_feature_importance.png')
plt.show()
