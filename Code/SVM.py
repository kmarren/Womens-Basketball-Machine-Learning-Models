# Binary basketball win/loss classification using a soft-margin SVM
# The SVM is trained with projected gradient ascent on the dual variables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

np.random.seed(62)

order = 1       # polynomial feature degree; 1 keeps the original six features
C = 0.05       # soft-margin parameter
eta = 1e-8      # step size
numIters = 20000


def polynomial_kernel(order, Xin):
    if order == 1:
        Xnew = Xin
    elif order == 2:
        # Degree 2 polynomial expansion
        Xnew = poly_expand_deg2(Xin)
    elif order == 3:
        # Degree 3 polynomial expansion
        Xnew = poly_expand_deg3(Xin)
    else:
        raise ValueError('Only specify polynomial degree 1, 2 or 3')
    return Xnew


# Polynomial feature expansion helpers
def poly_expand_deg2(X):
    # X is N x 6 with the six numeric basketball features

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]
    x5 = X[:, 4]
    x6 = X[:, 5]

    # Degree-2 features: constant, linear terms, squared terms, and pairwise products

    Phi = np.column_stack([
        np.ones((X.shape[0], 1)),
        x1, x2, x3, x4, x5, x6,
        x1**2, x2**2, x3**2, x4**2, x5**2, x6**2,
        x1 * x2, x1 * x3, x1 * x4, x1 * x5, x1 * x6,
        x2 * x3, x2 * x4, x2 * x5, x2 * x6,
        x3 * x4, x3 * x5, x3 * x6,
        x4 * x5, x4 * x6,
        x5 * x6
    ])

    return Phi


def poly_expand_deg3(X):
    # X is N x 6 with the six numeric basketball features

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]
    x5 = X[:, 4]
    x6 = X[:, 5]

    # Degree-3 features:
    # constant
    # degree 1 terms
    # degree 2 terms
    # degree 3 terms:
    # pure cubes, squared-linear products, and three-way products

    Phi = np.column_stack([
        np.ones((X.shape[0], 1)),
        # degree 1
        x1, x2, x3, x4, x5, x6,
        # degree 2
        x1**2, x2**2, x3**2, x4**2, x5**2, x6**2, 
        x1 * x2, x1 * x3, x1 * x4, x1 * x5, x1 * x6,
        x2 * x3, x2 * x4, x2 * x5, x2 * x6,
        x3 * x4, x3 * x5, x3 * x6,
        x4 * x5, x4 * x6,
        x5 * x6,
        # degree 3 pure cubes
        x1**3, x2**3, x3**3, x4**3, x5**3, x6**3,
        # degree 3 squared-linear terms
        (x1**2) * x2, (x1**2) * x3, (x1**2) * x4, (x1**2) * x5, (x1**2) * x6,
        (x2**2) * x1, (x2**2) * x3, (x2**2) * x4, (x2**2) * x5, (x2**2) * x6,
        (x3**2) * x1, (x3**2) * x2, (x3**2) * x4, (x3**2) * x5, (x3**2) * x6,
        (x4**2) * x1, (x4**2) * x2, (x4**2) * x3, (x4**2) * x5, (x4**2) * x6,
        (x5**2) * x1, (x5**2) * x2, (x5**2) * x3, (x5**2) * x4, (x5**2) * x6,
        (x6**2) * x1, (x6**2) * x2, (x6**2) * x3, (x6**2) * x4, (x6**2) * x5,
        x1 * x2 * x3,
        x1 * x2 * x4,
        x1 * x2 * x5,
        x1 * x2 * x6,
        x1 * x3 * x4,
        x1 * x3 * x5,
        x1 * x3 * x6,
        x1 * x4 * x5,
        x1 * x4 * x6,
        x1 * x5 * x6,
        x2 * x3 * x4,
        x2 * x3 * x5,
        x2 * x3 * x6,
        x2 * x4 * x5,
        x2 * x4 * x6,
        x2 * x5 * x6,
        x3 * x4 * x5,
        x3 * x4 * x6,
        x3 * x5 * x6,
        x4 * x5 * x6
    ])

    return Phi


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
data['win'] = np.where(data['Team_Points'] > data['Opp_Points'], 1, -1)

features = ['games_played_before_date', 'opponent_games_played_before_date', 'win_pct_before_date', 'opponent_win_pct_before_date', 'Team_SOS', 'Opponent_SOS']
X = data[features].values
y = data['win'].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

m = X.shape[0]
d = X.shape[1]

# Apply the same feature expansion to train and test data.
Xtrain = polynomial_kernel(order, X_train_raw)
Xtest = polynomial_kernel(order, X_test_raw)
d = Xtrain.shape[1]

# Use assignment-style variable names for the SVM training code
yTrain = y_train
yTest = y_test
mTrain = Xtrain.shape[0]

# Optimization settings
tol = 1e-4

# Model parameters
w = np.zeros(d)
b = 0
alpha = np.zeros(mTrain)
dualObjHistory = np.zeros(numIters)

# Dual quadratic matrix
Q = np.outer(yTrain, yTrain) * (Xtrain @ Xtrain.T)

# Projected gradient ascent
for iter in range(numIters):

    grad = np.ones((mTrain,)) - Q @ alpha
    alpha = alpha + eta * grad

    # Box constraint
    alpha = np.minimum(np.maximum(alpha, 0), C)

    # Equality constraint
    alpha = alpha - yTrain * ((yTrain @ alpha) / (yTrain @ yTrain))

    # Project again
    alpha = np.minimum(np.maximum(alpha, 0), C)

    # Store the dual objective value
    dualObjHistory[iter] = np.sum(alpha) - 0.5 * alpha @ Q @ alpha


# Recover the primal weight vector from the dual variables
w = Xtrain.T @ (alpha * yTrain)

# Compute bias b
svMargin = np.where((alpha > tol) & (alpha < C - tol))[0]

if len(svMargin) > 0:
    bVals = yTrain[svMargin] - Xtrain[svMargin, :] @ w
    b = np.mean(bVals)
else:
    svAll = np.where(alpha > tol)[0]
    if len(svAll) > 0:
        bVals = yTrain[svAll] - Xtrain[svAll, :] @ w
        b = np.mean(bVals)
    else:
        b = 0.0

# Predict on train and test sets
trainScores = Xtrain @ w + b
testScores = Xtest @ w + b
yPredTrain = np.where(trainScores >= 0, 1, -1)
yPredTest = np.where(testScores >= 0, 1, -1)

trainAccuracy = accuracy_score(yTrain, yPredTrain)
testAccuracy = accuracy_score(yTest, yPredTest)
testPrecision = precision_score(yTest, yPredTest, pos_label=1, zero_division=0)
testRecall = recall_score(yTest, yPredTest, pos_label=1, zero_division=0)
testF1 = f1_score(yTest, yPredTest, pos_label=1, zero_division=0)

print(f'Training accuracy: {100 * trainAccuracy:.2f}%')
print(f'Test accuracy: {100 * testAccuracy:.2f}%')
print(f'Test precision: {testPrecision:.4f}')
print(f'Test recall: {testRecall:.4f}')
print(f'Test F1 score: {testF1:.4f}')

# Test confusion matrix
confMat = confusion_matrix(yTest, yPredTest, labels=[-1, 1])

print('Test confusion matrix:')
print(confMat)
print('Rows = true, columns = predicted')
print('-1 = Loss, 1 = Win')

# Plot dual objective history
plt.figure()
plt.plot(dualObjHistory, linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel('Dual Objective')
plt.title('Dual Objective History')
plt.grid(True)

# ROC curve using held-out test scores

fpr, tpr, _ = roc_curve(yTest, testScores, pos_label=1)
roc_auc = auc(fpr, tpr)

print(f'Test ROC AUC: {roc_auc:.4f}')

plt.figure()
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC Curve')
plt.legend()
plt.grid(True)

uconn_vs_sc = np.array([[39, 39, 0.97435, 0.92307, 0.5909, 0.6243]])

X_new = polynomial_kernel(1, uconn_vs_sc)
score = X_new @ w + b
prediction = 1 if score >= 0 else -1
print(f"SVM predicts: {'Win' if prediction == 1 else 'Loss'}")

# Feature importance via weight magnitude for the original six features
# This block assumes order = 1.
feature_names = ['games_played', 'opp_games_played', 'win_pct', 
                 'opp_win_pct', 'Team_SOS', 'Opponent_SOS']

importances = np.abs(w)

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[indices])
plt.xticks(range(len(feature_names)), 
           [feature_names[i] for i in indices], 
           rotation=45, ha='right')
plt.title('SVM Feature Importances (Weight Magnitude)')
plt.tight_layout()
plt.savefig('svm_feature_importance.png')
plt.show()
