# Random forest basketball win/loss classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

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

features = ['Team', 'Opponent', 'games_played_before_date', 'opponent_games_played_before_date', 'win_pct_before_date', 'opponent_win_pct_before_date', 'Team_SOS', 'Opponent_SOS']
X = data[features]
y = data['win'].values
X = pd.get_dummies(X, columns=['Team', 'Opponent'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf_model = RandomForestClassifier(max_depth=64, random_state=0)
rf_model.fit(X_train, y_train)

rf_y_pred_train = rf_model.predict(X_train)
rf_y_pred_test = rf_model.predict(X_test)

CM = confusion_matrix(y_test, rf_y_pred_test)
print (CM)
print(metrics.accuracy_score(y_test, rf_y_pred_test))

y_test_num = (y_test == 1).astype(int)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test_num, rf_probs)
rf_auc = roc_auc_score(y_test_num, rf_probs)

print("Random Forest AUC:", rf_auc)

# ROC curve using predicted win probabilities
plt.figure()
plt.plot(rf_fpr, rf_tpr, linewidth=2, label=f'AUC = {rf_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Test ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_roc_curve.png')

new_row = pd.DataFrame({
    'Team': ['South Carolina'],
    'Opponent': ['UConn'],
    'games_played_before_date': [35],
    'opponent_games_played_before_date': [37],
    'win_pct_before_date': [1.0],
    'opponent_win_pct_before_date': [0.919],
    'Team_SOS': [11.2],
    'Opponent_SOS': [10.8]
})
new_row = pd.get_dummies(new_row, columns=['Team', 'Opponent'], drop_first=True)
new_row = new_row.reindex(columns=X_train.columns, fill_value=0)
print(f"RF predicts: {'Win' if rf_model.predict(new_row)[0] == 1 else 'Loss'}")


import matplotlib.pyplot as plt
import numpy as np

importances = rf_model.feature_importances_
feature_names = X_train.columns

# Sort features by their random forest importance scores
indices = np.argsort(importances)[::-1]

# Plot the top 15 features
plt.figure(figsize=(10, 6))
plt.bar(range(15), importances[indices[:15]])
plt.xticks(range(15), feature_names[indices[:15]], rotation=45, ha='right')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
