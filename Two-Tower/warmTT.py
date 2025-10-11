import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import time
from tqdm import tqdm
import argparse
import os

# 0. CONFIGURATION
SEED = 42
MODEL_PATH = "two_tower_warm_start.pth"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 1. DATA LOAD and PREPROCESS
def load_and_process_data():
    print("Loading data...")
    user_features_df = pd.read_csv('output4.csv')
    item_features_df = pd.read_csv('df_route_capped_normalized.csv').iloc[:, :-2]
    train_df = pd.read_csv('warm_train_data.csv')
    train_df.columns = train_df.columns.str.strip()

    interactions = train_df[['user_id', 'route_id', 'label']]
    train_users, test_users = train_test_split(interactions['user_id'].unique(), test_size=0.2, random_state=SEED)
    train_interactions = interactions[interactions['user_id'].isin(train_users)]
    test_interactions = interactions[interactions['user_id'].isin(test_users)]
    print(f"Data loaded. Train users: {len(train_users)}, Test users: {len(test_users)}")

    # User Features 전처리
    user_numerical_cols = ['hope2run', 'pc_scenery']
    user_categorical_cols = [col for col in user_features_df.columns if col not in ['user_id'] + user_numerical_cols]
    scaler_user = StandardScaler()
    user_features_df[user_numerical_cols] = scaler_user.fit_transform(user_features_df[user_numerical_cols])

    # Item Features 전처리
    item_numerical_cols = [col for col in item_features_df.select_dtypes(include=np.number).columns if col != 'route_id']
    item_features_df = item_features_df.dropna(subset=item_numerical_cols)
    scaler_item = StandardScaler()
    item_features_df[item_numerical_cols] = scaler_item.fit_transform(item_features_df[item_numerical_cols])

    user_features = {
        'numerical': user_numerical_cols,
        'categorical': user_categorical_cols,
        'df': user_features_df.set_index('user_id')
    }
    item_features = {
        'numerical': item_numerical_cols,
        'categorical': [],
        'df': item_features_df.set_index('route_id')
    }
    print("Feature processing complete.")
    
    return train_interactions, test_interactions, user_features, item_features, test_users

# 2. PyTorch Dataset
class RecSysDataset(Dataset):
    def __init__(self, interactions, user_features, item_features):
        self.interactions = interactions
        self.user_features_df = user_features['df']
        self.item_features_df = item_features['df']
        self.user_num_cols = user_features['numerical']
        self.user_cat_cols = user_features['categorical']
        self.item_num_cols = item_features['numerical']

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        item_id = row['route_id']
        label = row['label']

        user_feat_num = self.user_features_df.loc[user_id, self.user_num_cols].values.astype(np.float32)
        user_feat_cat = self.user_features_df.loc[user_id, self.user_cat_cols].values.astype(np.int64)
        item_feat_num = self.item_features_df.loc[item_id, self.item_num_cols].values.astype(np.float32)
        
        return {
            'user_feat_num': torch.tensor(user_feat_num, dtype=torch.float32),
            'user_feat_cat': torch.tensor(user_feat_cat, dtype=torch.long),
            'item_feat_num': torch.tensor(item_feat_num, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

# 3. TwoTower
class TwoTowerModel(nn.Module):
    def __init__(self, user_features, item_features, user_embedding_dim=128, item_embedding_dim=128):
        super().__init__()
        # User Tower
        user_cat_df = user_features['df'][user_features['categorical']]
        self.user_cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=user_cat_df[col].max() + 1, embedding_dim=8)
            for col in user_features['categorical']
        ])
        user_tower_input_dim = len(user_features['numerical']) + len(user_features['categorical']) * 8
        self.user_tower = nn.Sequential(
            nn.Linear(user_tower_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, user_embedding_dim)
        )

        # Item Tower
        item_tower_input_dim = len(item_features['numerical'])
        self.item_tower = nn.Sequential(
            nn.Linear(item_tower_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, item_embedding_dim)
        )

    def forward(self, user_feat_num, user_feat_cat, item_feat_num):
        # User embedding
        user_cat_embs = [emb(user_feat_cat[:, i]) for i, emb in enumerate(self.user_cat_embeddings)]
        user_cat_embs = torch.cat(user_cat_embs, dim=1)
        user_combined = torch.cat([user_feat_num, user_cat_embs], dim=1)
        user_embedding = self.user_tower(user_combined)
        
        # Item embedding
        item_embedding = self.item_tower(item_feat_num)
        
        return user_embedding, item_embedding

# 4. Train and Evaluation
def train(args):
    set_seed(SEED)
    train_interactions, _, user_features, item_features, _ = load_and_process_data()

    train_dataset = RecSysDataset(train_interactions, user_features, item_features)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=0)

    print("\nInitializing model...")
    model = TwoTowerModel(user_features, item_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.warm_start:
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Warm-start successful! Loaded pre-trained weights from {MODEL_PATH}")
        else:
            print(f"No saved model found at {MODEL_PATH}. Starting with a cold-start instead.")
    else:
        print("Starting with a cold-start (random weights).")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 20

    print(f"\nStarting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            user_feat_num = batch['user_feat_num'].to(device)
            user_feat_cat = batch['user_feat_cat'].to(device)
            item_feat_num = batch['item_feat_num'].to(device)
            labels = batch['label'].to(device)

            user_embedding, item_embedding = model(user_feat_num, user_feat_cat, item_feat_num)
            
            logits = torch.sum(user_embedding * item_embedding, dim=1)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        end_time = time.time()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Time: {end_time - start_time:.2f}s")

    print("\nTraining complete.")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model state saved to {MODEL_PATH} for future warm-starts.")


def evaluate():
    set_seed(SEED)
    _, test_interactions, user_features, item_features, test_users = load_and_process_data()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
        return

    print("\nInitializing model for evaluation...")
    model = TwoTowerModel(user_features, item_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    K_list = [1, 10]
    ground_truth = test_interactions[test_interactions['label'] == 1].groupby('user_id')['route_id'].apply(set)
    
    all_item_ids = item_features['df'].index.values
    all_item_num_features = torch.tensor(item_features['df'][item_features['numerical']].values, dtype=torch.float32).to(device)

    results = {f'Recall@{k}': [] for k in K_list}
    results.update({f'HR@{k}': [] for k in K_list})

    with torch.no_grad():
        print("Calculating embeddings for all items...")
        all_item_embeddings = model.item_tower(all_item_num_features)
        
        print(f"Evaluating for {len(test_users)} test users...")
        for user_id in tqdm(ground_truth.index):
            user_feat_num = torch.tensor(user_features['df'].loc[user_id, user_features['numerical']].values, dtype=torch.float32).unsqueeze(0).to(device)
            user_feat_cat = torch.tensor(user_features['df'].loc[user_id, user_features['categorical']].values, dtype=torch.long).unsqueeze(0).to(device)
            
            # Note: The original forward pass for the user tower doesn't need item features,
            # so we can pass a dummy tensor or the actual item features.
            # Here, we only need the user embedding part of the output.
            user_embedding, _ = model(user_feat_num, user_feat_cat, all_item_num_features[0].unsqueeze(0)) # Pass a dummy item feature
            user_embedding = user_embedding[0]
            
            scores = torch.matmul(user_embedding, all_item_embeddings.T)
            _, top_indices = torch.topk(scores, k=max(K_list))
            
            recommended_item_ids = all_item_ids[top_indices.cpu().numpy()]
            true_item_set = ground_truth[user_id]
            
            for k in K_list:
                top_k_recs = set(recommended_item_ids[:k])
                intersection = top_k_recs.intersection(true_item_set)
                
                recall = len(intersection) / len(true_item_set) if len(true_item_set) > 0 else 0.0
                results[f'Recall@{k}'].append(recall)
                
                hit = 1 if len(intersection) > 0 else 0
                results[f'HR@{k}'].append(hit)

    print("\n--- Final Evaluation Results ---")
    for k in K_list:
        avg_recall = np.mean(results[f'Recall@{k}'])
        avg_hr = np.mean(results[f'HR@{k}'])
        print(f"Recall@{k}: {avg_recall:.4f}")
        print(f"HR@{k}    : {avg_hr:.4f}")
        print("--------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Tower Recommendation Model Training and Evaluation")
    parser.add_argument('mode', choices=['train', 'evaluate'], help="Mode to run the script in: 'train' or 'evaluate'")
    parser.add_argument('--warm_start', action='store_true', help="Set this flag to use warm start for training")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate()