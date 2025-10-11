# -*- coding: utf-8 -*-
"""
Warm-Start DCN+DeepFM (two pre-split files) — v2
Fixes:
- DeepFM first-order weighting now matches the *actual* number of fields at runtime.
- Numeric embedding block robust to mismatches between expected/provided numeric dims.
- Optional one-time debug print of field counts on first batch.
"""
import os, sys, math, random, argparse, shutil
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- CLI --------------------
def get_args():
    ap = argparse.ArgumentParser(description="Warm-Start (two files) DCN+DeepFM v2")
    ap.add_argument('--user_feats', type=str, required=True)
    ap.add_argument('--item_feats', type=str, required=True)
    ap.add_argument('--train_pairs', type=str, required=True)
    ap.add_argument('--eval_pairs',  type=str, required=True)

    ap.add_argument('--epochs',      type=int, default=12)
    ap.add_argument('--batch',       type=int, default=1024)
    ap.add_argument('--emb_dim',     type=int, default=16)
    ap.add_argument('--dropout',     type=float, default=0.5)
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--lr_dcn',      type=float, default=1e-4)
    ap.add_argument('--lr_dfm',      type=float, default=1e-3)
    ap.add_argument('--wd',          type=float, default=0.0)
    ap.add_argument('--alpha_dcn',   type=float, default=0.6)
    ap.add_argument('--alpha_dfm',   type=float, default=0.4)

    ap.add_argument('--eval_k',      type=int, default=100)
    ap.add_argument('--eval_every',  type=int, default=3)
    ap.add_argument('--save_dir',    type=str, default='/root')
    ap.add_argument('--no_save',     action='store_true')
    ap.add_argument('--debug_fields_once', action='store_true',
                    help='Print detected field counts for first batch (useful for diagnosing mismatch).')
    return ap.parse_args()

# -------------------- Utils --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_id(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').astype('Int64')
    return s.astype(str)

def sanitize_numeric_df(df: pd.DataFrame, exclude_cols):
    for c in df.columns:
        if c in exclude_cols: continue
        if pd.api.types.is_bool_dtype(df[c]): continue
        if pd.api.types.is_numeric_dtype(df[c]):
            col = df[c].astype(float)
            col = col.replace([np.inf, -np.inf], np.nan)
            if col.isna().any():
                mean = col.mean(skipna=True)
                if pd.isna(mean): mean = 0.0
                col = col.fillna(mean)
            df[c] = col
    return df

def atomic_torch_save(obj, path: str):
    tmp = path + ".tmp"
    cpu_obj = {}
    for k, v in obj.items():
        if hasattr(v, "state_dict"):
            cpu_obj[k] = {kk: vv.cpu() for kk, vv in v.state_dict().items()}
        elif isinstance(v, dict):
            cpu_obj[k] = {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        else:
            cpu_obj[k] = v
    torch.save(cpu_obj, tmp)
    os.replace(tmp, path)

# -------------------- Encoders / Dataset --------------------
class FeatureEncoder:
    def __init__(self):
        self.user_cat_cols=[]; self.user_num_cols=[]
        self.item_cat_cols=[]; self.item_num_cols=[]
        self.user_cat_map={};  self.item_cat_map={}
        self.user_oov_idx={};  self.item_oov_idx={}
        self.user_num_mean={}; self.user_num_std={}
        self.item_num_mean={}; self.item_num_std={}
    def _split_cols(self, df, exclude):
        cat, num = [], []
        for c in df.columns:
            if c in exclude: continue
            if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
                cat.append(c)
            elif pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                num.append(c)
        return cat, num
    def fit(self, user_df, item_df, user_id_col, item_id_col):
        self.user_cat_cols, self.user_num_cols = self._split_cols(user_df, [user_id_col])
        self.item_cat_cols, self.item_num_cols = self._split_cols(item_df, [item_id_col])
        for c in self.user_num_cols:
            m = user_df[c].mean(skipna=True); s = user_df[c].std(skipna=True)
            self.user_num_mean[c] = float(0.0 if pd.isna(m) else m)
            self.user_num_std[c]  = float(1.0 if (pd.isna(s) or s==0) else s)
        for c in self.item_num_cols:
            m = item_df[c].mean(skipna=True); s = item_df[c].std(skipna=True)
            self.item_num_mean[c] = float(0.0 if pd.isna(m) else m)
            self.item_num_std[c]  = float(1.0 if (pd.isna(s) or s==0) else s)
        for c in self.user_cat_cols:
            vals = user_df[c].fillna("__NA__").astype(str).unique().tolist()
            self.user_cat_map[c] = {v:i for i,v in enumerate(vals)}; self.user_oov_idx[c] = len(vals)
        for c in self.item_cat_cols:
            vals = item_df[c].fillna("__NA__").astype(str).unique().tolist()
            self.item_cat_map[c] = {v:i for i,v in enumerate(vals)}; self.item_oov_idx[c] = len(vals)
        return self
    def transform_user_row(self, row):
        u_cat=[]; u_num=[]
        for c in self.user_cat_cols:
            idx = self.user_cat_map[c].get(str(row.get(c,"__NA__")), self.user_oov_idx[c]); u_cat.append(idx)
        for c in self.user_num_cols:
            v = row.get(c, np.nan); v = self.user_num_mean[c] if pd.isna(v) else v
            z = (float(v) - self.user_num_mean[c]) / self.user_num_std[c]; u_num.append(z)
        return np.array(u_cat, np.int64), np.array(u_num, np.float32)
    def transform_item_row(self, row):
        i_cat=[]; i_num=[]
        for c in self.item_cat_cols:
            idx = self.item_cat_map[c].get(str(row.get(c,"__NA__")), self.item_oov_idx[c]); i_cat.append(idx)
        for c in self.item_num_cols:
            v = row.get(c, np.nan); v = self.item_num_mean[c] if pd.isna(v) else v
            z = (float(v) - self.item_num_mean[c]) / self.item_num_std[c]; i_num.append(z)
        return np.array(i_cat, np.int64), np.array(i_num, np.float32)

class PairDataset(Dataset):
    def __init__(self, df_pairs, user_df, item_df, enc: FeatureEncoder):
        self.df = df_pairs.reset_index(drop=True); self.user_df = user_df; self.item_df = item_df; self.enc = enc
        self.ucache={}; self.icache={}
        self.u_cat_len = len(enc.user_cat_cols); self.i_cat_len = len(enc.item_cat_cols)
        self.u_num_len = len(enc.user_num_cols); self.i_num_len = len(enc.item_num_cols)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]; uid, iid, y = r["user_id"], r["item_id"], float(r["label"])
        if uid in self.ucache: u_cat,u_num = self.ucache[uid]
        else:
            try: urow = self.user_df.loc[uid]; u_cat,u_num = self.enc.transform_user_row(urow)
            except KeyError: u_cat = np.zeros((self.u_cat_len,), np.int64); u_num = np.zeros((self.u_num_len,), np.float32)
            self.ucache[uid]=(u_cat,u_num)
        if iid in self.icache: i_cat,i_num = self.icache[iid]
        else:
            try: irow = self.item_df.loc[iid]; i_cat,i_num = self.enc.transform_item_row(irow)
            except KeyError: i_cat = np.zeros((self.i_cat_len,), np.int64); i_num = np.zeros((self.i_num_len,), np.float32)
            self.icache[iid]=(i_cat,i_num)
        u_num = np.nan_to_num(u_num, nan=0.0, posinf=1e6, neginf=-1e6)
        i_num = np.nan_to_num(i_num, nan=0.0, posinf=1e6, neginf=-1e6)
        return (torch.tensor(u_cat, dtype=torch.long), torch.tensor(u_num, dtype=torch.float32),
                torch.tensor(i_cat, dtype=torch.long), torch.tensor(i_num, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

def split_cat_tensors(batch_cat: torch.Tensor):
    return [batch_cat[:,j] for j in range(batch_cat.shape[1])] if batch_cat.ndim==2 and batch_cat.shape[1]>0 else []

# -------------------- Models --------------------
class CatEmbeddingBlock(nn.Module):
    def __init__(self, field_dims, emb_dim):
        super().__init__(); self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim) for n in field_dims])
        for e in self.embs: nn.init.xavier_uniform_(e.weight)
    def forward(self, xs): return [emb(x) for x,emb in zip(xs, self.embs)]

class NumEmbeddingBlock(nn.Module):
    """Robust numeric embedding: gracefully handles mismatch between x.shape[1] and proj count."""
    def __init__(self, num_fields, emb_dim):
        super().__init__(); self.proj = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(num_fields)])
        for p in self.proj: nn.init.xavier_uniform_(p.weight); nn.init.zeros_(p.bias)
    def forward(self, x):
        if x.ndim != 2:
            return []  # unexpected, skip
        n_in = x.shape[1]; n_proj = len(self.proj)
        n = min(n_in, n_proj)
        outs = [self.proj[j](x[:,j:j+1]) for j in range(n)]
        if n < n_proj:
            B = x.shape[0]; out_dim = self.proj[0].out_features if n_proj>0 else 0
            for _ in range(n_proj - n):
                outs.append(torch.zeros((B, out_dim), device=x.device, dtype=x.dtype))
        return outs

class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__(); self.ws = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        for w in self.ws: nn.init.xavier_uniform_(w.unsqueeze(0))
    def forward(self, x0):
        x = x0
        for w,b in zip(self.ws, self.bs):
            wx = torch.matmul(x, w)[:, None] + b
            x = x0 * wx + x
        return x

class DCNModel(nn.Module):
    def __init__(self, u_cat_dims, i_cat_dims, u_num_dim, i_num_dim, emb_dim=16, cross_layers=3, dnn_layers=[256,128,64], dropout=0.5):
        super().__init__()
        self.u_cat = CatEmbeddingBlock(u_cat_dims, emb_dim); self.i_cat = CatEmbeddingBlock(i_cat_dims, emb_dim)
        self.u_num = NumEmbeddingBlock(u_num_dim, emb_dim) if u_num_dim>0 else None
        self.i_num = NumEmbeddingBlock(i_num_dim, emb_dim) if i_num_dim>0 else None
        field_cnt = len(u_cat_dims)+len(i_cat_dims)+u_num_dim+i_num_dim; input_dim = field_cnt * emb_dim
        self.cross = CrossNet(input_dim, num_layers=cross_layers)
        layers=[]; d=input_dim
        for h in dnn_layers: layers += [nn.Linear(d,h), nn.ReLU(), nn.Dropout(dropout)]; d=h
        self.deep = nn.Sequential(*layers); self.out = nn.Linear(input_dim + d, 1)
    def forward(self, u_cat_list, u_num, i_cat_list, i_num):
        fields = []; fields += self.u_cat(u_cat_list) if len(u_cat_list)>0 else []; fields += self.i_cat(i_cat_list) if len(i_cat_list)>0 else []
        fields += self.u_num(u_num) if self.u_num else []; fields += self.i_num(i_num) if self.i_num else []
        x = torch.cat(fields, dim=1) if fields else torch.zeros((u_num.shape[0],0), device=u_num.device, dtype=u_num.dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        cross_out = self.cross(x); deep_out  = self.deep(x)
        out = torch.cat([cross_out, deep_out], dim=1); return self.out(out).squeeze(1)

class DeepFM(nn.Module):
    def __init__(self, u_cat_dims, i_cat_dims, u_num_dim, i_num_dim, emb_dim=16, dnn_layers=[256,128,64], dropout=0.5):
        super().__init__()
        self.u_cat = CatEmbeddingBlock(u_cat_dims, emb_dim); self.i_cat = CatEmbeddingBlock(i_cat_dims, emb_dim)
        self.u_num = NumEmbeddingBlock(u_num_dim, emb_dim) if u_num_dim>0 else None
        self.i_num = NumEmbeddingBlock(i_num_dim, emb_dim) if i_num_dim>0 else None
        self.field_cnt_expected = len(u_cat_dims)+len(i_cat_dims)+u_num_dim+i_num_dim
        self.deep_in = self.field_cnt_expected * emb_dim
        layers=[]; d=self.deep_in
        for h in dnn_layers: layers += [nn.Linear(d,h), nn.ReLU(), nn.Dropout(dropout)]; d=h
        self.deep = nn.Sequential(*layers)
        # first-order weights sized to expected fields; we'll slice/pad at runtime to match actual fields
        self.first_order = nn.Parameter(torch.zeros(self.field_cnt_expected))
        self.out = nn.Linear(d + 2, 1)
        self._debug_printed = False
    def fm_second_order(self, X):
        sum_v = X.sum(dim=1)
        sum_v_sq = (sum_v*sum_v).sum(dim=1, keepdim=True)
        v_sq_sum = (X*X).sum(dim=2).sum(dim=1, keepdim=True)
        return 0.5*(sum_v_sq - v_sq_sum)
    def forward(self, u_cat_list, u_num, i_cat_list, i_num):
        fields = []; fields += self.u_cat(u_cat_list) if len(u_cat_list)>0 else []; fields += self.i_cat(i_cat_list) if len(i_cat_list)>0 else []
        fields += self.u_num(u_num) if self.u_num else []; fields += self.i_num(i_num) if self.i_num else []
        if len(fields)==0:
            # degenerate, avoid crash
            B = u_num.shape[0] if isinstance(u_num, torch.Tensor) else (i_num.shape[0] if isinstance(i_num, torch.Tensor) else 1)
            X = torch.zeros((B,1,self.deep[0].in_features//1), device=u_cat_list[0].device if len(u_cat_list)>0 else 'cpu')
        else:
            X = torch.stack(fields, dim=1)
        # One-time debug
        if (not self._debug_printed) and (os.environ.get("DFM_DEBUG_ONCE","0")=="1"):
            print(f"[DeepFM] expected_fields={self.field_cnt_expected}, actual_fields={X.shape[1]}", flush=True)
            self._debug_printed = True
        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        # First-order with runtime-safe slicing/padding
        w = self.first_order
        f_cur = X.shape[1]; f_exp = w.shape[0]
        if f_cur < f_exp:
            w_used = w[:f_cur]
        elif f_cur > f_exp:
            pad = torch.zeros((f_cur - f_exp,), device=w.device, dtype=w.dtype)
            w_used = torch.cat([w, pad], dim=0)
        else:
            w_used = w
        first = (X.mean(dim=2) * w_used).sum(dim=1, keepdim=True)
        # Second-order + Deep
        second = self.fm_second_order(X)
        deep_in = torch.cat([t for t in fields], dim=1) if len(fields)>0 else torch.zeros((X.shape[0], self.deep_in), device=X.device, dtype=X.dtype)
        deep_in = torch.nan_to_num(deep_in, nan=0.0, posinf=1e6, neginf=-1e6)
        # If deep_in width mismatches expected, pad/truncate to match
        if deep_in.shape[1] != self.deep[0].in_features:
            if deep_in.shape[1] < self.deep[0].in_features:
                pad = torch.zeros((deep_in.shape[0], self.deep[0].in_features - deep_in.shape[1]), device=deep_in.device, dtype=deep_in.dtype)
                deep_in = torch.cat([deep_in, pad], dim=1)
            else:
                deep_in = deep_in[:, :self.deep[0].in_features]
        deep_out = self.deep(deep_in)
        out = torch.cat([deep_out, first, second], dim=1); return self.out(out).squeeze(1)

# -------------------- Metrics --------------------
def hr_ndcg_userwise(scores: np.ndarray, pos_idx_set: set, k: int = 100) -> Tuple[float,float,float]:
    order = np.argsort(-scores)[:k]
    hits_in_topk = [i for i in order if i in pos_idx_set]
    hr = 1.0 if len(hits_in_topk) > 0 else 0.0
    recall = len(hits_in_topk) / max(1, len(pos_idx_set))
    dcg = 0.0
    for rank, idx in enumerate(order, start=1):
        if idx in pos_idx_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(pos_idx_set), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1)) if ideal_hits > 0 else 1.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return hr, ndcg, recall

def hit_at_1_and_hr10(scores: np.ndarray, pos_idx_set: set) -> Tuple[float,float]:
    order = np.argsort(-scores)
    hit1 = 1.0 if (len(order)>0 and (order[0] in pos_idx_set)) else 0.0
    top10 = order[:10]
    hr10 = 1.0 if any(i in pos_idx_set for i in top10) else 0.0
    return hit1, hr10

# -------------------- Train / Eval --------------------
class PairLoader:
    @staticmethod
    def read_pairs(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if 'item_id' not in df.columns and 'route_id' in df.columns:
            df = df.rename(columns={'route_id':'item_id'})
        need = {'user_id','item_id','label'}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Pairs file {path} misses columns: {missing}")
        df['user_id'] = normalize_id(df['user_id'])
        df['item_id'] = normalize_id(df['item_id'])
        df = df[~df['user_id'].eq('<NA>') & ~df['item_id'].eq('<NA>')]
        return df[['user_id','item_id','label']]

def train_one_epoch(model_dcn, model_dfm, dataloader, opt_dcn, opt_dfm, criterion, device, desc_text="Training", debug_once=False):
    model_dcn.train(); model_dfm.train(); n=0; loss_d=0.0; loss_f=0.0
    printed = False
    for u_cat, u_num, i_cat, i_num, y in tqdm(dataloader, desc=desc_text, leave=False):
        u_cat=u_cat.to(device); u_num=u_num.to(device); i_cat=i_cat.to(device); i_num=i_num.to(device); y=y.to(device)
        u_cat_list=split_cat_tensors(u_cat); i_cat_list=split_cat_tensors(i_cat)
        if debug_once and (not printed):
            print(f"[DEBUG] batch shapes: u_cat={u_cat.shape}, u_num={u_num.shape}, i_cat={i_cat.shape}, i_num={i_num.shape}", flush=True)
            printed = True
        opt_dcn.zero_grad(); logit_d = model_dcn(u_cat_list, u_num, i_cat_list, i_num); l_d = criterion(logit_d, y); l_d.backward()
        torch.nn.utils.clip_grad_norm_(model_dcn.parameters(), max_norm=5.0); opt_dcn.step()
        opt_dfm.zero_grad(); logit_f = model_dfm(u_cat_list, u_num, i_cat_list, i_num); l_f = criterion(logit_f, y); l_f.backward()
        torch.nn.utils.clip_grad_norm_(model_dfm.parameters(), max_norm=5.0); opt_dfm.step()
        loss_d += l_d.item()*y.size(0); loss_f += l_f.item()*y.size(0); n += y.size(0)
    return (loss_d/n if n>0 else float('nan')), (loss_f/n if n>0 else float('nan'))

def evaluate_userwise(models, eval_df: pd.DataFrame,
                      user_raw: pd.DataFrame, item_raw: pd.DataFrame,
                      enc: 'FeatureEncoder', args, device):
    model_dcn, model_dfm = models
    model_dcn.eval(); model_dfm.eval()
    HIT1s=[]; HR10s=[]; HRs=[]; NDCGs=[]; RECALLs=[]
    eval_k = args.eval_k
    eval_group = eval_df.groupby("user_id")
    with torch.no_grad():
        for uid, group in tqdm(eval_group, desc="Evaluating (Userwise)", leave=False):
            labels = group["label"].tolist()
            pos_idx_set = set(i for i,l in enumerate(labels) if l==1)
            if not pos_idx_set: continue
            try: urow = user_raw.loc[uid]
            except KeyError: continue
            u_cat_idx, u_num_val = enc.transform_user_row(urow)
            candidates = group["item_id"].tolist(); num_candidates=len(candidates)
            u_cat_batch = np.tile(u_cat_idx, (num_candidates, 1)); u_num_batch = np.tile(u_num_val, (num_candidates, 1))
            i_cat_batch=[]; i_num_batch=[]
            for it in candidates:
                try: irow = item_raw.loc[it]
                except KeyError: irow = None
                if irow is None:
                    ic = np.zeros((len(enc.item_cat_cols),), dtype=np.int64); inm = np.zeros((len(enc.item_num_cols),), dtype=np.float32)
                else:
                    ic, inm = enc.transform_item_row(irow)
                i_cat_batch.append(ic); i_num_batch.append(inm)
            i_cat_batch = np.stack(i_cat_batch, axis=0) if len(i_cat_batch)>0 else np.zeros((num_candidates,0),dtype=np.int64)
            i_num_batch = np.stack(i_num_batch, axis=0) if len(i_num_batch)>0 else np.zeros((num_candidates,0),dtype=np.float32)
            u_cat_t = torch.tensor(u_cat_batch, dtype=torch.long, device=device)
            u_num_t = torch.tensor(np.nan_to_num(u_num_batch, nan=0.0, posinf=1e6, neginf=-1e6), dtype=torch.float32, device=device)
            i_cat_t = torch.tensor(i_cat_batch, dtype=torch.long, device=device)
            i_num_t = torch.tensor(np.nan_to_num(i_num_batch, nan=0.0, posinf=1e6, neginf=-1e6), dtype=torch.float32, device=device)
            u_cat_list = split_cat_tensors(u_cat_t); i_cat_list = split_cat_tensors(i_cat_t)

            logit_d = model_dcn(u_cat_list, u_num_t, i_cat_list, i_num_t)
            logit_f = model_dfm(u_cat_list, u_num_t, i_cat_list, i_num_t)
            scores = torch.sigmoid(args.alpha_dcn*logit_d + args.alpha_dfm*logit_f).detach().cpu().numpy()

            hit1, hr10 = hit_at_1_and_hr10(scores, pos_idx_set)
            hr, ndcg, recall = hr_ndcg_userwise(scores, pos_idx_set, k=eval_k)

            HIT1s.append(hit1); HR10s.append(hr10); HRs.append(hr); NDCGs.append(ndcg); RECALLs.append(recall)
    metrics = {
        "Hit@1": float(np.mean(HIT1s)) if HIT1s else 0.0,
        "HR@10": float(np.mean(HR10s)) if HR10s else 0.0,
        "HR@{}".format(eval_k): float(np.mean(HRs)) if HRs else 0.0,
        "NDCG@{}".format(eval_k): float(np.mean(NDCGs)) if NDCGs else 0.0,
        "Recall@{}".format(eval_k): float(np.mean(RECALLs)) if RECALLs else 0.0,
        "Users(evaluated)": int(len(HIT1s))
    }
    return metrics

# -------------------- Main --------------------
def main():
    args = get_args(); set_seed(args.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = args.save_dir or '.'
    os.makedirs(save_dir, exist_ok=True)
    FINAL_MODEL_PATH = os.path.join(save_dir, f'warm_dcn_dfm_e{args.epochs}_k{args.eval_k}.pth')

    print(f"1. 데이터 로드 및 전처리 시작 (Device: {DEVICE})")
    user_raw = pd.read_csv(args.user_feats)
    item_raw = pd.read_csv(args.item_feats)

    # item cleanup
    cols_to_drop = ['crossing_score_capped', 'crossing_score_normalized']
    item_raw = item_raw.drop(columns=cols_to_drop, errors='ignore')

    # detect id cols (prefer explicit names, fall back to first col if needed)
    user_id_col = 'user_id' if 'user_id' in user_raw.columns else user_raw.columns[0]
    item_id_col_raw = 'item_id' if 'item_id' in item_raw.columns else ('route_id' if 'route_id' in item_raw.columns else item_raw.columns[0])
    if item_id_col_raw != "item_id":
        item_raw = item_raw.rename(columns={item_id_col_raw: "item_id"})

    # numeric sanitize
    user_raw = sanitize_numeric_df(user_raw, exclude_cols=[user_id_col])
    item_raw = sanitize_numeric_df(item_raw, exclude_cols=["item_id"])

    # normalize IDs and set index
    user_raw[user_id_col] = normalize_id(user_raw[user_id_col])
    item_raw["item_id"]    = normalize_id(item_raw["item_id"])
    user_raw = user_raw.drop_duplicates(subset=[user_id_col]).set_index(user_id_col)
    item_raw = item_raw.drop_duplicates(subset=["item_id"]).set_index("item_id")
    user_raw.index = normalize_id(user_raw.index.to_series())
    item_raw.index = normalize_id(item_raw.index.to_series())

    # pairs
    def read_pairs(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if 'item_id' not in df.columns and 'route_id' in df.columns:
            df = df.rename(columns={'route_id':'item_id'})
        need = {'user_id','item_id','label'}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Pairs file {path} misses columns: {missing}")
        df['user_id'] = normalize_id(df['user_id'])
        df['item_id'] = normalize_id(df['item_id'])
        df = df[~df['user_id'].eq('<NA>') & ~df['item_id'].eq('<NA>')]
        return df[['user_id','item_id','label']]

    train_pairs = read_pairs(args.train_pairs)
    eval_pairs  = read_pairs(args.eval_pairs)

    # filter pairs to users/items present in features (safety)
    train_pairs = train_pairs[train_pairs['user_id'].isin(user_raw.index) & train_pairs['item_id'].isin(item_raw.index)]
    eval_pairs  = eval_pairs [eval_pairs ['user_id'].isin(user_raw.index) & eval_pairs ['item_id'].isin(item_raw.index)]

    print(f"[INFO] Train pairs: {len(train_pairs)} (Pos: {int((train_pairs['label']==1).sum())}, Neg: {int((train_pairs['label']==0).sum())})")
    print(f"[INFO] Eval  pairs: {len(eval_pairs)}  (Pos: {int((eval_pairs['label']==1).sum())}, Neg: {int((eval_pairs['label']==0).sum())})")
    print(f"[INFO] Users (train/eval): {train_pairs['user_id'].nunique()} / {eval_pairs['user_id'].nunique()}")

    # encoder on TRAIN users only
    train_users = train_pairs["user_id"].unique()
    enc = FeatureEncoder().fit(user_raw.loc[train_users], item_raw, user_raw.index.name, "item_id")
    u_cat_dims = [len(enc.user_cat_map[c]) for c in enc.user_cat_cols]
    i_cat_dims = [len(enc.item_cat_map[c]) for c in enc.item_cat_cols]
    u_num_dim = len(enc.user_num_cols); i_num_dim = len(enc.item_num_cols)
    print(f"[INFO] Feature Dims - User Num/Cat: {u_num_dim}/{len(u_cat_dims)}, Item Num/Cat: {i_num_dim}/{len(i_cat_dims)}")

    # dataloader
    train_ds = PairDataset(train_pairs, user_raw, item_raw, enc)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)

    # models/opt
    dcn = DCNModel(u_cat_dims, i_cat_dims, u_num_dim, i_num_dim, emb_dim=args.emb_dim, dropout=args.dropout).to(DEVICE)
    dfm = DeepFM(u_cat_dims, i_cat_dims, u_num_dim, i_num_dim, emb_dim=args.emb_dim, dropout=args.dropout).to(DEVICE)
    opt_dcn = torch.optim.Adam(dcn.parameters(), lr=args.lr_dcn, weight_decay=args.wd)
    opt_dfm = torch.optim.Adam(dfm.parameters(), lr=args.lr_dfm, weight_decay=args.wd)
    crit = nn.BCEWithLogitsLoss()

    print("\n" + "="*70)
    print(f"  DCN+DeepFM 웜스타트 훈련 시작 ({args.epochs} Epochs)")
    print("="*70)

    best_ndcg = 0.0
    for epoch in range(1, args.epochs+1):
        ld, lf = train_one_epoch(dcn, dfm, train_loader, opt_dcn, opt_dfm, crit, DEVICE,
                                 desc_text=f"Training Epoch {epoch}/{args.epochs}",
                                 debug_once=args.debug_fields_once)
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            metrics = evaluate_userwise((dcn, dfm), eval_pairs, user_raw, item_raw, enc, args, DEVICE)
            print(f"[Epoch {epoch:02d}] DCN loss={ld:.6f}  DeepFM loss={lf:.6f} | "
                  f"Hit@1: {metrics['Hit@1']:.4f} | HR@10: {metrics['HR@10']:.4f} | "
                  f"HR@{args.eval_k}: {metrics['HR@%d'%args.eval_k]:.4f} | "
                  f"NDCG@{args.eval_k}: {metrics['NDCG@%d'%args.eval_k]:.4f} | "
                  f"Users: {metrics['Users(evaluated)']}",
                  flush=True)
            if (not args.no_save) and (metrics['NDCG@%d'%args.eval_k] > best_ndcg):
                best_ndcg = metrics['NDCG@%d'%args.eval_k]
                try:
                    atomic_torch_save({'dcn': dcn, 'dfm': dfm}, FINAL_MODEL_PATH)
                    print(f"[INFO] 최고 성능 모델 저장됨: {FINAL_MODEL_PATH} (Best NDCG={best_ndcg:.4f})", flush=True)
                except Exception as e:
                    print(f"[WARN] 체크포인트 저장 실패: {e}", flush=True)
        else:
            print(f"[Epoch {epoch:02d}] DCN loss={ld:.6f}  DeepFM loss={lf:.6f}", flush=True)

    print("\n" + "="*70)
    print(f"훈련 완료! 최고 NDCG@{args.eval_k}: {best_ndcg:.4f} (저장 경로: {FINAL_MODEL_PATH})")
    print("="*70)

if __name__ == "__main__":
    main()
