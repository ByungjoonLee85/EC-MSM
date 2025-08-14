# tabnet_sklearn.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

# ---------- Core Modules ----------
class Sparsemax(nn.Module):
    def forward(self, z):
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        z_cumsum = torch.cumsum(z_sorted, dim=-1)
        k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
        rhs = 1 + k * z_sorted
        is_gt = (rhs > z_cumsum).to(z.dtype)
        k_z = torch.max(is_gt * k, dim=-1, keepdim=True).values
        z_cumsum_k = torch.gather(z_cumsum, -1, (k_z.long() - 1))
        tau = (z_cumsum_k - 1) / k_z
        p = torch.clamp(z - tau, min=0.0)
        return p

class GLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bn_momentum=0.7):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.bn = nn.BatchNorm1d(out_dim * 2, momentum=bn_momentum)
    def forward(self, x):
        h = self.bn(self.fc(x))
        a, b = torch.chunk(h, 2, dim=-1)
        return a * torch.sigmoid(b)

class FeatureTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, n_glu=2, bn_momentum=0.7):
        super().__init__()
        layers = []
        dims = [in_dim] + [out_dim] * n_glu
        for i in range(n_glu):
            layers.append(GLUBlock(dims[i], out_dim, bn_momentum))
        self.blocks = nn.ModuleList(layers)
        self.skip_scale = math.sqrt(0.5)
    def forward(self, x):
        out = x
        for block in self.blocks:
            h = block(out)
            out = (h + out[:, :h.size(-1)]) * self.skip_scale if out.size(-1) >= h.size(-1) else h
        return out

class AttentiveTransformer(nn.Module):
    def __init__(self, in_dim, feat_dim, bn_momentum=0.7):
        super().__init__()
        self.fc = nn.Linear(in_dim, feat_dim, bias=False)
        self.bn = nn.BatchNorm1d(feat_dim, momentum=bn_momentum)
        self.sparsemax = Sparsemax()
    def forward(self, x, prior):
        a = self.bn(self.fc(x))
        a = a * prior
        M = self.sparsemax(a)
        return M

class TabNetTorch(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=16, n_a=16, n_steps=3, gamma=1.5,
                 n_shared=2, n_independent=2, bn_momentum=0.7):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        feat_dim = input_dim
        if n_shared > 0:
            self.shared_ft = FeatureTransformer(
                in_dim=feat_dim, out_dim=n_d + n_a, n_glu=n_shared, bn_momentum=bn_momentum
            )
        else:
            self.shared_ft = None

        self.step_ft = nn.ModuleList()
        self.attentive = nn.ModuleList()
        for _ in range(n_steps):
            self.step_ft.append(
                FeatureTransformer(
                    in_dim=(n_d + n_a) if self.shared_ft is not None else feat_dim,
                    out_dim=n_d + n_a, n_glu=n_independent, bn_momentum=bn_momentum
                )
            )
            self.attentive.append(
                AttentiveTransformer(in_dim=n_a, feat_dim=feat_dim, bn_momentum=bn_momentum)
            )

        self.head = nn.Linear(n_d, output_dim)
        self.register_buffer("prior", torch.ones(1, feat_dim))

    def forward(self, x):
        B = x.size(0)
        prior = self.prior.expand(B, -1)
        M = None
        aggregated_out = 0.0
        x_in = x

        for step in range(self.n_steps):
            if step == 0:
                M = prior / prior.sum(dim=1, keepdim=True)
            x_masked = x_in * M
            if self.shared_ft is not None:
                h = self.shared_ft(x_masked)
            else:
                h = x_masked
            h = self.step_ft[step](h)
            d, a = torch.split(h, [self.n_d, self.n_a], dim=-1)
            aggregated_out = aggregated_out + F.relu(d)
            a_stop = a
            M = self.attentive[step](a_stop.detach(), prior)
            prior = self.gamma * (prior - M)
        logits = self.head(aggregated_out)
        return logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.output_dim == 1:
                return torch.sigmoid(logits)
            else:
                return F.softmax(logits, dim=-1)

# ---------- sklearn-style Wrapper ----------
class TabNetClassifierSK(BaseEstimator, ClassifierMixin):
    def __init__(self, n_d=16, n_a=16, n_steps=3, gamma=1.5,
                 n_shared=2, n_independent=2, bn_momentum=0.7,
                 lr=1e-3, weight_decay=1e-5, epochs=200, batch_size=None, verbose=0):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.bn_momentum = bn_momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        output_dim = len(self.classes_)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = TabNetTorch(
            input_dim=X.shape[1],
            output_dim=output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_shared=self.n_shared,
            n_independent=self.n_independent,
            bn_momentum=self.bn_momentum
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_t = torch.tensor(X, device=device)
        y_t = torch.tensor(y, device=device)

        N = X_t.size(0)
        bs = N if self.batch_size is None else self.batch_size

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            epoch_loss = 0.0
            for i in range(0, N, bs):
                xb = X_t[i:i+bs]
                yb = y_t[i:i+bs]
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if self.verbose and (epoch % 20 == 0 or epoch == 1 or epoch == self.epochs):
                print(f"[{epoch}/{self.epochs}] loss={epoch_loss/N:.4f}")
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float32)
        device = next(self.model_.parameters()).device
        X_t = torch.tensor(X, device=device)
        probs = self.model_.predict_proba(X_t).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# ---------- Example ----------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = TabNetClassifierSK(epochs=200, batch_size=None, verbose=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, preds))
