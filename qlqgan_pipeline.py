

import os, sys, math, shutil, random, warnings, subprocess, importlib

def _ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except Exception:
        print(f"[setup] installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

for _pkg, _imp in [("torchmetrics", "torchmetrics"),
                   ("scikit-image", "skimage"),
                   ("torch-fidelity", "torch_fidelity")]:  # torch-fidelity is the FID backend torchmetrics needs
    _ensure(_pkg, _imp)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset, Dataset
from torchvision import datasets, transforms, utils, models
import pennylane as qml
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import wilcoxon, ttest_rel

warnings.filterwarnings("ignore")

# ============================================================================
#  Config
# ============================================================================
QUICK_MODE = False        # True -> tiny run to verify the pipeline end-to-end

IMG_SIZE        = 128
LATENT_DIM      = 16
N_QUBITS        = 4       # quantum noise dimensionality (raise to increase expressivity)
Q_DEPTH         = 2
CLASSICAL_NOISE = N_QUBITS # match the noise dimensionality of the quantum generator

BATCH_SIZE      = 64
VAE_EPOCHS      = 12
GAN_EPOCHS      = 80
CLS_EPOCHS      = 6
BETA_KL         = 1e-3    # small KL weight: regularize latent without over-blurring recon

GAN_BATCH       = 64
N_CRITIC        = 5
GP_LAMBDA       = 10.0

SYN_PER_CLASS   = 300

# the experiment grid
FRACTIONS        = [0.05, 0.10, 0.25, 1.00]   # added 0.05: scarcer data = larger augmentation effect
SEEDS            = list(range(8))             # 8 seeds for real statistical power
METRIC_FRACTIONS = [0.05, 0.10, 0.25, 1.00]   # FID + t-SNE at EVERY fraction (shows low-data collapse)

N_FID_REAL      = 1000                 # real images used for FID
DIV_PAIRS       = 300                  # pairs sampled for within-set diversity SSIM
OUT_DIR         = "qlqgan_results"

if QUICK_MODE:
    VAE_EPOCHS, GAN_EPOCHS, CLS_EPOCHS = 2, 6, 2
    FRACTIONS, SEEDS = [0.05, 0.10], [0, 1]
    SYN_PER_CLASS, N_FID_REAL = 50, 200
    METRIC_FRACTIONS = [0.05, 0.10]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================================
#  Data
# ============================================================================
# ============================================================================
#  Data
#  Public dataset: "Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary,
#  No Tumor)", Mendeley Data V1, 2025, doi:10.17632/zwr4ntf94j.1
#  Download it and arrange as ImageFolder:
#      <DATA_ROOT>/Train/<class>/*.jpg
#      <DATA_ROOT>/Test/<class>/*.jpg
#  Set the location via the BRAIN_MRI_DATA environment variable, or edit below.
# ============================================================================
DATA_ROOT = os.environ.get("BRAIN_MRI_DATA", "brain_tumor_mri_dataset")

# Optional Google Colab + Drive convenience (used in the original runtime):
if not os.path.exists(DATA_ROOT):
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        _candidate = "/content/drive/MyDrive/brain_tumor_mri_dataset"
        if os.path.exists(_candidate):
            DATA_ROOT = _candidate
    except Exception as e:
        print("Colab/Drive unavailable; using DATA_ROOT =", DATA_ROOT, "(", e, ")")

TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
TEST_DIR  = os.path.join(DATA_ROOT, "Test")

tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),     # -> [-1, 1]
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tfms)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=tfms)
classes  = train_ds.classes
num_classes = len(classes)
print("Classes:", classes, "| train:", len(train_ds), "| test:", len(test_ds))

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def stratified_indices(targets, frac, seed):
    """Class-balanced subsample of indices."""
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    idx = []
    for c in np.unique(targets):
        ci = np.where(targets == c)[0]
        rng.shuffle(ci)
        k = max(1, int(round(frac * len(ci))))
        idx.extend(ci[:k].tolist())
    rng.shuffle(idx)
    return idx


class TensorImageDataset(Dataset):
    """Synthetic images as (img_tensor, int_label) so it concatenates cleanly with ImageFolder Subsets."""
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        return self.imgs[i], int(self.labels[i])


# ============================================================================
#  VAE (KL-regularized autoencoder)
# ============================================================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   nn.ReLU(),   # 64
            nn.Conv2d(32, 64, 4, 2, 1),  nn.ReLU(),   # 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),   # 16
            nn.Conv2d(128, 256, 4, 2, 1),nn.ReLU(),   # 8
        )
        self.flatten   = nn.Flatten()
        self.fc_mu     = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(),  # 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(),  # 64
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    nn.Tanh(),  # 128
        )

    def encode(self, x):
        h = self.flatten(self.encoder_cnn(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def encode_mu(self, x):
        return self.encode(x)[0]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def train_vae(dataset, epochs):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    vae = ConvVAE(LATENT_DIM).to(device)
    opt = optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        vae.train(); tot, tot_r, tot_k = 0, 0, 0
        for imgs, _ in tqdm(loader, desc=f"VAE {epoch+1}/{epochs}", leave=False):
            imgs = imgs.to(device)
            opt.zero_grad()
            recon, mu, logvar = vae(imgs)
            rec = F.mse_loss(recon, imgs)
            kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + BETA_KL * kl
            loss.backward(); opt.step()
            tot += loss.item(); tot_r += rec.item(); tot_k += kl.item()
        n = len(loader)
        print(f"  VAE {epoch+1}: loss={tot/n:.4f}  recon={tot_r/n:.4f}  kl={tot_k/n:.4f}")
    return vae


def extract_latents(vae, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    Z, Y = [], []
    vae.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            Z.append(vae.encode_mu(imgs.to(device)).cpu())
            Y.append(torch.as_tensor(labels))
    return torch.cat(Z), torch.cat(Y)


# ============================================================================
#  Quantum circuit (batched) + generators + critic (WGAN-GP)
# ============================================================================
qdev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(qdev, interface="torch", diff_method="backprop")
def quantum_circuit(noise, weights):
    # noise: (batch, N_QUBITS) ; weights: (Q_DEPTH, N_QUBITS, 2)
    # PennyLane broadcasts over the leading batch dim of `noise`.
    for i in range(N_QUBITS):
        qml.RY(noise[:, i], wires=i)
    for d in range(Q_DEPTH):
        for i in range(N_QUBITS):
            qml.RY(weights[d, i, 0], wires=i)
            qml.RZ(weights[d, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QGenerator(nn.Module):
    """Stochasticity enters ONLY through the quantum circuit (the 'quantum generator' claim)."""
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(0.1 * torch.randn(Q_DEPTH, N_QUBITS, 2))
        self.label_emb = nn.Embedding(num_classes, N_QUBITS)
        self.fc = nn.Sequential(
            nn.Linear(N_QUBITS * 2, 64), nn.ReLU(),
            nn.Linear(64, LATENT_DIM),
        )

    def forward(self, labels):
        bs = labels.size(0)
        noise = torch.rand(bs, N_QUBITS) * math.pi            # CPU, uniform RY angles
        w = self.q_weights.to("cpu")                          # quantum eval on CPU (gradients still flow)
        meas = quantum_circuit(noise, w)                      # list of (bs,) tensors
        meas = torch.stack(meas, dim=1).float().to(labels.device)  # (bs, N_QUBITS)
        emb = self.label_emb(labels)
        return self.fc(torch.cat([meas, emb], dim=1))


class ClassicalGenerator(nn.Module):
    """Parameter-matched classical baseline: same noise dim, same conditioning, MLP mapping."""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, CLASSICAL_NOISE)
        self.fc = nn.Sequential(
            nn.Linear(CLASSICAL_NOISE * 2, 64), nn.ReLU(),
            nn.Linear(64, LATENT_DIM),
        )

    def forward(self, labels):
        bs = labels.size(0)
        noise = torch.randn(bs, CLASSICAL_NOISE, device=labels.device)
        emb = self.label_emb(labels)
        return self.fc(torch.cat([noise, emb], dim=1))


class Critic(nn.Module):
    """WGAN-GP critic: no sigmoid, LayerNorm (BatchNorm breaks the gradient penalty)."""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, LATENT_DIM)
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM * 2, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64),             nn.LayerNorm(64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, z, labels):
        return self.net(torch.cat([z, self.label_emb(labels)], dim=1))


def gradient_penalty(critic, real, fake, labels):
    bs = real.size(0)
    eps = torch.rand(bs, 1, device=real.device)
    inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(inter, labels)
    grads = torch.autograd.grad(
        outputs=score, inputs=inter,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_wgan(Z, Y, gen_type, epochs):
    ds = TensorDataset(Z, Y)
    loader = DataLoader(ds, batch_size=GAN_BATCH, shuffle=True, drop_last=True, num_workers=2)
    G = (QGenerator() if gen_type == "quantum" else ClassicalGenerator()).to(device)
    D = Critic().to(device)
    optG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    print(f"  [{gen_type}] generator params: {count_params(G)} | critic params: {count_params(D)}")
    for epoch in range(epochs):
        gtot, dtot, gn = 0.0, 0.0, 0
        for i, (z, y) in enumerate(loader):
            z, y = z.to(device), y.to(device)
            # ---- critic ----
            fake = G(y).detach()
            optD.zero_grad()
            d_loss = D(fake, y).mean() - D(z, y).mean() + GP_LAMBDA * gradient_penalty(D, z, fake, y)
            d_loss.backward(); optD.step()
            dtot += d_loss.item()
            # ---- generator ----
            if i % N_CRITIC == 0:
                optG.zero_grad()
                g_loss = -D(G(y), y).mean()
                g_loss.backward(); optG.step()
                gtot += g_loss.item(); gn += 1
        if (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"  [{gen_type}] epoch {epoch+1}/{epochs} | D={dtot/len(loader):.3f} | G={gtot/max(1,gn):.3f}")
    return G


# ============================================================================
#  Synthetic image generation
# ============================================================================
def generate_synthetic(G, vae, n_per_class):
    G.eval(); vae.eval()
    imgs_all, lbls_all = [], []
    with torch.no_grad():
        for c in range(num_classes):
            labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
            zs = [G(labels[s:s + 256]) for s in range(0, n_per_class, 256)]
            z = torch.cat(zs)
            imgs = [vae.decode(z[s:s + 128]).cpu() for s in range(0, z.size(0), 128)]
            imgs_all.append(torch.cat(imgs))
            lbls_all.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(imgs_all), torch.cat(lbls_all)


def save_grid(imgs, path, title):
    grid = utils.make_grid((imgs[:32] + 1) / 2, nrow=8)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0).squeeze().numpy(), cmap="gray")
    plt.axis("off"); plt.title(title)
    plt.savefig(path, bbox_inches="tight", dpi=120); plt.close()


# ============================================================================
#  Generation-quality metrics: FID, SSIM, t-SNE
# ============================================================================
def _prep_for_fid(x):
    x = (x + 1) / 2
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x.clamp(0, 1)


def compute_fid(real_imgs, fake_imgs):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(normalize=True).to(device)
        for s in range(0, real_imgs.size(0), 64):
            fid.update(_prep_for_fid(real_imgs[s:s + 64]).to(device), real=True)
        for s in range(0, fake_imgs.size(0), 64):
            fid.update(_prep_for_fid(fake_imgs[s:s + 64]).to(device), real=False)
        return float(fid.compute())
    except Exception as e:
        print("  FID failed:", e)
        return float("nan")


def compute_ssim(syn_imgs, syn_lbls, real_pool_imgs, real_pool_lbls, n=200):
    """Rough fidelity proxy: mean SSIM between a synthetic image and a random real image of the SAME class.
       (Not a diversity measure; report alongside FID, which is the primary metric.)"""
    try:
        from skimage.metrics import structural_similarity as ssim
        rng = np.random.RandomState(0)
        by_class = {c: np.where(real_pool_lbls.numpy() == c)[0] for c in range(num_classes)}
        vals = []
        for _ in range(n):
            i = rng.randint(len(syn_imgs)); c = int(syn_lbls[i])
            if len(by_class[c]) == 0:
                continue
            j = by_class[c][rng.randint(len(by_class[c]))]
            a = ((syn_imgs[i, 0] + 1) / 2).numpy()
            b = ((real_pool_imgs[j, 0] + 1) / 2).numpy()
            vals.append(ssim(a, b, data_range=1.0))
        return float(np.mean(vals)) if vals else float("nan")
    except Exception as e:
        print("  SSIM failed:", e)
        return float("nan")


def compute_diversity(syn_imgs, n_pairs=300):
    """Within-set diversity = how varied the synthetic batch is (detects mode collapse).
       Returns (mean_pairwise_ssim, mean_pixel_std).
         - mean_pairwise_ssim HIGH  -> samples look alike -> collapse (bad)
         - mean_pixel_std       LOW  -> little variation across samples -> collapse (bad)"""
    try:
        from skimage.metrics import structural_similarity as ssim
        rng = np.random.RandomState(0)
        imgs01 = ((syn_imgs + 1) / 2).clamp(0, 1)
        # pixel std across the batch, averaged over all pixels
        pixel_std = float(imgs01.std(dim=0).mean().item())
        # mean pairwise SSIM between random distinct synthetic samples
        vals = []
        N = imgs01.size(0)
        for _ in range(n_pairs):
            i, j = rng.randint(N), rng.randint(N)
            if i == j:
                continue
            a = imgs01[i, 0].numpy(); b = imgs01[j, 0].numpy()
            vals.append(ssim(a, b, data_range=1.0))
        pairwise = float(np.mean(vals)) if vals else float("nan")
        return pairwise, pixel_std
    except Exception as e:
        print("  diversity failed:", e)
        return float("nan"), float("nan")


def plot_tsne(vae, real_imgs, real_lbls, syn_imgs, syn_lbls, path, frac, n=600):
    try:
        from sklearn.manifold import TSNE
        vae.eval()
        def enc(imgs):
            zs = []
            with torch.no_grad():
                for s in range(0, imgs.size(0), 128):
                    zs.append(vae.encode_mu(imgs[s:s + 128].to(device)).cpu())
            return torch.cat(zs).numpy()
        ri = np.random.RandomState(0).choice(len(real_imgs), min(n, len(real_imgs)), replace=False)
        si = np.random.RandomState(1).choice(len(syn_imgs),  min(n, len(syn_imgs)),  replace=False)
        Zr, Zs = enc(real_imgs[ri]), enc(syn_imgs[si])
        emb = TSNE(n_components=2, init="pca", random_state=0).fit_transform(np.vstack([Zr, Zs]))
        er, es = emb[:len(Zr)], emb[len(Zr):]
        plt.figure(figsize=(7, 6))
        plt.scatter(er[:, 0], er[:, 1], s=8, alpha=0.5, label="real", marker="o")
        plt.scatter(es[:, 0], es[:, 1], s=8, alpha=0.5, label="synthetic", marker="x")
        plt.legend(); plt.title(f"t-SNE of real vs synthetic latents (f={frac})"); plt.axis("off")
        plt.savefig(path, bbox_inches="tight", dpi=120); plt.close()
    except Exception as e:
        print("  t-SNE failed:", e)


# ============================================================================
#  Classifier (pretrained ResNet-18) — a strong, realistic baseline
# ============================================================================
def build_resnet(num_classes):
    try:
        w = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=w)
    except Exception:
        net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


class GrayResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = build_resnet(num_classes)
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net(x)


def train_eval_classifier(real_subset, syn_ds, seed, epochs):
    set_seed(seed)
    train_data = ConcatDataset([real_subset, syn_ds]) if syn_ds is not None else real_subset
    loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    model = GrayResNet(num_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for imgs, labels in tqdm(loader, desc=f"  cls seed{seed} {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward(); opt.step()
    # eval
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            trues.extend(labels.numpy())
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="weighted")
    return acc, f1, (trues, preds)


# ============================================================================
#  Experiment driver
# ============================================================================
def real_image_pool(subset, max_n):
    """Materialize a sample of real images (for FID/SSIM)."""
    loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)
    imgs, lbls = [], []
    for x, y in loader:
        imgs.append(x); lbls.append(torch.as_tensor(y))
        if sum(t.size(0) for t in imgs) >= max_n:
            break
    return torch.cat(imgs)[:max_n], torch.cat(lbls)[:max_n]


records = []
GEN_SEED = 12345   # VAE + both generators trained once per fraction with this seed; only the classifier varies across SEEDS

for frac in FRACTIONS:
    print("\n" + "=" * 70)
    print(f"DATA FRACTION = {frac}")
    print("=" * 70)
    set_seed(GEN_SEED)

    idx = stratified_indices(train_ds.targets, frac, GEN_SEED)
    real_subset = Subset(train_ds, idx)
    print(f"Real training images available: {len(real_subset)}")

    # 1) VAE on the available data only (no leakage from the full set)
    vae = train_vae(real_subset, VAE_EPOCHS)
    Ztr, ytr = extract_latents(vae, real_subset)
    print("Latent matrix:", tuple(Ztr.shape))

    # 2) Train both generators on the SAME latents
    Gc = train_wgan(Ztr, ytr, "classical", GAN_EPOCHS)
    Gq = train_wgan(Ztr, ytr, "quantum",   GAN_EPOCHS)

    # 3) Generate synthetic image sets
    syn_c_imgs, syn_c_lbls = generate_synthetic(Gc, vae, SYN_PER_CLASS)
    syn_q_imgs, syn_q_lbls = generate_synthetic(Gq, vae, SYN_PER_CLASS)
    save_grid(syn_c_imgs, os.path.join(OUT_DIR, f"samples_classical_f{frac}.png"), f"Classical GAN (f={frac})")
    save_grid(syn_q_imgs, os.path.join(OUT_DIR, f"samples_quantum_f{frac}.png"),  f"Quantum GAN (f={frac})")
    syn_c_ds = TensorImageDataset(syn_c_imgs, syn_c_lbls)
    syn_q_ds = TensorImageDataset(syn_q_imgs, syn_q_lbls)

    # 4) Generation metrics, per fraction
    fid_c = fid_q = ssim_c = ssim_q = float("nan")
    divp_c = divp_q = divstd_c = divstd_q = float("nan")
    # diversity is cheap -> always compute (this is the mode-collapse evidence)
    divp_c, divstd_c = compute_diversity(syn_c_imgs, DIV_PAIRS)
    divp_q, divstd_q = compute_diversity(syn_q_imgs, DIV_PAIRS)
    print(f"  DIVERSITY classical: pairwiseSSIM={divp_c:.3f} pixelStd={divstd_c:.3f}  | "
          f"quantum: pairwiseSSIM={divp_q:.3f} pixelStd={divstd_q:.3f}")
    print("    (high pairwise SSIM or low pixel std = mode collapse)")

    if any(abs(frac - m) < 1e-9 for m in METRIC_FRACTIONS):
        rp_imgs, rp_lbls = real_image_pool(real_subset, N_FID_REAL)
        fid_c = compute_fid(rp_imgs, syn_c_imgs)
        fid_q = compute_fid(rp_imgs, syn_q_imgs)
        ssim_c = compute_ssim(syn_c_imgs, syn_c_lbls, rp_imgs, rp_lbls)
        ssim_q = compute_ssim(syn_q_imgs, syn_q_lbls, rp_imgs, rp_lbls)
        plot_tsne(vae, rp_imgs, rp_lbls, syn_c_imgs, syn_c_lbls,
                  os.path.join(OUT_DIR, f"tsne_classical_f{frac}.png"), frac)
        plot_tsne(vae, rp_imgs, rp_lbls, syn_q_imgs, syn_q_lbls,
                  os.path.join(OUT_DIR, f"tsne_quantum_f{frac}.png"), frac)
        print(f"  FID  classical={fid_c:.2f}  quantum={fid_q:.2f}")
        print(f"  SSIM classical={ssim_c:.3f}  quantum={ssim_q:.3f}")

    # 5) Downstream classifier over seeds: real-only vs +classical vs +quantum
    for seed in SEEDS:
        a_r, f_r, _ = train_eval_classifier(real_subset, None,     seed, CLS_EPOCHS)
        a_c, f_c, _ = train_eval_classifier(real_subset, syn_c_ds, seed, CLS_EPOCHS)
        a_q, f_q, _ = train_eval_classifier(real_subset, syn_q_ds, seed, CLS_EPOCHS)
        print(f"  seed {seed}: real={a_r:.4f}  +classical={a_c:.4f}  +quantum={a_q:.4f}")
        records.append(dict(fraction=frac, seed=seed,
                            acc_real=a_r, f1_real=f_r,
                            acc_classical=a_c, f1_classical=f_c,
                            acc_quantum=a_q, f1_quantum=f_q,
                            fid_classical=fid_c, fid_quantum=fid_q,
                            ssim_classical=ssim_c, ssim_quantum=ssim_q,
                            div_pairwise_ssim_classical=divp_c, div_pairwise_ssim_quantum=divp_q,
                            div_pixelstd_classical=divstd_c, div_pixelstd_quantum=divstd_q))


# ============================================================================
#  Aggregate, test, report
# ============================================================================
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "raw_results.csv"), index=False)

print("\n" + "=" * 70)
print("AGGREGATED RESULTS (mean +/- std over seeds)")
print("=" * 70)

summary_rows = []
for frac in FRACTIONS:
    sub = df[df.fraction == frac]
    row = {"fraction": frac}
    for k in ["acc_real", "acc_classical", "acc_quantum"]:
        row[k] = f"{sub[k].mean():.4f}+/-{sub[k].std(ddof=0):.4f}"
    # paired tests vs real-only and vs classical
    try:
        row["p_q_vs_real"] = f"{ttest_rel(sub.acc_quantum, sub.acc_real).pvalue:.3f}"
        row["p_q_vs_clf"]  = f"{ttest_rel(sub.acc_quantum, sub.acc_classical).pvalue:.3f}"
    except Exception:
        row["p_q_vs_real"] = row["p_q_vs_clf"] = "n/a"
    # generation metrics (constant across seeds within a fraction)
    row["FID_clf"] = f"{sub.fid_classical.iloc[0]:.1f}"
    row["FID_q"]   = f"{sub.fid_quantum.iloc[0]:.1f}"
    row["pixStd_clf"] = f"{sub.div_pixelstd_classical.iloc[0]:.3f}"
    row["pixStd_q"]   = f"{sub.div_pixelstd_quantum.iloc[0]:.3f}"
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
pd.set_option("display.width", 200)
print(summary.to_string(index=False))

print("\nGeneration quality per fraction (lower FID = better; higher pixel-std = more diverse):")
for frac in FRACTIONS:
    sub = df[df.fraction == frac].iloc[0]
    print(f"  f={frac}: FID  c={sub.fid_classical:.1f} q={sub.fid_quantum:.1f}  | "
          f"pixelStd c={sub.div_pixelstd_classical:.3f} q={sub.div_pixelstd_quantum:.3f}  | "
          f"pairwiseSSIM c={sub.div_pairwise_ssim_classical:.3f} q={sub.div_pairwise_ssim_quantum:.3f}")

print(f"\nAll artifacts saved in: {OUT_DIR}/")
print("  raw_results.csv, summary.csv, per-fraction sample grids, per-fraction t-SNE (classical & quantum)")
