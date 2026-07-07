import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 2 * 2, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 2, 2)
        x = self.decoder(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_encoder_output_shape():
    model = ConvEncoder(latent_dim=64)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 64), f"Esperado (4, 64), obtenido {out.shape}"


def test_decoder_output_shape():
    model = ConvDecoder(latent_dim=64)
    x = torch.randn(4, 64)
    out = model(x)
    assert out.shape == (4, 1, 28, 28), f"Esperado (4, 1, 28, 28), obtenido {out.shape}"


def test_autoencoder_forward():
    model = ConvAutoencoder(latent_dim=64)
    x = torch.randn(4, 1, 28, 28)
    recon, latent = model(x)
    assert recon.shape == (4, 1, 28, 28), f"Recon shape: {recon.shape}"
    assert latent.shape == (4, 64), f"Latent shape: {latent.shape}"


def test_autoencoder_recon_loss():
    model = ConvAutoencoder(latent_dim=64)
    x = torch.randn(4, 1, 28, 28)
    recon, _ = model(x)
    loss = nn.MSELoss()(recon, x)
    assert loss.item() >= 0.0, "La loss MSE debe ser >= 0"


def test_count_params():
    model = ConvEncoder(latent_dim=64)
    n = count_params(model)
    assert n > 0, f"Parametros debe ser > 0, obtenido {n}"
    assert isinstance(n, int), f"Debe ser int, obtenido {type(n)}"


def test_count_params_autoencoder():
    model = ConvAutoencoder(latent_dim=64)
    n = count_params(model)
    assert n == 1136961, f"Esperado 1136961 parametros, obtenido {n}"


def test_encoder_backward():
    model = ConvEncoder(latent_dim=64)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Gradientes no calculados"
            break


def test_different_latent_dims():
    for dim in [16, 32, 64, 128]:
        model = ConvAutoencoder(latent_dim=dim)
        x = torch.randn(2, 1, 28, 28)
        recon, latent = model(x)
        assert latent.shape == (2, dim), f"Dim {dim}: shape {latent.shape}"
        assert recon.shape == (2, 1, 28, 28)
