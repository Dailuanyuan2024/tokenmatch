"""Non-destructive visual variant of the uploaded Tokenmatch backbone.

Requires the ORIGINAL project's loss.py, Transformer.py, sTransformer.py,
and tokenmatch.py on sys.path. No backbone replacement.
"""
import torch
import torch.nn as nn
import tokenmatch as core
try:
    from .visual_inputs import VISUAL_DIM
except ImportError:
    from visual_inputs import VISUAL_DIM


def portable_batch_symeig(x):
    """Keep the original CPU eigensolve but return to the input device."""
    return torch.linalg.eigh(x.cpu(), UPLO="U").eigenvectors.to(x.device)


class Tokenmatch(core.Tokenmatch):
    """Add a shared-per-point 262->128->128 MLP before initial token creation.

    data['xs']:     B x 1 x N x 4, unchanged geometry for weighted 8-points.
    data['visual']: B x N x 262, encoded and aligned by visual_inputs.py.
    Optional data['sides'] keeps the original backbone's ratio/mutual contract.
    Only the initial block is augmented; refinement blocks are unchanged.
    """
    def __init__(self, config):
        if config.net_channels != 128:
            raise ValueError("The uploaded backbone hard-codes 128 channels.")
        if getattr(config, "cluster", 64) != 64:
            raise ValueError("The uploaded backbone hard-codes 64 cluster tokens.")
        super().__init__(config)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(VISUAL_DIM, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1),
        )
        # Original .cuda() in batch_symeig breaks CPU tests/nondefault devices.
        # The eigensolve, ordering, and CPU precision are otherwise unchanged.
        core.batch_symeig = portable_batch_symeig

    def forward(self, data):
        xs = data["xs"]
        if xs.ndim != 4 or xs.shape[1] != 1 or xs.shape[-1] != 4:
            raise ValueError("xs must be B x 1 x N x 4; never append descriptors to xs.")
        if "visual" not in data:
            raise KeyError("Missing visual inputs; coordinate-only caches cannot train this model.")
        visual = data["visual"]
        b, _, n, _ = xs.shape
        if visual.shape != (b, n, VISUAL_DIM):
            raise ValueError("visual must be B x N x 262 in the same row order as xs.")
        if visual.device != xs.device or visual.dtype != xs.dtype:
            raise ValueError("xs and visual must share a device and floating-point dtype.")
        if not torch.isfinite(visual).all():
            raise ValueError("Non-finite visual inputs.")

        inputs = xs.transpose(1, 3)
        if self.side_channel > 0:
            sides = data["sides"].transpose(1, 2).unsqueeze(3)
            inputs = torch.cat([inputs, sides], dim=1)

        block = self.weights_init
        geometric = block.conv(inputs)
        appearance = self.visual_encoder(visual.transpose(1, 2).unsqueeze(3))
        fused = geometric + appearance
        tokens = block.ge_cluster_token(fused)
        x0 = block.l0(tokens)
        down = block.cluster_down(x0)
        cluster_logits = block.linear_1(down).view(b, -1)
        filtered = block.filter(down, cluster_logits)
        up = block.cluster_up(x0, filtered)
        out = block.embed_1(block.l1(up))
        logits = block.linear_1(out).squeeze(3).squeeze(1)
        e_hat = core.weighted_8points(xs, logits)
        residual = core.batch_episym(xs[:, 0, :, :2], xs[:, 0, :, 2:4], e_hat)
        residual = residual.reshape(b, 1, n, 1)

        all_logits, all_e = [logits], [e_hat]
        for block in self.weights_iter:
            block_inputs = torch.cat([
                inputs, residual.detach(),
                torch.relu(torch.tanh(logits)).reshape(residual.shape).detach(),
            ], dim=1)
            logits, e_hat, residual = block(block_inputs, xs)
            all_logits.append(logits)
            all_e.append(e_hat)
        return all_logits, all_e
