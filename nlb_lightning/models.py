import pytorch_lightning as pl
import torch
from torch import nn


class SequentialAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float,
        weight_decay: float,
        dropout=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Instantiate bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        # Instantiate linear mapping to initial conditions
        self.ic_linear = nn.Linear(2 * hidden_size, hidden_size)
        # Instantiate autonomous GRU decoder
        self.decoder = nn.GRU(
            input_size=1,  # Not used
            hidden_size=hidden_size,
            batch_first=True,
        )
        # Instantiate linear readout
        self.readout = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )
        # Instantiate dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, observ, fwd_steps):

        batch_size, obs_steps, _ = observ.shape
        # Pass data through the model
        _, h_n = self.encoder(observ)
        # Combine output from fwd and bwd encoders
        h_n = torch.cat([*h_n], -1)
        # Compute initial condition
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        # Create an empty input tensor
        input_placeholder = torch.zeros((batch_size, obs_steps + fwd_steps, 1))
        input_placeholder = input_placeholder.to(self.device)
        # Unroll the decoder
        ic_drop = self.dropout(ic)
        latents, _ = self.decoder(input_placeholder, torch.unsqueeze(ic_drop, 0))
        # Map decoder state to logrates
        logrates = self.readout(latents)

        return logrates, latents

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_ix):

        heldin, heldin_forward, heldout, heldout_forward, behavior = batch
        # Pass data through the model
        fwd_steps = heldin_forward.shape[1]
        preds, latents = self.forward(heldin, fwd_steps)
        # Assemble the data
        heldin_full = torch.cat([heldin, heldin_forward], dim=1)
        heldout_full = torch.cat([heldout, heldout_forward], dim=1)
        data = torch.cat([heldin_full, heldout_full], dim=2)
        # Compute the Poisson log-likelihood
        loss = nn.functional.poisson_nll_loss(preds, data)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_ix):

        heldin, heldin_forward, heldout, heldout_forward, behavior = batch
        # Pass data through the model
        fwd_steps = heldin_forward.shape[1]
        preds, latents = self.forward(heldin, fwd_steps)
        # Assemble the data
        heldin_full = torch.cat([heldin, heldin_forward], dim=1)
        heldout_full = torch.cat([heldout, heldout_forward], dim=1)
        data = torch.cat([heldin_full, heldout_full], dim=2)
        # Compute the Poisson log-likelihood
        loss = nn.functional.poisson_nll_loss(preds, data)
        self.log("valid/loss", loss)
        self.log("hp_metric", loss)

        return loss
