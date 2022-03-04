import pytorch_lightning as pl
import torch
from torch import nn


class SequentialAutoencoder(pl.LightningModule):
    """A simple sequential autoencoder that demonstrates
    a model compatible with the `nlb_lightning` API. The
    recognition model is a bidirectional GRU that reads
    over the heldin data, followed by a linear layer that
    maps the last hidden states to an initial condition for
    a decoding GRU. The decoding GRU unrolls without inputs,
    and its states are linearly mapped to logrates. The
    model is trained using Poisson NLL of all observed,
    forward, heldin, and heldout data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        fwd_steps: int,
        learning_rate: float,
        weight_decay: float,
        dropout: float,
    ):
        """Initializes the model.

        Parameters
        ----------
        input_size : int
            The dimensionality of the input sequence (i.e.
            number of heldin neurons)
        hidden_size : int
            The hidden dimensionality of the network, which
            determines the dimensionality of both the encoders
            and decoders
        output_size : int
            The dimensionality of the output sequence (i.e.
            total number of heldin and heldout neurons)
        fwd_steps: int
            The number of time steps to unroll beyond T
        learning_rate : float
            The learning rate to use for optimization
        weight_decay : float
            The weight decay to regularize optimization
        dropout : float
            The ratio of neurons to drop in dropout layers
        """
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

    def forward(self, observ, use_logrates=False):
        """The forward pass of the model.

        Parameters
        ----------
        observ : torch.Tensor
            A BxTxN tensor of heldin neurons at observed
            time points.
        use_logrates: bool
            Whether to output logrates for training
            or firing rates for analysis.

        Returns
        -------
        torch.Tensor
            A Bx(T+fwd_steps)x(N+n_heldout) tensor of
            estimated firing rates
        torch.Tensor
            A Bx(T+fwd_steps)x(hidden_dim) tensor of
            latent states
        """
        batch_size, obs_steps, _ = observ.shape
        # Pass data through the model
        _, h_n = self.encoder(observ)
        # Combine output from fwd and bwd encoders
        h_n = torch.cat([*h_n], -1)
        # Compute initial condition
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        # Create an empty input tensor
        fwd_steps = self.hparams.fwd_steps
        input_placeholder = torch.zeros((batch_size, obs_steps + fwd_steps, 1))
        input_placeholder = input_placeholder.to(self.device)
        # Unroll the decoder
        ic_drop = self.dropout(ic)
        latents, _ = self.decoder(input_placeholder, torch.unsqueeze(ic_drop, 0))
        # Map decoder state to logrates
        logrates = self.readout(latents)
        if use_logrates:
            return logrates, latents
        else:
            return torch.exp(logrates), latents

    def configure_optimizers(self):
        """Sets up the optimizer.

        Returns
        -------
        torch.optim.Adam
            A configured optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_ix):
        """Computes, logs, and returns the loss.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch of data from the datamodule - contains
            heldin, heldin_forward, heldout, heldout_forward,
            and behavior tensors.
        batch_ix : int
            Ignored

        Returns
        -------
        torch.Tensor
            The scalar loss
        """

        input_data, recon_data, behavior = batch
        # Pass data through the model
        preds, latents = self.forward(input_data, use_logrates=True)
        # Compute the Poisson log-likelihood
        loss = nn.functional.poisson_nll_loss(preds, recon_data)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_ix):
        """Computes, logs, and returns the loss.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch of data from the datamodule. During the
            "val" phase, contains heldin, heldin_forward,
            heldout, heldout_forward, and behavior tensors.
            During the "test" phase, contains only the heldin
            tensor.
        batch_ix : int
            Ignored

        Returns
        -------
        torch.Tensor
            The scalar loss
        """

        # On test-phase data, compute loss only across heldin neurons
        if len(batch) == 1:
            (input_data,) = batch
            # Pass data through the model
            preds, latents = self.forward(input_data, use_logrates=True)
            # Isolate heldin predictions
            _, n_obs, n_heldin = input_data.shape
            preds = preds[:, :n_obs, :n_heldin]
            recon_data = input_data
        else:
            input_data, recon_data, behavior = batch
            # Pass data through the model
            preds, latents = self.forward(input_data, use_logrates=True)
        # Compute the Poisson log-likelihood
        loss = nn.functional.poisson_nll_loss(preds, recon_data)
        self.log("valid/loss", loss)
        self.log("hp_metric", loss)

        return loss
