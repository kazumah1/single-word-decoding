"""Multi-time scale CNN backbone based on the base implementation within simpleconv.py

Modifications from original
---------------------------
``MultiScaleConvSequence`` — a parallel bank of ``ConvSequence`` modules.
    Each branch is configured with a distinct ``dilation_growth`` rate so
    that the stack simultaneously learns features at complementary temporal
    scales before merging via a 1×1 projection.  This directly addresses
    the finding in Fig. 2A of d'Ascoli et al. (2025) that linear-decoder
    accuracy peaks at different post-onset latencies across datasets:

    * **Low growth** (e.g. ``dg=1``)  — compact receptive field for early
      evoked components (N1/P2, ~100-200 ms post-onset).
    * **Moderate growth** (e.g. ``dg=2``) — intermediate window bridging
      early perceptual and later cognitive components.
    * **High growth** (e.g. ``dg=4``) — wide receptive field for late
      semantic integration (N400, ~400 ms+).

``SimpleConv._build_encoder`` hook.
    The encoder construction step is factored out into an overridable method
    so that ``MultiScaleSimpleConv`` (and future subclasses) can substitute
    a different backbone without duplicating the rest of the init logic.

``MultiScaleSimpleConvConfig`` / ``MultiScaleSimpleConv``.
    Drop-in replacements for ``SimpleConvConfig`` / ``SimpleConv`` that wire
    in the multi-scale backbone.

``MultiScaleSimpleConvTimeAggConfig`` / ``MultiScaleSimpleConvTimeAgg``.
    Temporal-aggregation variant of the above, mirroring the existing
    ``SimpleConvTimeAgg`` pattern.

TODO:
- Move unit tests as well
- Simplify PositionGetter to work without MNE object?
- Simplify ConvSequence
- Test out temporal aggregation version
"""

import logging
import typing as tp
from functools import partial

import torch
from torch import nn
from torchvision.ops import MLP

from .base import BaseModelConfig
from .common import BahdanauAttention, ChannelMerger, LayerScale, MlpConfig, SubjectLayers
from .transformer import TransformerEncoderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SpatialFilter  (unchanged)
# ---------------------------------------------------------------------------

class SpatialFilter(nn.Module):
    """Learns spatial combinations of MEG/EEG channels (beamformers).

    Uses a Conv2d with kernel (n_channels, 1) so each filter is a weighted
    sum over all input channels at each time step independently — identical
    to a learned spatial beamformer. This is the core idea from EEGNet and
    EEG Conformer's spatial convolution step.

    Input:  [batch, n_channels, time]
    Output: [batch, n_filters, time]
    """

    def __init__(self, n_channels: int, n_filters: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)   # [B, 1, C, T]
        x = self.conv(x)     # [B, n_filters, 1, T]
        x = self.bn(x)
        return x.squeeze(2)  # [B, n_filters, T]


# ---------------------------------------------------------------------------
# ConvSequence  (unchanged)
# ---------------------------------------------------------------------------

class ConvSequence(nn.Module):
    """Stack of 1-D dilated convolutions at a single temporal resolution.

    Parameters
    ----------
    channels:
        Sequence of channel widths ``[C_in, C_1, …, C_out]``.  Each
        consecutive pair defines one convolutional layer.
    kernel:
        Convolution kernel size (must be odd when ``dilation_growth > 1``).
    dilation_growth:
        Multiplicative factor applied to the dilation after every layer.
        E.g. ``dilation_growth=2`` gives dilations 1, 2, 4, 8, …
    dilation_period:
        When set, the dilation is reset to 1 every ``dilation_period``
        layers, creating a periodic pattern.  Set to ``None`` to allow
        the dilation to grow monotonically throughout the full stack.
    stride:
        Convolution stride.  Use 1 (default) to preserve temporal length.
    dropout:
        Dropout probability applied after each activation.
    leakiness:
        Negative slope for ``LeakyReLU``; 0 gives standard ``ReLU``.
    groups:
        Number of groups for grouped convolutions.
    decode:
        Use ``ConvTranspose1d`` instead of ``Conv1d``.
    batch_norm:
        Insert ``BatchNorm1d`` before each activation.
    dropout_input:
        Additional dropout on the very first input of the sequence.
    skip:
        Add residual skip connections wherever ``chin == chout``.
    scale:
        Initial value for ``LayerScale`` placed on skip paths.
    rewrite:
        Append a 1×1 conv + activation after each main conv block.
    activation_on_last:
        Whether to apply a non-linearity after the final layer.
    post_skip:
        Append a depthwise conv on the skip path.
    glu:
        Apply a gated linear unit every ``glu`` layers (0 = disabled).
    glu_context:
        Context padding for the GLU gate convolution.
    glu_glu:
        Use ``GLU(dim=1)`` as the gate; otherwise use the main activation.
    activation:
        Callable returning an ``nn.Module`` activation.  Defaults to
        ``LeakyReLU(leakiness)``.
    """

    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel: int = 4,
        dilation_growth: int = 1,
        dilation_period: int | None = None,
        stride: int = 2,
        dropout: float = 0.0,
        leakiness: float = 0.0,
        groups: int = 1,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0,
        skip: bool = False,
        scale: float | None = None,
        rewrite: bool = False,
        activation_on_last: bool = True,
        post_skip: bool = False,
        glu: int = 0,
        glu_context: int = 0,
        glu_glu: bool = True,
        activation: tp.Any = None,
    ) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(
                Conv(
                    chin,
                    chout,
                    kernel,
                    stride,
                    pad,
                    dilation=dilation,
                    groups=groups if k > 0 else 1,
                )
            )
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context),
                        act,
                    )
                )
            else:
                self.glus.append(None)  # type: ignore

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x


# ---------------------------------------------------------------------------
# MultiScaleConvSequence  (NEW)
# ---------------------------------------------------------------------------

class MultiScaleConvSequence(nn.Module):
    """Parallel bank of ``ConvSequence`` modules at multiple temporal scales.

    Each branch is a full ``ConvSequence`` that shares the same channel
    schedule and convolutional hyper-parameters but is configured with a
    distinct ``dilation_growth`` rate.  This lets the network simultaneously
    build features over narrow (early ERP) and wide (late semantic) temporal
    windows:

    +----------+-----------------+--------------------------------------+
    | Growth   | Receptive field | Target EEG/MEG component            |
    +==========+=================+======================================+
    | 1        | narrow          | N1/P2 (~100-200 ms post-onset)       |
    +----------+-----------------+--------------------------------------+
    | 2        | intermediate    | Early / late overlap window          |
    +----------+-----------------+--------------------------------------+
    | 4        | wide            | N400 semantic integration (~400 ms+) |
    +----------+-----------------+--------------------------------------+

    Branch outputs are concatenated channel-wise, then projected back to
    ``out_channels`` by a 1×1 convolution followed by batch normalisation,
    so the output shape is identical to that of a plain ``ConvSequence``.

    Parameters
    ----------
    channels:
        Per-branch channel schedule ``[C_in, C_1, …, C_branch_out]``.
        Shared across all branches.
    dilation_growths:
        One ``dilation_growth`` value per parallel branch.  Must contain
        at least two entries.
    out_channels:
        Channel dimensionality of the merged output.  The concatenated
        tensor (``C_branch_out × n_branches`` channels) is projected to
        this size by the merge convolution.
    **conv_kwargs:
        Forwarded verbatim to every ``ConvSequence`` branch.  The keys
        ``dilation_growth`` and ``dilation_period`` are removed/overridden
        internally: ``dilation_growth`` is set per-branch from
        ``dilation_growths``; ``dilation_period`` is forced to ``None`` so
        that dilations grow monotonically through the full stack of each
        branch, maximising receptive-field diversity.

    Notes
    -----
    *Parameter cost*: each branch has the same parameter count as a single
    ``ConvSequence`` with the shared ``channels`` schedule.  The merge
    projection adds ``C_branch_out × n_branches × out_channels`` weights.

    *Temporal alignment*: when ``stride > 1``, branches may produce
    marginally different output lengths due to integer rounding.  The
    forward pass clips all branches to the shortest output before
    concatenating.
    """

    def __init__(
        self,
        channels: tp.Sequence[int],
        dilation_growths: tp.Sequence[int],
        out_channels: int,
        **conv_kwargs,
    ) -> None:
        super().__init__()

        if len(dilation_growths) < 2:
            raise ValueError(
                "MultiScaleConvSequence requires at least 2 entries in "
                "dilation_growths.  For a single scale use ConvSequence directly."
            )

        # Remove the singular dilation_growth if accidentally forwarded, and
        # force dilation_period=None so each branch's dilations grow without
        # periodic resets — this maximises per-branch receptive-field reach.
        conv_kwargs.pop("dilation_growth", None)
        conv_kwargs["dilation_period"] = None

        self.n_scales: int = len(dilation_growths)
        self.dilation_growths: tp.Tuple[int, ...] = tuple(dilation_growths)

        self.branches = nn.ModuleList([
            ConvSequence(channels, dilation_growth=dg, **conv_kwargs)
            for dg in dilation_growths
        ])

        # Merge projection: (C_branch_out × n_scales) → out_channels
        branch_out_channels: int = int(channels[-1])
        merged_channels: int = branch_out_channels * self.n_scales
        self.merge_proj = nn.Sequential(
            nn.Conv1d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        logger.info(
            "MultiScaleConvSequence: %d branches with dilation_growths=%s, "
            "branch_out=%d, merged=%d → out=%d",
            self.n_scales,
            list(dilation_growths),
            branch_out_channels,
            merged_channels,
            out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through all branches then merge.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, C_in, T)``.

        Returns
        -------
        torch.Tensor
            Merged output of shape ``(batch, out_channels, T')``.
            ``T' = min(T_branch_i)`` across branches (equal to ``T`` when
            ``stride=1``).
        """
        branch_outs = [branch(x) for branch in self.branches]

        # Clip to the shortest temporal dimension (defensive guard; branches
        # produce identical lengths when stride=1, which is the normal case).
        min_t = min(out.shape[-1] for out in branch_outs)
        branch_outs = [out[..., :min_t] for out in branch_outs]

        merged = torch.cat(branch_outs, dim=1)   # (B, C_branch × n_scales, T')
        return self.merge_proj(merged)             # (B, out_channels, T')


# ---------------------------------------------------------------------------
# SimpleConvConfig  (unchanged)
# ---------------------------------------------------------------------------

class SimpleConvConfig(BaseModelConfig):
    name: tp.Literal["SimpleConv"] = "SimpleConv"

    # Channels
    hidden: int = 16
    # Overall structure
    depth: int = 4
    linear_out: bool = False
    complex_out: bool = False
    # Conv layer
    kernel_size: int = 5
    growth: float = 1.0
    dilation_growth: int = 2
    dilation_period: int | None = None
    skip: bool = False
    post_skip: bool = False
    scale: float | None = None
    rewrite: bool = False
    groups: int = 1
    glu: int = 0
    glu_context: int = 0
    glu_glu: bool = True
    gelu: bool = False
    # Dropouts, BN, activations
    conv_dropout: float = 0.0
    dropout_input: float = 0.0
    batch_norm: bool = False
    relu_leakiness: float = 0.0
    # Optional transformer
    transformer_config: TransformerEncoderConfig | None = None
    # Subject specific settings
    n_subjects: int = 200
    subject_layers: bool = False
    subject_layers_dim: str = "input"  # or hidden
    subject_layers_id: bool = False
    # Attention multi-dataset support
    merger: bool = False
    merger_pos_dim: int = 2048
    merger_channels: int = 270
    merger_dropout: float = 0.2
    merger_penalty: float = 0.0
    merger_per_subject: bool = False
    # Spatial filter (beamformer) applied before temporal convolutions
    spatial_filters: int = 0  # 0 = disabled; e.g. 32 to learn 32 channel combinations
    # Architectural details
    dropout: float = 0.0
    dropout_rescale: bool = True
    initial_linear: int = 0
    initial_depth: int = 1
    initial_nonlin: bool = False
    backbone_out_channels: int | None = None  # If provided, the output of the
    # backbone (i.e. layer before the output heads) will have this dimensionality

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return SimpleConv(n_in_channels, n_outputs, config=self)


# ---------------------------------------------------------------------------
# MultiScaleSimpleConvConfig  (NEW)
# ---------------------------------------------------------------------------

class MultiScaleSimpleConvConfig(SimpleConvConfig):
    """Configuration for ``MultiScaleSimpleConv``.

    Extends ``SimpleConvConfig`` with ``dilation_growths``, a list of per-
    branch dilation growth rates that replaces the single ``dilation_growth``
    scalar.  All other fields are inherited unchanged.

    The single inherited ``dilation_growth`` field is **ignored** by
    ``MultiScaleSimpleConv``; use ``dilation_growths`` instead.

    Parameters
    ----------
    dilation_growths:
        Per-branch dilation growth rates.  Must have at least 2 entries.
        Default ``[1, 2, 4]`` creates three branches covering early (N1/P2),
        intermediate, and late (N400) M/EEG components.

    Examples
    --------
    Two-branch lightweight variant::

        cfg = MultiScaleSimpleConvConfig(dilation_growths=[1, 4])

    Four-branch wide-coverage variant::

        cfg = MultiScaleSimpleConvConfig(dilation_growths=[1, 2, 4, 8])
    """

    name: tp.Literal["MultiScaleSimpleConv"] = "MultiScaleSimpleConv"  # type: ignore[assignment]

    dilation_growths: list[int] = [1, 2, 4]

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return MultiScaleSimpleConv(n_in_channels, n_outputs, config=self)


# ---------------------------------------------------------------------------
# SimpleConv  (refactored: encoder construction extracted to _build_encoder)
# ---------------------------------------------------------------------------

class SimpleConv(nn.Module):
    """Single-scale convolutional backbone for M/EEG word decoding.

    Architecture (applied in order)
    --------------------------------
    1. Optional ``ChannelMerger`` — position-aware spatial attention over
       sensor channels.
    2. Optional ``SpatialFilter`` — learned spatial beamforming.
    3. Optional initial pointwise projection.
    4. Optional ``SubjectLayers`` — per-subject linear projection to handle
       inter-subject variability.
    5. **Encoder** — built by :meth:`_build_encoder`.  For this class that
       is a plain ``ConvSequence``; subclasses override ``_build_encoder``
       to substitute a different backbone without duplicating the rest of the
       init.
    6. Optional ``final`` projection (``linear_out`` / ``complex_out``).
    7. Optional ``TransformerEncoder`` for sentence-level context.

    Parameters
    ----------
    in_channels:
        Number of raw input sensor channels.
    out_channels:
        Dimensionality of the output word-embedding space.
    config:
        ``SimpleConvConfig`` (or a subclass) instance.  Defaults to
        ``SimpleConvConfig()`` with factory defaults.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: SimpleConvConfig | None = None,
    ):
        super().__init__()
        config = config if config is not None else SimpleConvConfig()

        self.out_channels = out_channels
        self.backbone_out_channels = (
            out_channels
            if config.backbone_out_channels is None
            else config.backbone_out_channels
        )

        activation: nn.Module | tp.Callable
        if config.gelu:
            activation = nn.GELU
        elif config.relu_leakiness:
            activation = partial(nn.LeakyReLU, config.relu_leakiness)
        else:
            activation = nn.ReLU

        assert config.kernel_size % 2 == 1, "For padding to work, this must be verified"

        self.merger = None
        self.dropout = None

        self.initial_linear = None
        if config.dropout > 0.0:
            raise NotImplementedError("To be reimplemented here.")
            # self.dropout = ChannelDropout(dropout, dropout_rescale)
        if config.merger:
            self.merger = ChannelMerger(
                config.merger_channels,
                pos_dim=config.merger_pos_dim,
                dropout=config.merger_dropout,
                usage_penalty=config.merger_penalty,
                n_subjects=config.n_subjects,
                per_subject=config.merger_per_subject,
            )
            in_channels = config.merger_channels

        self.spatial_filter = None
        if config.spatial_filters > 0:
            self.spatial_filter = SpatialFilter(in_channels, config.spatial_filters)
            in_channels = config.spatial_filters

        if config.initial_linear:
            init: list[nn.Module | tp.Callable] = [
                nn.Conv1d(in_channels, config.initial_linear, 1)
            ]
            for _ in range(config.initial_depth - 1):
                init += [
                    activation(),
                    nn.Conv1d(config.initial_linear, config.initial_linear, 1),
                ]
            if config.initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)  # type: ignore[arg-type]
            in_channels = config.initial_linear

        self.subject_layers = None
        if config.subject_layers:
            dim = {"hidden": config.hidden, "input": in_channels}[
                config.subject_layers_dim
            ]
            self.subject_layers = SubjectLayers(
                in_channels, dim, config.n_subjects, config.subject_layers_id
            )
            in_channels = dim

        # Compute the channel schedule shared by all encoder variants.
        sizes = [in_channels]
        sizes += [
            int(round(config.hidden * config.growth**k)) for k in range(config.depth)
        ]

        # Assemble the keyword-argument dict forwarded to the encoder.
        params: tp.Dict[str, tp.Any] = dict(
            kernel=config.kernel_size,
            stride=1,
            leakiness=config.relu_leakiness,
            dropout=config.conv_dropout,
            dropout_input=config.dropout_input,
            batch_norm=config.batch_norm,
            dilation_growth=config.dilation_growth,
            groups=config.groups,
            dilation_period=config.dilation_period,
            skip=config.skip,
            post_skip=config.post_skip,
            scale=config.scale,
            rewrite=config.rewrite,
            glu=config.glu,
            glu_context=config.glu_context,
            glu_glu=config.glu_glu,
            activation=activation,
        )

        final_channels = sizes[-1]

        self.final: nn.Module | nn.Sequential | None = None
        pad = 0
        kernel = 1
        stride = 1

        if config.linear_out:
            assert not config.complex_out
            self.final = nn.ConvTranspose1d(
                final_channels, self.backbone_out_channels, kernel, stride, pad
            )
        elif config.complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                activation(),
                nn.ConvTranspose1d(
                    2 * final_channels, self.backbone_out_channels, kernel, stride, pad
                ),
            )
        else:
            # Without a separate final projection, the encoder itself outputs
            # backbone_out_channels and should not apply an activation on its
            # last layer (the downstream transformer / aggregation provides
            # the non-linearity).
            params["activation_on_last"] = False
            sizes[-1] = self.backbone_out_channels

        # Delegate to the encoder-construction hook so subclasses can swap in
        # a different backbone (e.g. MultiScaleConvSequence) without
        # duplicating the pre-processing logic above.
        self.encoder = self._build_encoder(sizes, params)

        self.transformer = None
        if config.transformer_config:
            self.transformer = config.transformer_config.build(
                dim=self.backbone_out_channels
            )

    # ------------------------------------------------------------------
    # Encoder construction hook
    # ------------------------------------------------------------------

    def _build_encoder(
        self,
        sizes: tp.List[int],
        params: tp.Dict[str, tp.Any],
    ) -> nn.Module:
        """Construct and return the temporal encoder module.

        Override in subclasses to replace ``ConvSequence`` with an
        alternative backbone without touching the rest of ``__init__``.

        Parameters
        ----------
        sizes:
            Channel schedule ``[C_in, C_1, …, C_out]`` after all pre-
            processing layers.  The last element equals
            ``self.backbone_out_channels`` when neither ``linear_out`` nor
            ``complex_out`` is active.
        params:
            Keyword arguments assembled from the active config, ready to
            be unpacked into an encoder constructor.

        Returns
        -------
        nn.Module
            Module stored as ``self.encoder``.
        """
        return ConvSequence(sizes, **params)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the full backbone.

        Parameters
        ----------
        x:
            M/EEG input of shape ``(batch, n_channels, n_times)``.
        subject_ids:
            Integer subject indices of shape ``(batch,)``.  Required when
            ``merger`` or ``subject_layers`` are enabled.
        channel_positions:
            2-D sensor positions of shape ``(batch, n_channels, 2)``.
            Required when ``merger`` is enabled.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, backbone_out_channels, n_times)``.
        """
        length = x.shape[-1]

        # if self.dropout is not None:
        #     x = self.dropout(x, batch)

        if self.merger is not None:
            x = self.merger(x, subject_ids, channel_positions)

        if self.spatial_filter is not None:
            x = self.spatial_filter(x)

        if self.initial_linear is not None:
            x = self.initial_linear(x)

        if self.subject_layers is not None:
            x = self.subject_layers(x, subject_ids)

        x = self.encoder(x)
        if self.final is not None:
            x = self.final(x)
        assert x.shape[-1] >= length
        x = x[:, :, :length]

        if self.transformer:
            x = self.transformer(x.transpose(1, 2)).transpose(1, 2)

        return x


# ---------------------------------------------------------------------------
# MultiScaleSimpleConv  (NEW)
# ---------------------------------------------------------------------------

class MultiScaleSimpleConv(SimpleConv):
    """``SimpleConv`` with a multi-scale parallel convolutional backbone.

    Replaces the single ``ConvSequence`` encoder with a
    ``MultiScaleConvSequence`` that runs one ``ConvSequence`` branch per
    entry in ``config.dilation_growths``.  Every other module — merger,
    spatial filter, subject layers, transformer — is inherited unchanged.

    Motivation
    ----------
    The linear-decoder analysis in Fig. 2A of d'Ascoli et al. (2025) shows
    that decoding accuracy peaks at different post-onset latencies for
    different datasets: some within ~200 ms (N1/P2, perceptual) and others
    extending to 500 ms+ (N400, semantic integration).  A fixed-resolution
    ``ConvSequence`` with a periodic dilation reset (``dilation_period=5``)
    cannot optimally cover both early and late components.  The parallel
    multi-scale bank dedicates independent capacity to each timescale and
    merges them before the temporal aggregation step.

    Implementation note
    --------------------
    ``_multiscale_dilation_growths`` is set on the instance *before*
    ``super().__init__()`` is called.  This is valid Python because
    ``__new__`` already created the object by the time ``__init__`` runs.
    It allows ``_build_encoder`` — which is invoked inside
    ``SimpleConv.__init__`` — to read the correct dilation schedule.

    Parameters
    ----------
    in_channels:
        Number of raw input sensor channels.
    out_channels:
        Output embedding dimensionality.
    config:
        ``MultiScaleSimpleConvConfig`` instance.  Defaults to the class
        defaults (three branches, ``dilation_growths=[1, 2, 4]``).

    See Also
    --------
    MultiScaleConvSequence : backbone used by this class.
    MultiScaleSimpleConvConfig : companion configuration dataclass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: MultiScaleSimpleConvConfig | None = None,
    ) -> None:
        config = config if config is not None else MultiScaleSimpleConvConfig()

        # Store dilation_growths on the instance BEFORE calling super().__init__().
        # SimpleConv.__init__ calls self._build_encoder(), which is already
        # overridden on this class and needs to know the dilation schedule.
        # Because __new__ has already created the object at this point, setting
        # an attribute here is safe even though super().__init__() has not run.
        self._multiscale_dilation_growths: tp.List[int] = list(config.dilation_growths)

        super().__init__(in_channels=in_channels, out_channels=out_channels, config=config)

    def _build_encoder(
        self,
        sizes: tp.List[int],
        params: tp.Dict[str, tp.Any],
    ) -> nn.Module:
        """Build a ``MultiScaleConvSequence`` in place of ``ConvSequence``.

        Called from ``SimpleConv.__init__`` after the pre-processing layers
        and the channel schedule have been computed.

        Parameters
        ----------
        sizes:
            Channel schedule after all pre-processing, passed from
            ``SimpleConv.__init__``.  The last element is
            ``self.backbone_out_channels`` (set before this call).
        params:
            Conv hyper-parameters assembled from the config.  ``dilation_growth``
            (singular) and ``dilation_period`` are stripped internally by
            ``MultiScaleConvSequence`` and replaced with per-branch values.

        Returns
        -------
        MultiScaleConvSequence
        """
        # Strip the single dilation_growth — replaced by per-branch values.
        branch_params = {k: v for k, v in params.items() if k != "dilation_growth"}

        return MultiScaleConvSequence(
            channels=sizes,
            dilation_growths=self._multiscale_dilation_growths,
            out_channels=self.backbone_out_channels,
            **branch_params,
        )


# ---------------------------------------------------------------------------
# SimpleConvTimeAgg  (unchanged)
# ---------------------------------------------------------------------------

class SimpleConvTimeAggConfig(SimpleConvConfig):
    name: tp.Literal["SimpleConvTimeAgg"] = "SimpleConvTimeAgg"  # type: ignore

    # SimpleConv-specific parameters override
    merger: bool = False
    subject_layers: bool = False
    # Temporal aggregation
    time_agg_out: tp.Literal["gap", "linear", "att"] = "gap"

    # Output head(s)
    output_head_config: MlpConfig | dict[str, MlpConfig] | None = None

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return SimpleConvTimeAgg(n_in_channels, n_outputs, config=self)


class SimpleConvTimeAgg(SimpleConv):
    """SimpleConv with temporal aggregation layer and potentially two output heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: SimpleConvTimeAggConfig | None = None,
    ):
        config = config if config is not None else SimpleConvTimeAggConfig()
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, config=config
        )

        # Output aggregation layer
        self.time_agg_out: nn.Module | None
        if config.time_agg_out == "gap":
            self.time_agg_out = nn.AdaptiveAvgPool1d(1)
        elif config.time_agg_out == "linear":
            self.time_agg_out = nn.LazyLinear(1)
        elif config.time_agg_out == "att":
            self.time_agg_out = BahdanauAttention(input_size=None, hidden_size=256)
        elif config.time_agg_out == "eegnet":
            self.time_agg_out = EEGNet(
                n_in_channels=self.backbone_out_channels,
                n_outputs=self.backbone_out_channels,
            )
        else:
            self.time_agg_out = None

        # Separate output head(s)
        self.output_head: None | MLP | dict[str, MLP]
        if config.output_head_config is None:
            self.output_head = None
        else:
            if self.time_agg_out is None:
                raise NotImplementedError("Output heads require temporal aggregation.")
            if isinstance(config.output_head_config, MlpConfig):
                self.output_head = config.output_head_config.build(
                    input_size=self.backbone_out_channels
                )
            elif isinstance(config.output_head_config, dict):
                self.output_head = nn.ModuleDict()
                for name, head_config in config.output_head_config.items():
                    self.output_head[name] = head_config.build(
                        input_size=self.backbone_out_channels
                    )

    def forward(  # type: ignore
        self, x, subject_ids=None, channel_positions=None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x = super().forward(
            x, subject_ids=subject_ids, channel_positions=channel_positions
        )

        if self.time_agg_out is not None:
            x = self.time_agg_out(x)
            if x.ndim == 3:
                x = x.squeeze(2)  # Remove singleton dimension

        # Apply output heads (e.g. for separate CLIP and MSE losses)
        if isinstance(self.output_head, MLP):
            x = self.output_head(x)
        elif isinstance(self.output_head, nn.ModuleDict):
            x = {name: head(x) for name, head in self.output_head.items()}

        return x


# ---------------------------------------------------------------------------
# MultiScaleSimpleConvTimeAgg  (NEW)
# ---------------------------------------------------------------------------

class MultiScaleSimpleConvTimeAggConfig(MultiScaleSimpleConvConfig):
    """Configuration for ``MultiScaleSimpleConvTimeAgg``.

    Combines the multi-scale backbone with a temporal aggregation step that
    collapses the time axis to a fixed-size vector, ready for the CLIP /
    D-SigLIP contrastive objectives described in the paper.

    Parameters
    ----------
    dilation_growths:
        Per-branch dilation growth rates (inherited from
        ``MultiScaleSimpleConvConfig``).  Default ``[1, 2, 4]``.
    time_agg_out:
        Temporal aggregation strategy:

        ``"gap"``
            Global average pooling — parameter-free, fast.
        ``"att"``
            Bahdanau attention over the time axis — learns a soft importance
            mask; recommended when the relevant component can occur at
            variable latencies across datasets.
        ``"linear"``
            Single linear projection of the time axis to a scalar — requires
            a fixed input length at initialisation time.

    output_head_config:
        Optional MLP projection applied after aggregation.  Pass a single
        ``MlpConfig`` for a shared head, or ``dict[str, MlpConfig]`` for
        named per-task heads (e.g. separate CLIP and regression heads).
    """

    name: tp.Literal["MultiScaleSimpleConvTimeAgg"] = "MultiScaleSimpleConvTimeAgg"  # type: ignore[assignment]

    # Mirrors SimpleConvTimeAggConfig overrides
    merger: bool = False
    subject_layers: bool = False

    time_agg_out: tp.Literal["gap", "linear", "att"] = "att"
    output_head_config: MlpConfig | dict[str, MlpConfig] | None = None

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return MultiScaleSimpleConvTimeAgg(n_in_channels, n_outputs, config=self)


class MultiScaleSimpleConvTimeAgg(MultiScaleSimpleConv):
    """Multi-scale backbone with temporal aggregation and optional output head.

    Extends ``MultiScaleSimpleConv`` with a temporal pooling / attention step
    that collapses ``(batch, backbone_out, T')`` → ``(batch, backbone_out)``,
    followed by an optional MLP projection.  This is a drop-in replacement
    for ``SimpleConvTimeAgg`` when the multi-scale backbone is desired.

    Full pipeline
    -------------
    ::

        M/EEG input  (B, C, T)
            │
            ▼  [optional] ChannelMerger / SpatialFilter / SubjectLayers
            │
            ▼  MultiScaleConvSequence
            │     branch 1 (dg=1)  ─╮
            │     branch 2 (dg=2)  ─┼─ cat → 1×1 BN proj → (B, backbone_out, T')
            │     branch 3 (dg=4)  ─╯
            │
            ▼  [optional] TransformerEncoder  (sentence-level context)
            │
            ▼  TemporalAggregation  → (B, backbone_out)
            │
            ▼  [optional] OutputHead MLP

    Parameters
    ----------
    in_channels:
        Number of raw input sensor channels.
    out_channels:
        Output embedding dimensionality.
    config:
        ``MultiScaleSimpleConvTimeAggConfig`` instance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: MultiScaleSimpleConvTimeAggConfig | None = None,
    ) -> None:
        config = config if config is not None else MultiScaleSimpleConvTimeAggConfig()
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, config=config
        )

        # ------------------------------------------------------------------
        # Temporal aggregation
        # ------------------------------------------------------------------
        self.time_agg_out: nn.Module | None
        if config.time_agg_out == "gap":
            self.time_agg_out = nn.AdaptiveAvgPool1d(1)
        elif config.time_agg_out == "linear":
            self.time_agg_out = nn.LazyLinear(1)
        elif config.time_agg_out == "att":
            self.time_agg_out = BahdanauAttention(input_size=None, hidden_size=256)
        else:
            self.time_agg_out = None

        # ------------------------------------------------------------------
        # Output projection head(s)
        # ------------------------------------------------------------------
        self.output_head: None | MLP | nn.ModuleDict
        if config.output_head_config is None:
            self.output_head = None
        else:
            if self.time_agg_out is None:
                raise NotImplementedError("Output heads require temporal aggregation.")
            if isinstance(config.output_head_config, MlpConfig):
                self.output_head = config.output_head_config.build(
                    input_size=self.backbone_out_channels
                )
            elif isinstance(config.output_head_config, dict):
                self.output_head = nn.ModuleDict(
                    {
                        name: head_cfg.build(input_size=self.backbone_out_channels)
                        for name, head_cfg in config.output_head_config.items()
                    }
                )
            else:
                raise TypeError(
                    f"Unsupported output_head_config type: {type(config.output_head_config)}"
                )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass through the full multi-scale pipeline.

        Parameters
        ----------
        x:
            M/EEG input of shape ``(batch, n_channels, n_times)``.
        subject_ids:
            Integer subject indices of shape ``(batch,)``.
        channel_positions:
            Sensor positions of shape ``(batch, n_channels, 2)``.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            * No output head: tensor ``(batch, backbone_out_channels)``
              after aggregation (or ``(batch, backbone_out_channels, T')``
              when ``time_agg_out`` is ``None``).
            * Single MLP head: tensor ``(batch, head_out_dim)``.
            * Multiple heads: ``dict[str, Tensor]`` keyed by head name.
        """
        # Shared backbone: merger → spatial filter → subject layers →
        # multi-scale encoder → optional transformer
        x = super().forward(
            x, subject_ids=subject_ids, channel_positions=channel_positions
        )  # (B, backbone_out_channels, T')

        # Temporal aggregation → (B, backbone_out_channels)
        if self.time_agg_out is not None:
            x = self.time_agg_out(x)
            if x.ndim == 3:
                x = x.squeeze(2)

        # Optional projection head(s)
        if isinstance(self.output_head, MLP):
            x = self.output_head(x)
        elif isinstance(self.output_head, nn.ModuleDict):
            x = {name: head(x) for name, head in self.output_head.items()}

        return x
