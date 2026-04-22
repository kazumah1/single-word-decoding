# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import os
from collections import defaultdict

import json
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

from neuralset.infra.utils import environment_variables
from neuraltrain.metrics import Rank

from .decoder import Decoder
from .utils import agg_per_group, agg_retrieval_preds


class InitialEvaluation(Callback):
    """
    Run an initial evaluation before training starts to get chance level baseline.
    """

    def __init__(self):
        pass

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.validate_loop.run()
        for metric in pl_module.metrics.values():
            metric.reset()
        return


class TestRetrieval(Callback):
    """Accumulate predictions on entire test set before evaluating a metric."""

    def __init__(
        self,
        event_type="Word",
        event_field="text",
        retrieval_set_sizes=[None, 250],
        decoder: Decoder | None = None,
        config_name: str | None = None,
    ):
        self.event_type = event_type
        self.event_field = event_field
        self.retrieval_set_sizes = retrieval_set_sizes
        self.full_outputs = {}
        self.decoder = decoder
        self.config_name = config_name

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        if not hasattr(pl_module, "retrieval_metrics") and not isinstance(
            pl_module.retrieval_metrics, nn.ModuleDict
        ):
            raise ValueError(
                "The LightningModule needs a retrieval_metrics ModuleDict that contains the "
                "metrics to evaluate on the full test set."
            )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self.full_outputs = {
            idx: defaultdict(list) for idx in range(len(trainer.val_dataloaders))
        }

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.full_outputs = {
            idx: defaultdict(list) for idx in range(len(trainer.test_dataloaders))
        }

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._collate_outputs(outputs, batch, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._collate_outputs(outputs, batch, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._compute_metrics(trainer, pl_module, step_name="val")
        self._save_outputs(trainer, step_name="val")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._compute_metrics(trainer, pl_module, step_name="test")
        self._save_outputs(trainer, step_name="test")

    def _save_outputs(self, trainer, step_name):
        n_loaders = (
            len(trainer.val_dataloaders)
            if step_name == "val"
            else len(trainer.test_dataloaders)
        )
        for dataloader_idx in range(n_loaders):
            full = self.full_outputs[dataloader_idx]
            for key in ["y_pred", "y_true"]:
                full[key] = torch.cat(full[key], dim=0)

            save_dir = os.path.join(trainer.logger.save_dir, "retrieval_outputs")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(full, os.path.join(save_dir, f"{step_name}_{dataloader_idx}.pt"))


    def _label_prefix(self, prefix: str) -> str:
        return f"[{self.config_name}] {prefix}" if self.config_name else prefix

    def _slug_prefix(self, prefix: str) -> str:
        if not self.config_name:
            return prefix
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_"
                    for c in self.config_name)
        return f"{safe}__{prefix}"


    def _save_eval_results(self, trainer, step_name, dataloader_idx, retrieval_out, sentence_out, sentence_accs,):
        if not (trainer.logger and getattr(trainer.logger, "save_dir", None)):
            return
        flat: dict[str, float] = {}
        for src in (retrieval_out, sentence_out):
            for k, v in (src or {}).items():
                try:
                    flat[k] = float(v)
                except Exception:
                    pass
        if sentence_accs:
            flat["sentence_accuracy_mean"] = float(np.mean(sentence_accs))
        out_path = os.path.join(trainer.logger.save_dir, "eval_results.json")
        existing: dict = {}
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        key = f"{step_name}_{dataloader_idx}"
        block = existing.get(key, {})
        block.update(flat)
        if self.config_name:
            block["config_name"] = self.config_name
        existing[key] = block
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)

    def _collate_outputs(self, outputs, batch, dataloader_idx):
        y_pred, y_true = outputs
        full = self.full_outputs[dataloader_idx]

        full["y_pred"].append(y_pred.cpu())
        full["y_true"].append(y_true.cpu())

        for segment in batch.segments:
            trigger = segment._trigger
            full[self.event_field].append(trigger[self.event_field])
            full["subject_id"].append(trigger["subject"])
            full["sequence_id"].append(trigger["sequence_id"])
            full["timeline"].append(trigger["timeline"])

    def _compute_metrics(self, trainer, pl_module, step_name):
        n_loaders = (
            len(trainer.val_dataloaders)
            if step_name == "val"
            else len(trainer.test_dataloaders)
        )
        retrieval_metrics = {
            k: v
            for k, v in pl_module.retrieval_metrics.items()
            if (k.startswith(step_name) and "retrieval" in k)
        }
        sentence_metrics = {
            k: v
            for k, v in pl_module.retrieval_metrics.items()
            if (k.startswith(step_name) and "sentence" in k)
        }
        for dataloader_idx in range(n_loaders):
            full = self.full_outputs[dataloader_idx]
            groups_pred = full[self.event_field]
            subjects_pred = full["subject_id"]
            sentence_pred = full["sequence_id"]
            timeline_pred = full["timeline"]
            sentence_uids = [
                f"{sequence}_{timeline}"
                for sequence, timeline in zip(sentence_pred, timeline_pred)
            ]
            y_pred = torch.cat(full["y_pred"], dim=0)
            y_true = torch.cat(full["y_true"], dim=0)
            # Save full-dataset refs before sentence-metric subsetting (used for figures)
            y_pred_full, y_true_full = y_pred, y_true
            groups_pred_full, subjects_pred_full = groups_pred, subjects_pred

            all_retrieval_out = {}
            for retrieval_set_size in self.retrieval_set_sizes:
                out = self._get_retrieval_metrics(
                    y_pred,
                    y_true,
                    groups_pred,
                    subjects_pred,
                    retrieval_metrics,
                    retrieval_set_size=retrieval_set_size,
                )
                all_retrieval_out.update(out)
                for key, value in out.items():
                    key += f"_{dataloader_idx}"
                    pl_module.log(key, value)

            if step_name == "val":
                # keep only one subject for sentence metrics (quite slow)
                idx = np.where(np.array(subjects_pred) == subjects_pred[0])[0]
                y_pred, y_true = y_pred[idx], y_true[idx]
                groups_pred = [groups_pred[i] for i in idx]
                sentence_uids = [sentence_uids[i] for i in idx]
            out_metrics, true_sentences, pred_sentences, corr_sentences, sentence_accs = (
                self._get_sentence_metrics(
                    y_pred,
                    y_true,
                    groups_pred,
                    sentence_uids,
                    sentence_metrics,
                )
            )
            for key, value in out_metrics.items():
                key += f"_{dataloader_idx}"
                pl_module.log(key, value)
            self._save_eval_results(
                trainer, step_name, dataloader_idx,
                retrieval_out=all_retrieval_out,
                sentence_out=out_metrics,
                sentence_accs=sentence_accs,
                )
            save_dir = os.path.join(
                trainer.logger.save_dir,
                f"decoded_sentences",
            )
            os.makedirs(save_dir, exist_ok=True)

            with open(
                os.path.join(save_dir, f"{step_name}_{dataloader_idx}.txt"), "w"
            ) as f:
                for true, pred, corr in zip(
                    true_sentences, pred_sentences, corr_sentences
                ):
                    f.write(f"True: {true}\n")
                    f.write(f"Pred: {pred}\n")
                    f.write(f"Corr: {corr}\n")
                    f.write("\n")
            f.close()
            try:
                self._save_figures(
                    trainer, step_name, dataloader_idx,
                    y_pred_full, y_true_full, groups_pred_full, subjects_pred_full,
                    all_retrieval_out, sentence_accs,
                )
            except Exception:
                pass

    def _get_sentence_metrics(
        self,
        y_pred,
        y_true,
        true_words,
        sentence_uids,
        metrics,
    ):

        agg_y_true, agg_groups_true = agg_per_group(
            y_true, groups=true_words, agg_func="first"
        )
        scores = Rank._compute_sim(y_pred, agg_y_true)

        if self.decoder:
            self.decoder.id2word = {i: w for i, w in enumerate(agg_groups_true)}
        scores = torch.Tensor(scores)

        pred_sentences, true_sentences, accs, corr_sentences = [], [], [], []
        for sentence_uid in np.unique(sentence_uids):
            idx = np.where(np.array(sentence_uids) == sentence_uid)[0]
            sentence_scores = scores[idx]

            true_sentence = " ".join([true_words[i] for i in idx])
            pred_sentence = " ".join(
                [agg_groups_true[i] for i in sentence_scores.argmax(dim=1)]
            )
            if self.decoder is not None:
                corr_sentence = (
                    self.decoder.decode(sentence_scores)
                    if self.decoder
                    else pred_sentence
                )
            else:
                corr_sentence = pred_sentence

            pred_sentences.append(pred_sentence)
            true_sentences.append(true_sentence)
            corr_sentences.append(corr_sentence)

            acc = np.mean(
                [
                    w1 == w2
                    for w1, w2 in zip(pred_sentence.split(" "), true_sentence.split(" "))
                ]
            )
            accs.append(acc)
        # sort by accuracy
        idx = np.argsort(accs)[::-1]
        pred_sentences = [pred_sentences[i] for i in idx]
        true_sentences = [true_sentences[i] for i in idx]
        corr_sentences = [corr_sentences[i] for i in idx]

        out = {}
        for correct in [False, True]:
            for metric_name, metric in metrics.items():
                metric_name += f"_correct={correct}"
                preds = corr_sentences if correct else pred_sentences
                with environment_variables(TOKENIZERS_PARALLELISM="false"):
                    res = metric(preds, true_sentences)
                if "bert" in metric_name:
                    res = torch.mean(res["f1"])
                out[metric_name] = res

        return out, true_sentences, pred_sentences, corr_sentences, accs

    @classmethod
    def _get_retrieval_metrics(
        cls, y_pred, y_true, groups_pred, subjects_pred, metrics, retrieval_set_size=None
    ):
        out = {}

        # Keep only the most frequent groups
        if retrieval_set_size is not None:
            groups_df = pd.DataFrame({"label": groups_pred})
            counts = groups_df.label.value_counts()
            most_frequent = set(counts.index[:retrieval_set_size])
            indices = groups_df.label.isin(most_frequent).values
            indices = np.where(indices)[0]  # Get indices where condition is True
            indices = torch.from_numpy(indices)
            y_pred, y_true = y_pred[indices], y_true[indices]
            groups_pred = [groups_pred[i] for i in indices]
            subjects_pred = [subjects_pred[i] for i in indices]

        # Remove repetitions in retrieval set
        agg_y_true, agg_groups_true = agg_per_group(
            y_true, groups=groups_pred, agg_func="first"
        )

        for metric_name, metric in metrics.items():
            metric = metric.to("cpu")
            if metric_name.endswith("subject-agg"):
                subjects = subjects_pred
            elif metric_name.endswith("instance-agg"):
                subjects = None
            else:
                subjects = torch.arange(y_pred.shape[0], device=y_pred.device)

            agg_y_pred, agg_groups_pred = agg_retrieval_preds(
                y_pred,
                groups_pred=groups_pred,
                subjects_pred=subjects,
            )
            if retrieval_set_size is not None:
                metric_name += f"_size={retrieval_set_size}"
            else:
                metric_name += f"_size=all"

            metric.reset()
            metric.update(agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true)

            out[metric_name] = metric.compute()

            if "agg" not in metric_name:
                # compute frequency corrected average
                ranks = metric._compute_ranks(
                    agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true
                )
                macro_average = np.mean(
                    list(metric._compute_macro_average(ranks, agg_groups_pred).values())
                )
                out[metric_name + "_macro"] = macro_average
        return out

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def _save_figures(
        self,
        trainer,
        step_name,
        dataloader_idx,
        y_pred,
        y_true,
        groups_pred,
        subjects_pred,
        all_retrieval_out,
        sentence_accs,
    ):
        """Generate and save diagnostic figures alongside the existing outputs."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not hasattr(trainer.logger, "save_dir") or trainer.logger.save_dir is None:
            return

        fig_dir = os.path.join(trainer.logger.save_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        title_prefix = self._label_prefix(f"{step_name}_{dataloader_idx}")
        file_prefix  = self._slug_prefix(f"{step_name}_{dataloader_idx}")

        # Precompute per-query ranks once; shared by several figures.
        ranks_per_query, groups_per_query, n_candidates = (None, None, None)
        try:
            ranks_per_query, groups_per_query, n_candidates = (
                self._compute_per_query_ranks(y_pred, y_true, groups_pred)
            )
        except Exception:
            pass

        try:
            self._fig_retrieval_bar_chart(fig_dir, title_prefix, file_prefix, all_retrieval_out, plt)
        except Exception:
            pass
        try:
            self._fig_similarity_heatmap(fig_dir, title_prefix, file_prefix, y_pred, y_true, groups_pred, plt)
        except Exception:
            pass
        try:
            self._fig_similarity_heatmap_row_normalized(
                fig_dir, title_prefix, file_prefix, y_pred, y_true, groups_pred, plt
            )
        except Exception:
            pass
        try:
            self._fig_per_word_rank(
                fig_dir, title_prefix, file_prefix, ranks_per_query, groups_per_query, plt,
                y_pred=y_pred, y_true=y_true, groups_pred=groups_pred,
            )
        except Exception:
            pass
        
        if ranks_per_query:
            try:
                self._fig_rank_distribution(
                    fig_dir, title_prefix, file_prefix, ranks_per_query, n_candidates, plt
                )
            except Exception:
                pass
            try:
                self._fig_topk_accuracy_curve(
                    fig_dir, title_prefix, file_prefix, ranks_per_query, n_candidates, plt
                )
            except Exception:
                pass

        if sentence_accs:
            try:
                self._fig_sentence_accuracy(fig_dir, title_prefix, file_prefix, sentence_accs, plt)
            except Exception:
                pass

    @staticmethod
    def _fig_retrieval_bar_chart(fig_dir, title_prefix, file_prefix, all_retrieval_out, plt):
        """
        Bar chart of all retrieval metrics.
        
        Rank-style metrics (median rank, values on the order of the vocabulary size)
        and accuracy-style metrics (fractions in [0, 1]) are plotted on separate
        subplots so neither visually dominates the other. The previous single-axis
        version squashed accuracy bars to near-zero height next to the rank bars.
        """
        scalar_metrics = {}
        for k, v in all_retrieval_out.items():
            try:
                scalar_metrics[k] = float(v)
            except (TypeError, ValueError):
                pass
        if not scalar_metrics:
            return

        def _shorten(k):
            # e.g. "val_retrieval_acc10_instance-agg_size=250_macro" -> "acc10_instance-agg_size=250_macro"
            return k.split("_retrieval_", 1)[-1] if "_retrieval_" in k else k

        rank_items, acc_items, other_items = [], [], []
        for k, v in scalar_metrics.items():
            if "rank" in k.lower():
                rank_items.append((k, v))
            elif "acc" in k.lower():
                acc_items.append((k, v))
            else:
                other_items.append((k, v))
        groups = [
            ("Median Rank (lower is better)", rank_items, "steelblue"),
            ("Top-k Accuracy (higher is better)", acc_items, "seagreen"),
        ]
        if other_items:
            groups.append(("Other", other_items, "slategray"))
        groups = [g for g in groups if g[1]]
        if not groups:
            return
        widths = [max(3, len(items) * 0.6) for _, items, _ in groups]
        fig, axes = plt.subplots(
            1, len(groups),
            figsize=(sum(widths) + 2, 5),
            gridspec_kw={"width_ratios": widths},
        )
        if len(groups) == 1:
            axes = [axes]
        for ax, (title, items, color) in zip(axes, groups):
            labels = [_shorten(k) for k, _ in items]
            values = [v for _, v in items]
            bars = ax.bar(range(len(items)), values, color=color, alpha=0.85)
            ax.set_xticks(range(len(items)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Value")
            ax.set_title(title, fontsize=10)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
            ax.grid(True, axis="y", alpha=0.3)
            if title.startswith("Top-k"):
                ax.set_ylim(0, max(1.0, max(values) * 1.15))
        fig.suptitle(f"{title_prefix} — Retrieval Metrics", fontsize=11)

        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"{file_prefix}_retrieval_metrics.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)

    @staticmethod
    def _fig_similarity_heatmap(fig_dir, title_prefix, file_prefix, y_pred, y_true, groups_pred, plt):
        """Cosine-similarity heatmap between sampled predictions and unique word embeddings."""
        agg_y_true, agg_groups_true = agg_per_group(y_true, groups=groups_pred, agg_func="first")
        word_to_agg_idx = {w: i for i, w in enumerate(agg_groups_true)}

        # First-occurrence prediction index per unique word
        group_to_pred_idx: dict = {}
        for i, g in enumerate(groups_pred):
            if g not in group_to_pred_idx:
                group_to_pred_idx[g] = i

        # Subsample up to 50 words, reproducibly
        n_words = min(50, len(agg_groups_true))
        rng = np.random.default_rng(seed=0)
        chosen = list(rng.choice(len(agg_groups_true), n_words, replace=False))
        selected_words = [
            agg_groups_true[i] for i in chosen if agg_groups_true[i] in group_to_pred_idx
        ]

        pred_indices = [group_to_pred_idx[w] for w in selected_words]
        agg_indices = [word_to_agg_idx[w] for w in selected_words]

        sample_y_pred = y_pred[pred_indices]
        sample_y_true = agg_y_true[agg_indices]
        scores = Rank._compute_sim(sample_y_pred, sample_y_true).numpy()

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(scores, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(selected_words)))
        ax.set_yticks(range(len(selected_words)))
        ax.set_xticklabels(selected_words, rotation=90, fontsize=6)
        ax.set_yticklabels(selected_words, fontsize=6)
        ax.set_xlabel("Candidate word (true embedding)")
        ax.set_ylabel("Query word (predicted embedding)")
        ax.set_title(f"{title_prefix} — Cosine Similarity Matrix (n={len(selected_words)} words)")
        plt.colorbar(im, ax=ax, label="Cosine similarity")
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"{file_prefix}_similarity_matrix.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)

    @staticmethod
    def _compute_per_query_ranks(y_pred, y_true, groups_pred):
        """Return (ranks, groups, n_candidates) for every query in the retrieval set.
        ranks are 1-indexed (1 == top-1 correct retrieval). Aggregates y_true to a
        vocabulary of unique words first, so a rank of 1 means the query's correct
        word embedding had the highest cosine similarity among all unique candidates.
        """
        agg_y_true, agg_groups_true = agg_per_group(
            y_true, groups=groups_pred, agg_func="first"
        )
        group_to_idx = {w: i for i, w in enumerate(agg_groups_true)}

        scores = Rank._compute_sim(y_pred, agg_y_true)
        sorted_cols = scores.argsort(dim=1, descending=True)

        ranks, groups = [], []
        for i, g in enumerate(groups_pred):
            if g not in group_to_idx:
                continue
            correct_idx = group_to_idx[g]
            rank = (sorted_cols[i] == correct_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            groups.append(g)
        return ranks, groups, len(agg_groups_true)
    
    @staticmethod
    def _fig_per_word_rank(fig_dir, title_prefix, file_prefix, ranks_per_query, groups_per_query, plt,y_pred=None, y_true=None, groups_pred=None,):
        """Horizontal bar charts of the best- and worst-decoded words by median rank."""
        if not ranks_per_query or not groups_per_query:
            # Fallback: recompute if caller didn't pass the precomputed ranks.
            if y_pred is None or y_true is None or groups_pred is None:
                return
            ranks_per_query, groups_per_query, _ = (
                TestRetrieval._compute_per_query_ranks(y_pred, y_true, groups_pred)
            )
            if not ranks_per_query:
                return
        word_ranks: dict = defaultdict(list)
        for g, r in zip(groups_per_query, ranks_per_query):
            word_ranks[g].append(r)
        word_median = {w: float(np.median(r)) for w, r in word_ranks.items()}
        sorted_words = sorted(word_median.items(), key=lambda x: x[1])
        plt.close(fig)

    @staticmethod
    def _fig_sentence_accuracy(fig_dir, title_prefix, file_prefix, sentence_accs, plt):
        """Histogram of per-sentence word-level accuracy."""
        accs = np.array(sentence_accs, dtype=float)
        mean_acc = float(np.mean(accs))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(accs, bins=min(20, len(accs)), color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(mean_acc, color="firebrick", linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_acc:.2f}")
        ax.set_xlabel("Word-level Accuracy per Sentence")
        ax.set_ylabel("Count")
        ax.set_title(f"{title_prefix} — Sentence Accuracy Distribution")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"{file_prefix}_sentence_accuracy.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)

    @staticmethod
    def _fig_rank_distribution(fig_dir, title_prefix, file_prefix, ranks_per_query, n_candidates, plt):
        """Histogram of per-query ranks over the full retrieval set.
        Complements the median-rank bars by showing the whole distribution, including
        the mass at rank 1 (correct top-1 retrievals) and the tail.
        """
        ranks = np.asarray(ranks_per_query, dtype=float)
        if ranks.size == 0 or not n_candidates:
            return
        median_rank = float(np.median(ranks))
        mean_rank = float(np.mean(ranks))
        chance = (n_candidates + 1) / 2.0
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bins = int(min(50, max(5, n_candidates)))
        ax.hist(ranks, bins=n_bins, color="steelblue", alpha=0.85, edgecolor="white")
        ax.axvline(median_rank, color="firebrick", linestyle="--", linewidth=1.5,
                   label=f"Median: {median_rank:.1f}")
        ax.axvline(mean_rank, color="darkorange", linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_rank:.1f}")
        ax.axvline(chance, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Chance: {chance:.1f}")
        ax.set_xlabel(
            f"Rank of correct word (1 = best, {n_candidates} = worst)"
        )
        ax.set_ylabel("Count")
        ax.set_title(
            f"{title_prefix} — Rank Distribution "
            f"(N={len(ranks)} queries, V={n_candidates} candidates)"
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"{file_prefix}_rank_distribution.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)
    @staticmethod
    def _fig_topk_accuracy_curve(fig_dir, title_prefix, file_prefix, ranks_per_query, n_candidates, plt):
        """Top-k accuracy curve across k, with chance baseline.
        acc@k = fraction of queries whose correct word is within the top-k nearest
        neighbors. A model that beats chance lifts this curve above the diagonal.
        """
        ranks = np.asarray(ranks_per_query, dtype=int)
        if ranks.size == 0 or not n_candidates:
            return
        ks = np.arange(1, n_candidates + 1)
        acc_at_k = np.array([(ranks <= k).mean() for k in ks])
        chance = ks / float(n_candidates)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, acc_at_k, color="steelblue", linewidth=2, label="Model")
        ax.plot(ks, chance, color="gray", linestyle="--", linewidth=1.2, label="Chance")
        ax.fill_between(ks, chance, acc_at_k,
                        where=(acc_at_k >= chance), color="steelblue", alpha=0.15)
        for k_mark in (1, 5, 10):
            if k_mark <= n_candidates:
                ax.axvline(k_mark, color="lightgray", linewidth=0.6)
                ax.annotate(
                    f"acc@{k_mark}={acc_at_k[k_mark - 1]:.2f}",
                    xy=(k_mark, acc_at_k[k_mark - 1]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7,
                )
        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy @ k")
        ax.set_title(
            f"{title_prefix} — Top-k Retrieval Accuracy (V={n_candidates})"
        )
        ax.set_xlim(1, n_candidates)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"{file_prefix}_topk_accuracy.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)
    @staticmethod
    def _fig_similarity_heatmap_row_normalized(
        fig_dir, title_prefix, file_prefix, y_pred, y_true, groups_pred, plt
    ):
        """Row-centered cosine-similarity heatmap.
        When the raw similarities live in a narrow band (e.g. 0.9990..0.9995 for
        a near-untrained model), the absolute heatmap is uninformative. Subtracting
        each row's mean reveals the *relative* preference of every query across
        candidates, which is what retrieval actually cares about. Correct-word
        diagonal cells should become the brightest cells in their row as training
        progresses.
        """
        agg_y_true, agg_groups_true = agg_per_group(
            y_true, groups=groups_pred, agg_func="first"
        )
        word_to_agg_idx = {w: i for i, w in enumerate(agg_groups_true)}
        group_to_pred_idx: dict = {}
        for i, g in enumerate(groups_pred):
            if g not in group_to_pred_idx:
                group_to_pred_idx[g] = i
        n_words = min(50, len(agg_groups_true))
        rng = np.random.default_rng(seed=0)
        chosen = list(rng.choice(len(agg_groups_true), n_words, replace=False))
        selected_words = [
            agg_groups_true[i] for i in chosen if agg_groups_true[i] in group_to_pred_idx
        ]
        if not selected_words:
            return
        pred_indices = [group_to_pred_idx[w] for w in selected_words]
        agg_indices = [word_to_agg_idx[w] for w in selected_words]
        sample_y_pred = y_pred[pred_indices]
        sample_y_true = agg_y_true[agg_indices]
        scores = Rank._compute_sim(sample_y_pred, sample_y_true).numpy()
        # Row-center so each query's preferences are visible regardless of its
        # absolute similarity offset.
        centered = scores - scores.mean(axis=1, keepdims=True)
        vmax = float(np.max(np.abs(centered))) or 1e-9
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(
            centered, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        ax.set_xticks(range(len(selected_words)))
        ax.set_yticks(range(len(selected_words)))
        ax.set_xticklabels(selected_words, rotation=90, fontsize=6)
        ax.set_yticklabels(selected_words, fontsize=6)
        ax.set_xlabel("Candidate word (true embedding)")
        ax.set_ylabel("Query word (predicted embedding)")
        ax.set_title(
            f"{title_prefix} — Row-Centered Cosine Similarity (n={len(selected_words)} words)"
        )
        plt.colorbar(im, ax=ax, label="Similarity − row mean")
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"{file_prefix}_similarity_matrix_row_norm.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)
class TrainingCurves(Callback):
    """Collect per-epoch scalar metrics and plot learning curves at the end of fit.
    Listens to `trainer.callback_metrics` at the end of every train and validation
    epoch and records any scalar (float-castable) values. On `on_fit_end`, it writes
    one PNG per logical metric name to `<logger.save_dir>/figures/`, overlaying
    train vs. val curves whenever both exist (e.g. `train_cnn_loss` vs
    `val_cnn_loss`). This fills the biggest visualization gap the original
    callbacks had: no way to see loss / accuracy evolving over epochs.
    """
    def __init__(self, include_patterns: list[str] | None = None, config_name: str | None = None):
        self.include_patterns = include_patterns
        self.config_name = config_name
        self._history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    def _keep(self, key: str) -> bool:
        if not self.include_patterns:
            return True
        return any(p in key for p in self.include_patterns)
    def _record(self, trainer):
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        for key, value in trainer.callback_metrics.items():
            if not self._keep(key):
                continue
            try:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                self._history[key].append((epoch, float(value)))
            except (TypeError, ValueError, RuntimeError):
                continue
    def on_train_epoch_end(self, trainer, pl_module):
        self._record(trainer)
    def on_validation_epoch_end(self, trainer, pl_module):
        self._record(trainer)
    def on_fit_end(self, trainer, pl_module):
        try:
            self._plot_curves(trainer)
        except Exception:
            pass
    @staticmethod
    def _logical_name(key: str) -> tuple[str, str | None]:
        """Split a key like 'val_cnn_loss' into (base='cnn_loss', split='val')."""
        for split in ("train", "val", "test"):
            prefix = f"{split}_"
            if key.startswith(prefix):
                return key[len(prefix):], split
        return key, None
    def _plot_curves(self, trainer):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if not hasattr(trainer.logger, "save_dir") or trainer.logger.save_dir is None:
            return
        if not self._history:
            return
        fig_dir = os.path.join(trainer.logger.save_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(dict)
        for key, series in self._history.items():
            base, split = self._logical_name(key)
            grouped[base][split or "value"] = series
        # Dump raw history to disk so downstream analysis isn't stuck with PNGs.
        flat = {k: v for k, v in self._history.items()}
        try:
            torch.save(flat, os.path.join(fig_dir, "training_history.pt"))
            # CSV history for multi-config aggregation (one row per (metric, epoch) point).
            csv_path = os.path.join(trainer.logger.save_dir, "training_history.csv")
            try:
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["config_name", "metric", "epoch", "value"])
                    for key, series in self._history.items():
                        for epoch, value in series:
                            w.writerow([self.config_name or "", key, epoch, value])
            except Exception:
                pass

        except Exception:
            pass
        split_colors = {"train": "steelblue", "val": "firebrick", "test": "seagreen",
                        "value": "slategray"}
        for base, splits in grouped.items():
            fig, ax = plt.subplots(figsize=(7, 4))
            plotted = False
            for split, series in splits.items():
                if not series:
                    continue
                # Average duplicate epoch entries (e.g. val runs multiple times).
                per_epoch: dict[int, list[float]] = defaultdict(list)
                for epoch, v in series:
                    per_epoch[epoch].append(v)
                epochs = sorted(per_epoch.keys())
                values = [float(np.mean(per_epoch[e])) for e in epochs]
                ax.plot(
                    epochs, values, marker="o", markersize=3, linewidth=1.5,
                    color=split_colors.get(split, None), label=split,
                )
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel("Epoch")
            ax.set_ylabel(base)
            ax.set_title(f"Training Curve — {base}"+ (f"  ({self.config_name})" if self.config_name else ""))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            plt.tight_layout()
            safe = base.replace("/", "_").replace(" ", "_")
            plt.savefig(
                os.path.join(fig_dir, f"curve_{safe}.png"),
                dpi=100, bbox_inches="tight",
            )
            plt.close(fig)
            
    #     # log table to wandb if possible
    #     if retrieval_set_size is not None:
    #         metric = Rank()
    #         agg_y_pred, agg_groups_pred = agg_retrieval_preds(
    #             _y_pred,
    #             groups_pred=_groups_pred,
    #             subjects_pred=torch.arange(_y_pred.shape[0], device=_y_pred.device),
    #         )
    #         columns = ["Word", "Rank", "Preds", "Ratios"]
    #         ranks = metric._compute_ranks(
    #             agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true
    #         )
    #         macro_ranks = metric._compute_macro_average(ranks, agg_groups_pred)
    #         pred_labels = metric._get_most_frequent_predictions(
    #             agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true, k=10
    #         )
    #         data = []
    #         for true_label in agg_groups_true:
    #             pred_labels_ = [x[0] for x in pred_labels[true_label]]
    #             counts_ = np.array([x[1] for x in pred_labels[true_label]], dtype=float)
    #             counts_ = [f"{x:.2f}" for x in counts_]
    #             data.append(
    #                 [
    #                     true_label,
    #                     macro_ranks[true_label],
    #                     ",".join(pred_labels_),
    #                     ",".join(counts_),
    #                 ]
    #             )
    #         data = sorted(data, key=lambda x: x[1])  # sort by ranks
    #         if hasattr(pl_module.logger, "log_table"):
    #             pl_module.logger.log_table(
    #                 key=step_name + "_retrieval_rank_" + pl_module.logger.experiment.name,
    #                 columns=columns,
    #                 data=data,
    #             )
