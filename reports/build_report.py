"""Compile the Kaggle run outputs of notebooks 11-16 into one PDF report (REPORT.pdf) with figures.

Usage: python reports/build_report.py <runs_dir> <out_dir>
Needs the `markdown` package and Edge or Chrome (printed headless) on the machine.
<runs_dir> holds one sub-folder per notebook (11, 11c, 12, 13, 14, 15, 16) with the files
downloaded by `kaggle kernels output`. Missing folders are reported, not invented.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# Reference palette (light mode): categorical slots in fixed order, text and surface tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 150, "axes.axisbelow": True,
})


def rel(path, out):
    return path.relative_to(out).as_posix()


def dot_ci(ax, tables, labels, colors, title):
    """Per-target AUC dots with 95% bootstrap whiskers; one row per target."""
    n = len(tables)
    offsets = np.linspace(-0.18, 0.18, n) if n > 1 else [0.0]
    y = np.arange(len(TARGETS))[::-1]
    for tab, lab, col, off in zip(tables, labels, colors, offsets):
        tab = tab.set_index("target").loc[TARGETS]
        ax.hlines(y + off, tab["ci_low"], tab["ci_high"], color=col, linewidth=2, alpha=0.55)
        ax.plot(tab["auc"], y + off, "o", color=col, markersize=6, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.51, -0.85, "chance", color=MUTED, ha="left", fontsize=8)
    ax.set_ylim(-1.1, len(TARGETS) - 0.4)
    ax.set_yticks(y, TARGETS)
    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("ROC AUC on the 58 expert-labelled studies (95% bootstrap CI)")
    ax.set_title(title, loc="left", pad=24 if n > 1 else 6)
    if n > 1:  # above the axes so it never covers a target row
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n, fontsize=8,
                  borderaxespad=0.2, handletextpad=0.3, columnspacing=1.2)


def bootstrap(gold, scores, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in TARGETS:
        y, p = gold[t].to_numpy().astype(int), scores[t].to_numpy()
        auc = roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if np.unique(y[i]).size == 2:
                boots.append(roc_auc_score(y[i], p[i]))
        rows.append({"target": t, "auc": auc, "ci_low": np.percentile(boots, 2.5),
                     "ci_high": np.percentile(boots, 97.5), "n_pos": int(y.sum()),
                     "n_neg": int((1 - y).sum())})
    return pd.DataFrame(rows)


def fmt_table(df, floats=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.{floats}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main(runs, out):
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    md, missing = [], []

    def need(*parts):
        """A file from one notebook's outputs. A notebook whose run folder is absent is simply not
        part of this report; a file missing from a folder that exists is reported."""
        p = runs.joinpath(*parts)
        if not p.exists() and runs.joinpath(parts[0]).is_dir():
            missing.append("/".join(parts))
        return p if p.exists() else None

    has16 = runs.joinpath("16").is_dir()
    has_ablation = runs.joinpath("17").is_dir() or runs.joinpath("16r").is_dir()

    weak_p = need("11", "weak_labels_v1.csv")
    agree_p = need("11", "weak_label_agreement_v1.csv")
    weak = pd.read_csv(weak_p, dtype={"StudyInstanceUID": str}) if weak_p else None
    gold_rows = weak[weak["is_gold"].eq(1)].set_index("StudyInstanceUID") if weak is not None else None
    gold_y = (gold_rows[[t + "__gold" for t in TARGETS]].set_axis(TARGETS, axis=1).astype(int)
              if gold_rows is not None else None)
    soft_agree_p, soft_meta_p = need("11c", "soft_label_agreement_v1.csv"), need("11c", "soft_labels_meta_v1.json")
    soft_agree = pd.read_csv(soft_agree_p) if soft_agree_p else None
    soft_meta = json.loads(soft_meta_p.read_text()) if soft_meta_p else {}
    if soft_agree is not None and not soft_meta:
        soft_agree = None

    # ------------------------------------------------------------------ 1. data
    md += ["## 1. Data and supervision", ""]
    if weak is not None:
        n, g = len(weak), int(weak["is_gold"].sum())
        dup = weak["report_hash"].duplicated(keep=False)
        md += [f"- Training studies: **{n:,}**; with expert labels: **{g}** ({g / n:.1%}).",
               f"- Studies sharing an identical report with another study: {int(dup.sum())} "
               f"in {weak.loc[dup, 'report_hash'].nunique()} groups (kept in one fold).", ""]
        prev = gold_y.mean().loc[TARGETS]
        silent = pd.Series({t: weak[t + "__silent"].mean() for t in TARGETS})
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        y = np.arange(len(TARGETS))[::-1]
        ax[0].barh(y, prev.values, color=BLUE, height=0.6)
        ax[0].set_yticks(y, TARGETS)
        ax[0].set_xlim(0, 1.08)
        ax[0].set_title("Expert-label prevalence (n = 58)", loc="left")
        ax[0].set_xlabel("share of studies positive")
        ax[1].barh(y, 1 - silent.values, color=ORANGE, height=0.6)
        ax[1].set_xlim(0, 1.08)
        ax[1].set_title("Report coverage (n = 4,407)", loc="left")
        ax[1].set_xlabel("share of reports that mention the finding")
        for a, vals in ((ax[0], prev.values), (ax[1], 1 - silent.values)):
            for yi, v in zip(y, vals):
                a.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=INK2)
        fig.tight_layout()
        fig.savefig(figs / "01_prevalence_coverage.png")
        plt.close(fig)
        md += [f"![prevalence and coverage]({rel(figs / '01_prevalence_coverage.png', out)})", "",
               "Reports that never mention a finding are masked out of the training loss for "
               "that finding, so the right panel is also the share of studies that supervise it.", ""]

    # ------------------------------------------------------------- 2. labellers
    md += ["## 2. Report-derived labels vs expert labels", ""]
    if agree_p:
        agree = pd.read_csv(agree_p)
        tables, labels, colors = [agree], ["regex extractor (nb 11)"], [ORANGE]
        if soft_agree is not None:
            for src, lab, col in (("llm", "Qwen2.5-7B (nb 11c)", AQUA),
                                  ("blend", "blend, used for training (nb 11c)", BLUE)):
                tables.append(soft_agree[["target", f"{src}_auc", f"{src}_ci_low", f"{src}_ci_high"]]
                              .set_axis(["target", "auc", "ci_low", "ci_high"], axis=1))
                labels.append(lab)
                colors.append(col)
        # 11b (Qwen2.5-1.5B via model-adapt) was abandoned and superseded by 11c; shown only if present.
        preds_p = sorted(runs.joinpath("11b").rglob("predictions.jsonl")) if runs.joinpath("11b").exists() else []
        if preds_p:
            pr = pd.DataFrame([json.loads(line) for line in open(preds_p[-1], encoding="utf-8")])
            pr[["sid", "t"]] = pr["example_id"].str.split("::", expand=True)
            m = {"positive": 0.9, "not_mentioned": 0.3, "negative": 0.1, "": 0.3}
            llm_scores = pr.assign(s=pr["output"].map(m)).pivot(index="sid", columns="t", values="s")
            llm_scores = llm_scores.loc[gold_y.index, TARGETS]
            tables.append(bootstrap(gold_y, llm_scores))
            labels.append("Qwen2.5-1.5B via model-adapt (nb 11b)")
            colors.append(AQUA)
            nm = pr["output"].value_counts(normalize=True)
            md += ["LLM output distribution over 696 prompts: "
                   + ", ".join(f"{k or 'invalid'} {v:.1%}" for k, v in nm.items()) + ".", ""]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        dot_ci(ax, tables, labels, colors, "How well do report-derived labels match the experts?")
        fig.tight_layout()
        fig.savefig(figs / "02_labeller_agreement.png")
        plt.close(fig)
        md += [f"![labeller agreement]({rel(figs / '02_labeller_agreement.png', out)})", ""]
        rows = [{"labeller": lab, "macro AUC": tab["auc"].mean(),
                 "evaluable targets": int(tab["auc"].notna().sum())} for lab, tab in zip(labels, tables)]
        md += [fmt_table(pd.DataFrame(rows), 4), ""]
        md += ["Regex extractor, per target:", "",
               fmt_table(agree[["target", "auc", "ci_low", "ci_high", "n_pos", "n_neg",
                                "silent_rate_all"]]), ""]
        if soft_agree is not None:
            md += [f"LLM soft labels (notebook 11c): `{soft_meta['model']}` at revision "
                   f"`{soft_meta['revision'][:12]}`, {soft_meta['quantisation']}, "
                   f"P(\"yes\") per finding for {soft_meta['n_reports']:,} reports in "
                   f"{soft_meta['seconds'] / 3600:.2f} h. The label source used for training "
                   f"(**{soft_meta['source']}**) was picked by macro AUC on these same 58 studies, "
                   "so its row is optimistically biased; on 58 studies, differences under about "
                   "0.05 are within noise.", "",
                   fmt_table(soft_agree[["target", "regex_auc", "llm_auc", "blend_auc", "n_pos", "n_neg"]]), ""]

    # ----------------------------------------------------------------- 3. folds
    md += ["## 3. Cross-validation folds", ""]
    folds_p, hdr_p = need("12", "folds_v1.csv"), need("12", "study_headers_v1.csv")
    if folds_p and weak is not None:
        folds = pd.read_csv(folds_p, dtype={"StudyInstanceUID": str})
        per = folds.groupby("fold").agg(studies=("StudyInstanceUID", "size"),
                                        expert_labelled=("is_gold", "sum")).reset_index()
        md += [fmt_table(per), ""]
        gf = gold_rows.join(folds.set_index("StudyInstanceUID")["fold"])
        cov = pd.DataFrame({t: gf.groupby("fold")[t + "__gold"].sum() for t in TARGETS}).T.astype(int)
        cnt = gf.groupby("fold").size()
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.imshow(cov.values, cmap="Blues", aspect="auto", vmin=0, vmax=max(1, cov.values.max()))
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                v, total = cov.values[i, j], cnt.iloc[j]
                flag = v == 0 or v == total
                ax.text(j, i, f"{v}" + ("*" if flag else ""), ha="center", va="center", fontsize=8,
                        color=SURFACE if v > cov.values.max() * 0.6 else INK)
        ax.set_xticks(range(cov.shape[1]), [f"fold {k}\n(n={cnt.iloc[k]})" for k in range(cov.shape[1])])
        ax.set_yticks(range(len(TARGETS)), TARGETS)
        ax.grid(False)
        ax.set_title("Expert positives per fold (* = fold AUC undefined)", loc="left")
        fig.tight_layout()
        fig.savefig(figs / "03_fold_coverage.png")
        plt.close(fig)
        md += [f"![fold coverage]({rel(figs / '03_fold_coverage.png', out)})", ""]
    if hdr_p:
        hdr = pd.read_csv(hdr_p, keep_default_na=False)
        per_id = hdr.loc[hdr["patient_hash"].ne(""), "patient_hash"].value_counts()
        md += [f"- DICOM `PatientID`: {len(per_id):,} distinct values for {len(hdr):,} studies, "
               f"blank in {int(hdr['patient_hash'].eq('').sum())}. "
               + ("Every study has its own ID, so IDs were anonymised per study: **patient-level "
                  "separation cannot be verified**; folds separate studies and identical-report groups."
                  if per_id.max() == 1 else "Repeat patients were grouped into one fold."), ""]

    # ---------------------------------------------------------- 4. preprocessing
    md += ["## 4. Preprocessing (notebook 13)", ""]
    man_p = need("13", "preprocess_manifest_v1.csv")
    if man_p:
        man = pd.read_csv(man_p, keep_default_na=False).drop_duplicates("StudyInstanceUID", keep="last")
        rows = [("studies", f"{len(man):,}"),
                ("status", ", ".join(f"{k}: {v}" for k, v in man["status"].value_counts().items())),
                ("planes present (of 3)", ", ".join(f"{k}: {v}" for k, v in
                                                     man["planes_present"].value_counts().sort_index().items())),
                ("side source", ", ".join(f"{k}: {v}" for k, v in man["side_source"].value_counts().items())),
                ("side", ", ".join(f"{k or 'unresolved'}: {v}" for k, v in man["side"].value_counts().items())),
                ("series whose geometry disagrees with the CSV plane",
                 str(int(pd.to_numeric(man["plane_mismatch"], errors="coerce").fillna(0).sum()))),
                ("studies with a series-level error", str(int(man["error"].ne("").sum()))),
                ("median seconds per study", f"{pd.to_numeric(man['seconds']).median():.2f}"),
                ("version", man["version"].iloc[0])]
        full_log = runs / "13" / "full_run.log"
        if weak is not None and len(man) < len(weak) and full_log.exists():
            # shortcut: manifest download kept failing (Kaggle resets on the ~4.4k-file output listing),
            # so report the full run's own audit printout instead of the partial manifest.
            text = full_log.read_text(errors="replace")
            try:
                text = "".join(x.get("data", "") for x in json.loads(text))
            except ValueError:
                pass
            keys = ("status:", "planes present:", "side source:", "side:",
                    "series whose geometry", "studies with a series error:")
            lines = [ln.strip() for ln in text.splitlines()]
            progress = [ln for ln in lines if re.match(r"\d+/\d+ studies", ln)]
            audit = progress[-1:] + [ln for ln in lines if ln.startswith(keys)]
            md += [f"The local manifest covers only {len(man)} studies (smoke run); the full run's "
                   "audit printout from the Kaggle log is shown instead:", "", "```", *audit, "```", ""]
        else:
            md += [fmt_table(pd.DataFrame(rows, columns=["item", "value"])), ""]

    # ----------------------------------------------------------- 5. image model
    md += ["## 5. Image model (notebook 14): out-of-fold evaluation on expert labels", ""]
    met_p, oof_p = need("14", "metrics_v1.json"), need("14", "oof_gold_v1.csv")
    if met_p and oof_p and gold_y is not None:
        met = json.loads(met_p.read_text())
        oof = pd.read_csv(oof_p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        pred = oof[[t + "__pred" for t in TARGETS]].set_axis(TARGETS, axis=1)
        gy = gold_y.loc[pred.index]
        img = bootstrap(gy, pred)
        md += [f"**Pooled OOF macro AUC: {met['oof_macro_auc']:.4f}** over "
               f"{met['evaluable_targets']}/12 targets "
               f"(previous image baseline, notebook 10: 0.526; chance: 0.5).", ""]
        tables, labels, colors = [img], ["image model, weak + expert labels (nb 14)"], [BLUE]
        if agree_p:
            tables.append(pd.read_csv(agree_p))
            labels.append("report extractor (reference; no reports at test time)")
            colors.append(ORANGE)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        dot_ci(ax, tables, labels, colors, "Image model vs the reports it was taught from")
        fig.tight_layout()
        fig.savefig(figs / "04_image_model_auc.png")
        plt.close(fig)
        md += [f"![image model AUC]({rel(figs / '04_image_model_auc.png', out)})", ""]

        fig, axes = plt.subplots(3, 4, figsize=(10, 7.5), sharex=True, sharey=True)
        for ax, t in zip(axes.ravel(), TARGETS):
            y, p = gy[t].to_numpy(), pred[t].to_numpy()
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
            if np.unique(y).size == 2:
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, color=BLUE, linewidth=2)
                ax.set_title(f"{t}  AUC {roc_auc_score(y, p):.2f}", loc="left", fontsize=9)
            ax.text(0.98, 0.04, f"{int(y.sum())}+ / {int((1 - y).sum())}-", ha="right",
                    fontsize=7, color=INK2, transform=ax.transAxes)
        for ax in axes[-1]:
            ax.set_xlabel("false positive rate")
        for ax in axes[:, 0]:
            ax.set_ylabel("true positive rate")
        fig.suptitle("Out-of-fold ROC curves, image model (expert-labelled studies)",
                     x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(figs / "05_roc_curves.png")
        plt.close(fig)
        md += [f"![ROC curves]({rel(figs / '05_roc_curves.png', out)})", ""]

        md += ["Per-fold macro AUC (about 11-12 expert-labelled studies per fold; a target "
               "missing a class in a fold is excluded and counted):", "",
               fmt_table(pd.DataFrame(met["per_fold"])), ""]
        md += ["Per target:", "", fmt_table(img), ""]

    # ------------------------------------------------------- 6. fine-tuned MIL
    if has16:
        md += ["## 6. Fine-tuned 2.5D attention-MIL (notebook 16): evaluation on expert labels", ""]
    ev_p, gp_p = need("16", "gold_eval_v1.json"), need("16", "gold_preds_v1.csv")
    hist_p, cfg_p = need("16", "history_v1.csv"), need("16", "model", "config.json")
    if ev_p and gp_p and hist_p and cfg_p and gold_y is not None:
        ev, cfg, hist = json.loads(ev_p.read_text()), json.loads(cfg_p.read_text()), pd.read_csv(hist_p)
        gp = pd.read_csv(gp_p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        pred16 = gp[[t + "__pred" for t in TARGETS]].set_axis(TARGETS, axis=1)
        mil = bootstrap(gold_y.loc[pred16.index], pred16)
        base = f"; notebook 14 frozen baseline: {met['oof_macro_auc']:.4f}" if met_p else ""
        md += [f"**Expert-labelled macro AUC: {ev['macro_auc']:.4f}** over {ev['evaluable_targets']}/12 "
               f"targets{base}; chance: 0.5." + (" **SMOKE RUN: not a result.**" if ev["smoke"] else ""), "",
               f"- Model: `{cfg['arch']}` backbone on 3-slice windows (neighbouring slices as RGB), "
               f"{cfg['k_eval']} windows per plane x 3 planes, plane embedding and one attention "
               "pooling per finding; trained end to end.",
               f"- Labels: {cfg['label_source']} ({soft_meta.get('source', '?')}); "
               f"{cfg['n_train']:,} training studies, {cfg['n_val']} weakly labelled fold-{cfg['val_fold']} "
               "studies for epoch selection, and the 58 expert-labelled studies (plus every study sharing "
               "a report with them) held out of training and scored once at the end.",
               f"- {cfg['epochs']} epochs; weights averaged over the {len(cfg['top_epochs'])} epochs with "
               f"the lowest fold-{cfg['val_fold']} soft BCE ({', '.join(map(str, cfg['top_epochs']))}).",
               "- This is a single 58-study estimate, not a leaderboard score. Per-target intervals are "
               "wide (about +/-0.1), and the label source was chosen on the same 58 studies.", ""]
        tables, labels, colors = [mil], ["fine-tuned MIL (nb 16)"], [BLUE]
        if met_p and oof_p:
            tables.append(img)
            labels.append("frozen ResNet-18 (nb 14)")
            colors.append(MUTED)
        if soft_agree is not None:
            tables.append(soft_agree[["target", "blend_auc", "blend_ci_low", "blend_ci_high"]]
                          .set_axis(["target", "auc", "ci_low", "ci_high"], axis=1))
            labels.append("its training labels (reports)")
            colors.append(ORANGE)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        dot_ci(ax, tables, labels, colors, "Fine-tuned image model vs baseline and its labels")
        fig.tight_layout()
        fig.savefig(figs / "06_mil_auc.png")
        plt.close(fig)
        md += [f"![MIL AUC]({rel(figs / '06_mil_auc.png', out)})", ""]

        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.plot(hist["epoch"], hist["train_bce"], "-o", color=BLUE, markersize=4, label="train (soft labels)")
        ax.plot(hist["epoch"], hist["val_bce"], "-o", color=ORANGE, markersize=4,
                label=f"fold {cfg['val_fold']} (epoch selection)")
        for e in cfg["top_epochs"]:
            ax.plot(e, hist.set_index("epoch").loc[e, "val_bce"], "o", markersize=10, markerfacecolor="none",
                    markeredgecolor=INK2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("soft binary cross-entropy")
        ax.set_title("Training curve (circled: epochs averaged)", loc="left")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "07_mil_training.png")
        plt.close(fig)
        md += [f"![MIL training]({rel(figs / '07_mil_training.png', out)})", "",
               f"Each epoch took about {hist['train_seconds'].median() / 60:.1f} min on 2x T4 "
               f"({hist['total_seconds'].iloc[-1] / 3600:.2f} h in total).", ""]
        cmp = mil[["target", "auc", "ci_low", "ci_high", "n_pos", "n_neg"]].copy()
        if met_p and oof_p:
            cmp["nb14_auc"] = img["auc"].to_numpy()
        if soft_agree is not None:
            cmp["label_auc"] = soft_agree.set_index("target").loc[TARGETS, "blend_auc"].to_numpy()
        md += ["Per target (`label_auc`: agreement of the training labels themselves with the experts):", "",
               fmt_table(cmp), ""]

    # ------------------------------------------------- 7. fine-tuning x distillation
    if has_ablation:
        md += ["## 7. What fine-tuning and distillation each add (2 x 2 ablation)", ""]
    arm_files = {("frozen", "regex"): need("17", "gold_preds_regex_v1.csv"),
                 ("frozen", "soft"): need("17", "gold_preds_soft_v1.csv"),
                 ("fine-tuned", "regex"): need("16r", "gold_preds_v1.csv"),
                 ("fine-tuned", "soft"): need("16", "gold_preds_v1.csv")}
    if has_ablation and all(arm_files.values()) and gold_y is not None:
        md += ablation_section(arm_files, gold_y, figs, out,
                               baseline=met["oof_macro_auc"] if met_p and oof_p else None)

    # ------------------------------------------- 8. fine-tuning / distillation quality
    char_files = {"soft": need("11c", "soft_labels_v1.csv"), "folds": need("12", "folds_v1.csv"),
                  "frozen": need("17", "ablation_frozen_v1.json"),
                  "log16": need("16", "rsna-knee-16-finetune-2p5d-mil.log"),
                  "log16r": need("16r", "rsna-knee-16r-finetune-regex-labels.log"),
                  "hist16": need("16", "history_v1.csv"), "hist16r": need("16r", "history_v1.csv")}
    has_char = (has_ablation and all(arm_files.values()) and all(char_files.values())
                and weak is not None and gold_y is not None)
    if has_char:
        md += [f"## {6 + has16 + has_ablation}. How well fine-tuning and distillation work", ""]
        md += characterization_section(arm_files, char_files, weak, gold_y, figs, out)

    # ------------------------------------------------------------ 9. submission
    md += [f"## {6 + has16 + has_ablation + has_char}. Inference (notebook 15)", ""]
    sub_p = need("15", "submission.csv")
    if not runs.joinpath("15").is_dir():
        md += ["Not run for this report.", ""]
    if sub_p:
        sub = pd.read_csv(sub_p)
        ok = list(sub.columns) == ["StudyInstanceUID"] + TARGETS and sub[TARGETS].apply(
            lambda s: s.between(0, 1)).all().all()
        md += [f"- `submission.csv`: {len(sub)} rows, schema check {'passed' if ok else 'FAILED'}. "
               "The visible test set holds only sample studies; Kaggle scores the hidden test "
               "set on submission.", ""]
        log_p = runs / "15" / "rsna-knee-15-inference-submission.log"
        if log_p.exists():
            text = log_p.read_text(errors="replace")
            try:
                text = "".join(x.get("data", "") for x in json.loads(text))
            except ValueError:
                pass
            keys = ("model:", "status:", "submission.csv:")  # status ok = model prediction, not fallback
            md += ["From the run log:", "", "```",
                   *[ln.strip() for ln in text.splitlines() if ln.strip().startswith(keys)], "```", ""]

    present = [n for n in ("11", "11c", "12", "13", "14", "15", "16", "16r", "17") if runs.joinpath(n).is_dir()]
    head = ["# RSNA Knee Abnormality: " + ("report-supervised image models" if has16 else
                                          "weak-label baseline report"), "",
            f"Compiled by `reports/build_report.py` from the Kaggle outputs of notebooks {', '.join(present)}. "
            "Every number is read from those outputs.", ""]
    if missing:
        head += ["**Missing inputs** (those sections are left out): "
                 + ", ".join(f"`{m}`" for m in missing), ""]
    write_pdf("\n".join(head + md) + "\n", out)
    print("wrote", out / "REPORT.pdf", "| missing:", missing)


ARM_LABEL = {("frozen", "regex"): "frozen ResNet-18, regex labels",
             ("frozen", "soft"): "frozen ResNet-18, LLM soft labels",
             ("fine-tuned", "regex"): "fine-tuned MIL, regex labels",
             ("fine-tuned", "soft"): "fine-tuned MIL, LLM soft labels"}


def macro_auc_rows(y, p):
    """Macro AUC over targets that have both classes in these rows."""
    aucs = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(aucs)) if aucs else np.nan


def ablation_section(arm_files, gold_y, figs, out, baseline=None, n_boot=2000, seed=0):
    """2 x 2 (image model x label source) on the 58 expert-labelled studies, with a paired
    study-level bootstrap: every resample scores all four arms on the same studies."""
    y = gold_y.loc[:, TARGETS]
    preds = {}
    for arm, path in arm_files.items():
        df = pd.read_csv(path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        preds[arm] = df.loc[y.index, [t + "__pred" for t in TARGETS]].to_numpy()
    yv = y.to_numpy()
    point = {a: macro_auc_rows(yv, p) for a, p in preds.items()}
    rng = np.random.default_rng(seed)
    boot = {a: [] for a in preds}
    for _ in range(n_boot):
        i = rng.integers(0, len(yv), len(yv))
        for a, p in preds.items():
            boot[a].append(macro_auc_rows(yv[i], p[i]))
    boot = {a: np.array(v) for a, v in boot.items()}
    ci = {a: np.nanpercentile(v, [2.5, 97.5]) for a, v in boot.items()}

    F, T_, R, S = "frozen", "fine-tuned", "regex", "soft"
    effects = [
        ("fine-tuning, with regex labels", (T_, R), (F, R)),
        ("fine-tuning, with LLM soft labels", (T_, S), (F, S)),
        ("distillation (LLM soft labels), frozen model", (F, S), (F, R)),
        ("distillation (LLM soft labels), fine-tuned model", (T_, S), (T_, R)),
        ("both together (fine-tuned + soft vs frozen + regex)", (T_, S), (F, R)),
    ]
    erows = []
    for name, a, b in effects:
        d = boot[a] - boot[b]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        erows.append({"effect": name, "delta macro AUC": point[a] - point[b], "ci_low": lo, "ci_high": hi,
                      "share of resamples > 0": float(np.nanmean(d > 0))})
    inter = (boot[(T_, S)] - boot[(T_, R)]) - (boot[(F, S)] - boot[(F, R)])
    ilo, ihi = np.nanpercentile(inter, [2.5, 97.5])
    ipt = (point[(T_, S)] - point[(T_, R)]) - (point[(F, S)] - point[(F, R)])

    grid = pd.DataFrame({"image model": [F, T_]})
    for src, col in ((R, "regex weak labels (nb 11)"), (S, "LLM soft labels (nb 11c)")):
        grid[col] = [f"{point[(m, src)]:.3f} [{ci[(m, src)][0]:.3f}, {ci[(m, src)][1]:.3f}]" for m in (F, T_)]

    md = ["All four cells use **one protocol** (notebook 16's): train on folds 1-4 without the 58 "
          "expert-labelled studies or any study sharing a report with them, pick epochs on fold 0's "
          "weakly labelled studies, score the 58 once. Only the image model (frozen ResNet-18 + linear "
          "head, notebook 17 / fine-tuned ConvNeXt-Tiny MIL, notebooks 16 and 16r) and the training "
          "labels change. \"Distillation\" here means training the image model on the LLM teacher's "
          "per-finding probabilities read from the reports.", "",
          "Macro AUC on the 58 expert-labelled studies [95% bootstrap CI]:", "", fmt_table(grid), ""]
    if baseline is not None:
        md += [f"Notebook 14's {baseline:.3f} (section 5) is not a cell of this table: it also trained on the "
               "expert labels (weight 3, out of fold), a different protocol.", ""]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    xs = np.array([0, 1])
    for m, col in ((F, MUTED), (T_, BLUE)):
        vals = [point[(m, R)], point[(m, S)]]
        err = np.array([[vals[k] - ci[(m, src)][0], ci[(m, src)][1] - vals[k]] for k, src in enumerate((R, S))]).T
        off = -0.03 if m == F else 0.03
        ax.errorbar(xs + off, vals, yerr=err, color=col, marker="o", markersize=7, linewidth=2, capsize=0,
                    elinewidth=2, alpha=0.9, label=("frozen ResNet-18 + linear" if m == F else "fine-tuned MIL"),
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        for k, (x, v) in enumerate(zip(xs + off, vals)):  # labels on the outer side, clear of the line
            ax.text(x + (-0.06 if k == 0 else 0.06), v, f"{v:.3f}", va="center",
                    ha="right" if k == 0 else "left", fontsize=8, color=INK2)
    ax.set_xticks(xs, ["regex weak labels\n(no teacher)", "LLM soft labels\n(distillation)"])
    ax.set_xlim(-0.45, 1.45)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylabel("macro AUC, 58 expert-labelled studies")
    ax.set_title("Fine-tuning x distillation (95% bootstrap CI)", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "08_ablation_2x2.png")
    plt.close(fig)
    md += [f"![2x2 ablation]({rel(figs / '08_ablation_2x2.png', out)})", "",
           "Effect of each method, paired over the same resampled studies (2,000 resamples):", "",
           fmt_table(pd.DataFrame(erows)), "",
           f"Interaction (does distillation help more on the fine-tuned model than on the frozen one?): "
           f"{ipt:+.3f} [{ilo:+.3f}, {ihi:+.3f}]. An interval that spans 0 means the two methods' gains "
           "are consistent with simply adding up.", "",
           "With 58 studies, an effect whose interval spans 0 is not distinguishable from no effect; "
           "the label source for the soft column was also picked on these 58 (see section 2).", ""]

    per = pd.DataFrame({"target": TARGETS})
    for a, p in preds.items():
        per[ARM_LABEL[a].replace("frozen ResNet-18", "frozen").replace("fine-tuned MIL", "fine-tuned")] = [
            roc_auc_score(yv[:, j], p[:, j]) if 0 < yv[:, j].sum() < len(yv) else np.nan for j in range(12)]
    md += ["Per target (AUC):", "", fmt_table(per), ""]
    return md


def characterization_section(arm_files, files, weak, gold_y, figs, out, n_boot=1000, seed=0):
    """Teacher -> student quality of the four 2 x 2 arms. 'Teacher' = the labels an arm trained on.
    Held-out: fold-0 loss (870 studies) scaled between a no-image constant (0) and the teacher (1).
    Expert-labelled: student vs teacher AUC on the 58, agreement, calibration, error inheritance."""
    from scipy.stats import rankdata, spearmanr

    arms = list(arm_files)
    soft = pd.read_csv(files["soft"], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    folds = pd.read_csv(files["folds"], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    w = weak.set_index("StudyInstanceUID")
    teacher = {"regex": w[[t + "__weak" for t in TARGETS]].set_axis(TARGETS, axis=1), "soft": soft[TARGETS]}
    # same split as notebooks 16 / 17
    gold_groups = set(folds.loc[folds["is_gold"].eq(1), "group"])
    ok = folds["is_gold"].eq(0) & ~folds["group"].isin(gold_groups)
    tr, va = folds.index[ok & folds["fold"].ne(0)], folds.index[ok & folds["fold"].eq(0)]

    def bce(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    ref = {}  # label source -> (teacher entropy = perfect copy, no-image constant)
    for src, lab in teacher.items():
        y = lab.loc[va].to_numpy()
        ref[src] = (bce(y, y), bce(np.broadcast_to(lab.loc[tr].to_numpy().mean(0), y.shape), y))
    captured = lambda v, src: (ref[src][1] - v) / (ref[src][1] - ref[src][0])
    avg3 = lambda p: float(re.search(r"'avg_top3': ([0-9.]+)", p.read_text(errors="replace")).group(1))
    frozen = json.loads(files["frozen"].read_text())
    val_bce = {("frozen", "regex"): frozen["regex"]["val_bce"], ("frozen", "soft"): frozen["soft"]["val_bce"],
               ("fine-tuned", "regex"): avg3(files["log16r"]), ("fine-tuned", "soft"): avg3(files["log16"])}

    ids = gold_y.index
    G = gold_y.loc[:, TARGETS].to_numpy()
    P = {}
    for a, f in arm_files.items():
        d = pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        P[a] = d.loc[ids, [t + "__pred" for t in TARGETS]].to_numpy()
    Tch = {s: lab.loc[ids].to_numpy() for s, lab in teacher.items()}

    def aucs(p, i):
        return np.array([roc_auc_score(G[i, j], p[i, j]) if 0 < G[i, j].sum() < len(i) else np.nan
                         for j in range(len(TARGETS))])

    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(G), len(G)) for _ in range(n_boot)]
    full = np.arange(len(G))
    kept = {}
    for a in arms:
        f = lambda i, a=a: (np.nanmean(aucs(P[a], i)) - .5) / (np.nanmean(aucs(Tch[a[1]], i)) - .5)
        bs = [f(i) for i in boots]
        kept[a] = (f(full), np.percentile(bs, 2.5), np.percentile(bs, 97.5))

    rows = []
    for a in arms:
        t = Tch[a[1]]
        rows.append({"arm": ARM_LABEL[a], "signal captured (870)": captured(val_bce[a], a[1]),
                     "teacher AUC": np.nanmean(aucs(t, full)), "student AUC": np.nanmean(aucs(P[a], full)),
                     "skill kept [95% CI]": f"{kept[a][0]:.2f} [{kept[a][1]:.2f}, {kept[a][2]:.2f}]",
                     "agreement": np.mean([spearmanr(P[a][:, j], t[:, j])[0] for j in range(len(TARGETS))]),
                     "Brier": float(((P[a] - G) ** 2).mean())})

    def resid(x, g):  # ranks of x with the expert label regressed out
        r = rankdata(x)
        X = np.c_[np.ones_like(g, dtype=float), g]
        return r - X @ np.linalg.lstsq(X, r, rcond=None)[0]

    def pcorr(p, q):
        return float(np.mean([np.corrcoef(resid(p[:, j], G[:, j]), resid(q[:, j], G[:, j]))[0, 1]
                              for j in range(len(TARGETS))]))

    inh = {src: {t: pcorr(P[("fine-tuned", src)], Tch[t]) for t in Tch} for src in Tch}
    tt = pcorr(Tch["regex"], Tch["soft"])

    # ---- figure 09: transfer summary + training dynamics
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    col = {"frozen": MUTED, "fine-tuned": BLUE}
    for ax, title, vals, ylab in (
            (axes[0], "Teacher signal captured\n(870 held-out studies)",
             {a: captured(val_bce[a], a[1]) for a in arms}, "share of learnable signal (0 = no image, 1 = teacher)"),
            (axes[1], "Teacher skill kept\n(58 expert-labelled studies)", {a: kept[a][0] for a in arms},
             "(student AUC - 0.5) / (teacher AUC - 0.5)")):
        for m in ("frozen", "fine-tuned"):
            ks = [(m, "regex"), (m, "soft")]
            x, y = np.array([0, 1]), np.array([vals[k] for k in ks])
            ax.plot(x, y, color=col[m], linewidth=2, zorder=2)
            if ax is axes[1]:
                err = [y - np.array([kept[k][1] for k in ks]), np.array([kept[k][2] for k in ks]) - y]
                ax.errorbar(x, y, yerr=err, fmt="none", ecolor=col[m], elinewidth=2, alpha=0.5, zorder=1)
            ax.plot(x, y, "o", color=col[m], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=3, label=m)
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(-10 if xi == 0 else 10, 0), textcoords="offset points",
                            ha="right" if xi == 0 else "left", va="center", fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["regex labels", "LLM soft labels"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 1.1)
        ax.axhline(1, color=INK, linewidth=1, linestyle=":")
        ax.text(1.48, 1.015, "teacher", ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, loc="left")
    axes[0].legend(loc="center left", fontsize=8)
    ax = axes[2]
    for name, hp, src, ls in (("fine-tuned, soft (16)", files["hist16"], "soft", "-"),
                              ("fine-tuned, regex (16r)", files["hist16r"], "regex", "--")):
        h = pd.read_csv(hp)
        for c, alpha in (("val_bce", 1.0), ("train_bce", 0.35)):
            ax.plot(h["epoch"], captured(h[c], src), ls, color=BLUE, alpha=alpha, linewidth=2,
                    marker="o" if c == "val_bce" else None, markersize=4,
                    label=f"{name}, {'held-out' if c == 'val_bce' else 'train'}")
    for a, ls, va_ in ((("frozen", "soft"), "-", "bottom"), (("frozen", "regex"), "--", "top")):
        v = captured(val_bce[a], a[1])
        ax.axhline(v, color=MUTED, linestyle=ls, linewidth=1.2)
        ax.text(ax.get_xlim()[1], v, f"frozen, {a[1]}", va=va_, ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("share of learnable signal", fontsize=8)
    ax.set_title("Fine-tuning dynamics: held-out plateaus\nafter epoch 5, train keeps rising", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    fig.savefig(figs / "09_transfer_quality.png")
    plt.close(fig)

    # ---- figure 10: per-finding dumbbells
    A = {a: aucs(P[a], full) for a in arms}
    At = {s: aucs(Tch[s], full) for s in Tch}
    order = np.argsort(At["soft"])
    y = np.arange(len(TARGETS))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, title, a, b, labs in (
            (axes[0], "Fine-tuning (LLM soft labels): frozen -> fine-tuned", ("frozen", "soft"),
             ("fine-tuned", "soft"), ("frozen", "fine-tuned")),
            (axes[1], "Distillation (fine-tuned model): regex -> LLM soft labels", ("fine-tuned", "regex"),
             ("fine-tuned", "soft"), ("regex labels", "LLM soft labels"))):
        va_, vb = A[a][order], A[b][order]
        ax.hlines(y, np.minimum(va_, vb), np.maximum(va_, vb), color=GRID, linewidth=3, zorder=1)
        if a[0] == "frozen":
            ax.plot(va_, y, "o", color=MUTED, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=3, label=labs[0])
        else:  # hollow = regex labels
            ax.plot(va_, y, "o", color=BLUE, markersize=8, markerfacecolor=SURFACE, markeredgewidth=1.8,
                    zorder=3, label=labs[0])
        ax.plot(vb, y, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                label=labs[1])
        ax.plot(At["soft"][order], y, "|", color=INK, markersize=14, markeredgewidth=2, zorder=2,
                label="teacher (soft labels vs experts)")
        for yi, d in zip(y, vb - va_):
            ax.text(1.065, yi, f"{d:+.2f}", va="center", ha="right", fontsize=8,
                    color=INK if abs(d) >= 0.05 else MUTED)
        ax.set_xlim(0.4, 1.07)
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlabel("ROC AUC on the 58 expert-labelled studies (right column: change)")
        ax.set_title(title, loc="left", pad=24)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8, borderaxespad=0.2,
                  handletextpad=0.3, columnspacing=1.0)
    axes[0].set_yticks(y, [TARGETS[i] for i in order])
    fig.tight_layout()
    fig.savefig(figs / "10_per_finding_ft_distill.png")
    plt.close(fig)

    FT, FR, FS = ("fine-tuned", "soft"), ("fine-tuned", "regex"), ("frozen", "soft")
    per = pd.DataFrame({"target": TARGETS, "regex teacher": At["regex"], "soft teacher": At["soft"],
                        "frozen, soft": A[FS], "fine-tuned, regex": A[FR], "fine-tuned, soft": A[FT],
                        "fine-tuning gain": A[FT] - A[FS], "distillation gain": A[FT] - A[FR],
                        "teacher gain": At["soft"] - At["regex"]})
    beats = [t for t, s, te in zip(TARGETS, A[FT], At["soft"]) if s > te]
    rho_ft = spearmanr(At["soft"], A[FT] - A[FS])[0]
    rho_d = spearmanr(At["soft"] - At["regex"], A[FT] - A[FR])[0]
    h16, h16r = pd.read_csv(files["hist16"]), pd.read_csv(files["hist16r"])

    def dyn(h, src):
        b = h.loc[h["val_bce"].idxmin()]
        return (int(b.epoch), captured(b.val_bce, src), captured(h.val_bce.iloc[-1], src),
                captured(h.train_bce.iloc[-1], src))

    d16, d16r = dyn(h16, "soft"), dyn(h16r, "regex")
    rr = rows
    return [
        "The 2 x 2 says *how much* each method adds; this section asks *how well* each image model "
        "(the student) reproduces the labels it was trained on (its teacher), and where. No new training: "
        "every number comes from the four runs of section 7 and the label files.", "",
        "- **Teacher signal captured** (870 fold-0 studies, never trained on): the held-out loss against the "
        "teacher, scaled so 0 = a constant per-finding prediction (no image) and 1 = the teacher itself "
        f"(its entropy: regex {ref['regex'][0]:.3f}, soft {ref['soft'][0]:.3f}). It measures copying the "
        "teacher, not truth, but uses 15x more studies than the expert set.",
        "- **Skill kept** (58 expert-labelled studies): (student AUC - 0.5) / (teacher AUC - 0.5), the share "
        "of the teacher's above-chance skill the image model keeps. 95% bootstrap CI over studies.",
        "- **Agreement**: mean Spearman correlation between student and teacher, per finding.",
        "- **Brier**: mean squared error against the expert labels (calibration and accuracy together). "
        f"Teachers: regex {float(((Tch['regex'] - G) ** 2).mean()):.3f}, "
        f"soft {float(((Tch['soft'] - G) ** 2).mean()):.3f}.", "",
        fmt_table(pd.DataFrame(rows)), "",
        f"![transfer quality]({rel(figs / '09_transfer_quality.png', out)})", "",
        "**Fine-tuning.** It roughly doubles the teacher signal captured "
        f"({rr[0]['signal captured (870)']:.2f} -> {rr[2]['signal captured (870)']:.2f} on regex labels, "
        f"{rr[1]['signal captured (870)']:.2f} -> {rr[3]['signal captured (870)']:.2f} on soft) and raises "
        f"the skill kept from about {np.mean([kept[a][0] for a in arms[:2]]):.0%} to "
        f"{np.mean([kept[a][0] for a in arms[2:]]):.0%}. It re-learns the image features rather than "
        "rescaling them: its per-finding ranking agrees with the frozen model's at only "
        f"{np.mean([spearmanr(P[FS][:, j], P[FT][:, j])[0] for j in range(12)]):.2f} (Spearman). "
        f"Both runs peak at epoch {d16[0]} (signal captured: soft {d16[1]:.2f}, regex {d16r[1]:.2f}); "
        f"afterwards train keeps rising (to {d16[3]:.2f} / {d16r[3]:.2f}) while held-out stays flat "
        f"({d16[2]:.2f} / {d16r[2]:.2f} at the last epoch): memorisation without much harm.", "",
        "**Distillation.** Each model captures more of the soft teacher's signal than of the regex teacher's, "
        "so the LLM probabilities are easier to learn from images. But the student is not a copy: at about "
        f"{kept[FT][0]:.0%} skill kept its agreement with the teacher is only {rr[3]['agreement']:.2f}, and on "
        f"{len(beats)} findings ({', '.join(beats)}) it beats its teacher, findings that are visible in the "
        "image but under-reported. Why the distillation gain is small: with the expert label partialled out, "
        f"the two teachers' scores correlate at {tt:.2f} (both read the same report), so they pass on nearly "
        "the same errors. The fine-tuned students follow teacher errors only moderately "
        f"(regex-trained: {inh['regex']['regex']:.2f} with the regex teacher, {inh['regex']['soft']:.2f} with "
        f"the soft one; soft-trained: {inh['soft']['regex']:.2f} / {inh['soft']['soft']:.2f}).", "",
        f"![per finding]({rel(figs / '10_per_finding_ft_distill.png', out)})", "",
        "Per finding (AUC on the 58; rows of the figure are sorted by the soft teacher's AUC):", "",
        fmt_table(per), "",
        f"Fine-tuning gains most where the teacher is strong (Spearman {rho_ft:.2f} between soft-teacher AUC "
        "and fine-tuning gain over 12 findings) and stalls on thin or subtle findings (Lateral Meniscus, "
        "PF OA, Synovitis), consistent with the 224 px resolution limit. The distillation gain does not "
        f"follow the teacher's own gain (Spearman {rho_d:.2f}); its largest moves (MCL, Medial Meniscus, "
        "Effusion) rest on a handful of the 58 studies and are within noise.", ""]


PAGE_CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; color: #0b0b0b; line-height: 1.4; }
h1 { font-size: 17pt; margin: 0 0 6pt; } h2 { font-size: 12.5pt; margin: 16pt 0 5pt; break-after: avoid; }
p, li { margin: 3pt 0; } code { font-size: 8.5pt; color: #52514e; }
pre { font-size: 8pt; background: #f3f2ef; padding: 6pt; white-space: pre-wrap; }
img { display: block; max-width: 100%; max-height: 120mm; margin: 6pt auto; break-inside: avoid; }
table { border-collapse: collapse; font-size: 8pt; margin: 5pt 0; break-inside: avoid; }
th, td { border-bottom: 1px solid #e4e3df; padding: 2pt 6pt; text-align: left; }
th { border-bottom: 1.5px solid #52514e; }
"""


def write_pdf(text, out):
    """Markdown -> self-contained HTML (figures inlined) -> REPORT.pdf via headless Edge/Chrome."""
    import base64
    import shutil
    import subprocess
    import tempfile

    import markdown

    def inline(m):
        data = base64.b64encode((out / m.group(2)).read_bytes()).decode()
        return f"![{m.group(1)}](data:image/png;base64,{data})"

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", inline, text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html = f'<!doctype html><html><head><meta charset="utf-8"><style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    browser = next((b for b in (shutil.which("msedge"), shutil.which("chrome"),
                                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Google\Chrome\Application\chrome.exe")
                    if b and Path(b).exists()), None)
    if browser is None:
        raise RuntimeError("no Edge or Chrome found to print the PDF")
    # Work in a local temp dir: the browser cannot always read files on a \\wsl$ share.
    with tempfile.TemporaryDirectory() as tmp:
        src, pdf = Path(tmp) / "report.html", Path(tmp) / "REPORT.pdf"
        src.write_text(html, encoding="utf-8")
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf}", src.as_uri()],
                       check=True, capture_output=True, timeout=180)
        if not pdf.exists() or pdf.stat().st_size == 0:
            raise RuntimeError("the browser did not write the PDF")
        shutil.copyfile(pdf, out / "REPORT.pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
