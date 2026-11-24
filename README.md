# RSNA Knee Abnormality Detection

Learning-first exploration of knee MRI studies for the RSNA Kaggle competition,
working toward a reproducible end-to-end baseline. Each numbered notebook is one
step; all of them run on Kaggle with the competition data attached.

## Task and data (verified against the competition CSVs)

- 4,407 training studies; the target is one probability per study for 12 findings
  (`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`,
  `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`).
- Metric: mean of the 12 per-finding ROC AUCs.
- **Only 58 studies carry the 12 expert labels.** The other 4,349 have a free-text
  radiology report (about 9 languages) and no labels.
- Identifier levels: labels are per study (`StudyInstanceUID`); `train_series.csv`
  maps each study to 3-14 series (plane, fluid-sensitive, fat-suppressed flags);
  images are single-slice DICOM files under `train_series/<study>/<series>/`.
  The CSVs contain no patient identifier.
- `test.csv` has no reports, so report-derived information can supervise training but
  is not available at inference.

## Status

| Notebook | What it does | Status |
|---|---|---|
| 01-05 | dataset overview, DICOM metadata, visualisation, normalisation, augmentation | run on Kaggle |
| 06 | 58 expert-labelled studies -> 46/12 split; one series per plane | run |
| 07 | DICOM -> ordered, normalised 224 px volumes for those 58 studies | run (174/174 series ok) |
| 08 | PyTorch Dataset/DataLoader | run |
| 09 | frozen ResNet-18 features + linear head (single split) | run |
| 10 | 5-fold CV of 09 | run: **OOF macro AUC 0.526**, chance level |
| 11 | report -> weak labels for all 4,407 studies (rule-based, multilingual) | run: agreement with experts **macro AUC 0.862** |
| 11b | LLM report labels via [model-adapt](https://github.com/Z-bros/model-adapt-overcome) | abandoned: model-adapt is private and cannot be pip-installed on Kaggle; superseded by 11c |
| 11c | Qwen2.5-7B-Instruct reads every report -> P("yes") per finding (soft labels) | run: LLM 0.866, regex+LLM blend **0.875**; 4,407 reports in 1.28 h |
| 12 | leakage-aware 5-fold split (report-hash groups, greedy balancing of expert-labelled studies) | run: 881-882 studies and 11-12 expert-labelled per fold |
| 13 | all studies -> `(3, 16, 224, 224)` uint8 cache via [medprep](https://github.com/Z-bros/Med_Im_Prepper), orientation + laterality normalised | run: 4,407/4,407 ok, 0 plane mismatches |
| 14 | frozen ResNet-18 + linear head on weak labels (masked BCE), OOF evaluation | run: **OOF macro AUC 0.663** |
| 16 | fine-tuned ConvNeXt-Tiny 2.5D attention-MIL on 11c soft labels | run: **expert-labelled macro AUC 0.841** |
| 16r | 16 with notebook 11's regex labels instead of 11c's (ablation) | run: **0.828** |
| 17 | frozen ResNet-18 + linear head under 16's protocol, on regex and on soft labels (ablation, CPU) | run: regex **0.706**, soft **0.721** |
| 15 | offline inference -> `submission.csv` with schema checks (loads 14's or 16's bundle) | run with 16's model: 3/3 visible test studies ok; **not submitted** |

`kaggle-rsna-competition.ipynb` is a separate competition notebook; the laterality
and slice-ordering ideas in 13 come from it. The report extractor in 11 is copied
verbatim from my competition notebook *rsna-knee-read-the-report-then-the-knee*.

## Results

All numbers are measured on the **58 expert-labelled studies**, the only ground truth in the
training data. With 9-35 positives per finding, per-finding 95% intervals are about +/-0.1,
so differences under about 0.05 in macro AUC are within noise. These are not leaderboard scores.

| Model | Trained on | Macro AUC (58 studies) |
|---|---|---|
| 10: frozen ResNet-18, 58 expert labels only | 46 studies per fold | 0.526 (chance) |
| 14: frozen ResNet-18 + linear head | regex weak labels, 4,407 studies (pooled OOF) | 0.663 |
| **16: fine-tuned ConvNeXt-Tiny 2.5D attention-MIL** | 11c soft labels, 3,475 studies | **0.841** |
| *reference: the training labels themselves (reports)* | *not available at test time* | *0.875* |

16 per finding: Medial OA 0.97, Baker's 0.96, Contusion 0.93, Fracture 0.87, Effusion 0.87,
ACL 0.85, Lateral OA 0.84, Medial Meniscus 0.82, MCL 0.80, PF OA 0.76, Lateral Meniscus 0.74,
Synovitis 0.69. Large, high-contrast findings are learned best. The biggest gaps against
the labels are small structures (ACL 0.85 vs 0.97, MCL 0.80 vs 0.96), which points to image
resolution rather than label quality as the next limit.

### What fine-tuning and distillation each add (2 x 2 ablation)

Same protocol for all four cells (16's split; epochs picked on fold 0's weakly labelled studies;
the 58 scored once). "Distillation" = training the image model on the LLM teacher's
probabilities (11c) instead of the regex scores (11).

| Image model | regex weak labels (11) | LLM soft labels (11c) |
|---|---|---|
| frozen ResNet-18 + linear head (17) | 0.706 | 0.721 |
| fine-tuned ConvNeXt-Tiny MIL (16r / 16) | 0.828 | **0.841** |

Paired bootstrap over the 58 studies (2,000 resamples): fine-tuning adds **+0.12**
(95% CI +0.08 to +0.17) with either label source; distillation adds +0.016 (-0.005 to +0.038)
on the frozen model and +0.013 (-0.009 to +0.036) on the fine-tuned one, positive in 94% / 87%
of resamples but not distinguishable from zero with 58 studies. The two gains add up
(interaction -0.003). Almost all of the gain over the baseline comes from fine-tuning.

How well each model reproduces its teacher (report v2, section 8): fine-tuning raises the share of the
teacher's skill kept on the 58 from about 58% to 91%, and doubles the teacher signal captured on 870
held-out studies (0.19 -> 0.34 with soft labels). Distillation stays small because the regex and LLM
teachers share most of their errors (partial correlation 0.87: both read the same report). The fine-tuned
model beats its teacher on 5 image-visible findings (Medial OA, Lateral OA, Effusion, Baker's, Contusion).

Caveats: the label source (blend) was chosen on the same 58 studies, and 14 is pooled out of
fold over five models while 16 is one model scored on studies it never saw. Full write-ups:
[`reports/baseline-v1/REPORT.pdf`](reports/baseline-v1/REPORT.pdf) (11-14) and
[`reports/v2/REPORT.pdf`](reports/v2/REPORT.pdf) (11-17, incl. the 2 x 2 ablation and the fine-tuning / distillation characterization), built by `reports/build_report.py`
from the Kaggle outputs.

The approach of 11c + 16 (LLM soft labels from the reports, then a fine-tuned 2.5D
attention-MIL image model, with the expert-labelled studies used only for validation)
follows the top public notebooks of this competition (0.92-0.958 public leaderboard). The
implementation here is our own; no code is copied from them.

## Run order on Kaggle

Attach the competition data to every notebook. "Input" means **Add Input -> Notebook
output** of the earlier notebook. Each notebook finds its inputs by file name.

| Order | Notebook | Accelerator / internet | Extra inputs | Main outputs |
|---|---|---|---|---|
| 1 | 11 | CPU / off | - | `weak_labels_v1.csv`, `weak_label_agreement_v1.csv` |
| 2 | 11c | GPU 2x T4 / **on** (Qwen2.5-7B weights) | 11 | `soft_labels_v1.csv`, `soft_label_agreement_v1.csv`, `soft_labels_meta_v1.json` |
| 3 | 12 | CPU / off | 11 | `folds_v1.csv`, `study_headers_v1.csv` |
| 4 | 13 | CPU / **on** (installs medprep) | - | `cache/*.npz`, `preprocess_manifest_v1.csv` |
| 5 | 14 (baseline) | GPU / **on** (ResNet-18 weights) | 11, 12, 13 | `model/`, `oof_gold_v1.csv`, `metrics_v1.json` |
| 6 | 16 | GPU 2x T4 / **on** (ConvNeXt weights) | 11, 11c, 12, 13 | `model/`, `gold_eval_v1.json`, `history_v1.csv` |
| 6a | 16r (ablation) | GPU 2x T4 / **on** (ConvNeXt weights) | 11, 12, 13 | `model/`, `gold_eval_v1.json`, `gold_preds_v1.csv` |
| 6b | 17 (ablation) | CPU / off | 11, 11c, 12, 14 | `ablation_frozen_v1.json`, `gold_preds_{regex,soft}_v1.csv` |
| 7 | 15 | GPU / **off** | **one** of 16 or 14, medprep dataset | `submission.csv` |

Notes:
- 13: run once with `LIMIT = 20` and check the audit cell and montage, then with
  `LIMIT = None`. It resumes where it stopped. Expect about 10.6 GB of output.
- 11c: run with `STAGE = "pilot"` (58 studies, about 5 min) to compare the label sources and
  project the run time, then with `STAGE = "full"` (about 1.3 h on 2x T4).
- 16: `SMOKE = True` is a 64-study wiring check; the full run (`SMOKE = False`) took 1.6 h on
  2x T4. 11c needs a completed version before Kaggle accepts it as an input.
- 16r is 16 with `LABELS = "regex"` (1.7 h on 2x T4); 17 reuses 14's frozen features (2 min on CPU).
  Both exist only for the ablation and are not needed for a submission.
- 15: upload the Med_Im_Prepper `medprep/` folder (or a wheel built with
  `python -m pip wheel . --no-deps -w dist`) as a Kaggle dataset, because internet is off.
  It reads `model_kind` from the attached bundle's `config.json` and refuses a SMOKE bundle.
- The competition data has been mounted at both `/kaggle/input/competitions/<slug>` and
  `/kaggle/input/<slug>` on different runs; every notebook looks in both.
- Pushing from the command line: `kaggle/<slug>/` holds only `kernel-metadata.json`, whose
  `code_file` is the notebook in the repo root. Copy both into one folder, then push it, e.g.
  `mkdir -p /tmp/k && cp kaggle/rsna-knee-12-folds/kernel-metadata.json 12-folds.ipynb /tmp/k/ && kaggle kernels push -p /tmp/k`.
  Add `--accelerator NvidiaTeslaT4` for GPU notebooks (`enable_gpu: true`).

## Design decisions and known limitations

- **Supervision.** 14 trains on expert labels where they exist (weight 3) and on the
  report extractor's scores elsewhere; targets a report never mentions are masked
  (weight 0). 16 trains only on 11c's soft labels (mean of the regex score and the LLM's
  P("yes")); the 58 expert-labelled studies, and every study sharing a report with one of
  them, are excluded from its training and scored once at the end. Its epochs are chosen
  on fold 0's weakly labelled studies, never on the 58.
- **Evaluation** uses only the 58 expert labels, pooled out of fold. With 9-35
  positives per finding, intervals are wide. Findings with a single class in a fold
  are reported as undefined (NaN) and excluded from that fold's mean.
- **Patient-level separation** cannot be read from the CSVs. 12 groups studies that
  share an identical report. The DICOM `PatientID` turned out to be unique per study
  (anonymised), so **patient-level separation cannot be verified**; separation is
  study-level and report-level only.
- **Leakage control.** Nothing in preprocessing is fitted on data (per-volume
  normalisation). Feature standardisation is fitted on training folds only. The
  training length is chosen on weakly labelled studies, never on the 58.
- **DICOM.** Slices are ordered by patient coordinate. The modality LUT is applied via
  medprep. MONOCHROME1, pixel padding, multiframe data and oblique orientations are
  rejected per series and logged. Volumes are flipped (never rotated) to a fixed
  patient-axis orientation, and right knees are mirrored to a left-knee convention.
  Within 20 mm of the midline the side stays unresolved, is counted, and is not
  flipped.
- **Models.** 14's encoder is frozen ImageNet ResNet-18 with slice-mean pooling, a
  deliberately simple baseline with no augmentation. 16 fine-tunes ConvNeXt-Tiny end to end
  on 3-slice windows (neighbouring slices as RGB), 8 per plane, with a plane embedding and
  one attention pooling per finding, training-only medprep augmentation, and weights
  averaged over its three best epochs.

## Roadmap

1. ~~Run 11-15 on Kaggle and record the metrics.~~ Done (14: 0.663).
2. ~~Fine-tune the image model with augmentation.~~ Done as 11c + 16 (0.841); 2 x 2 ablation (16r, 17) shows fine-tuning gives most of the gain.
3. Submit 15 with 16's model for a leaderboard score (not yet done).
4. Past 0.85: a 336 px cache with more slices for small structures (ACL, MCL, menisci),
   a stronger backbone (e.g. CoAtNet), or an ensemble.
5. **Fine-tuning / distillation via model-adapt** (needs a training milestone in that
   repository first): LoRA on an LLM report teacher, then report-teacher ->
   image-student distillation.
6. **Weight-space programming (research).** A general-purpose LLM reads or edits a
   small network's weights directly, without gradient descent. Candidate target: 14's
   linear head (1539 -> 12 per fold), not the encoder.
   - Read: ask which plane's features drive each finding, and check against anatomy.
   - Edit: make targeted edits (e.g. bias recalibration) and compare them with
     gradient-trained heads.
   - An edit is kept only if it improves out-of-fold macro AUC on folds it was not
     designed on.
   - This is early preprint work (late 2026) and reported to be fragile beyond small
     networks:
     [overview](https://www.datacamp.com/blog/weight-space-programming),
     [medical imaging](https://arxiv.org/pdf/2611.12066),
     [critical evaluation](https://arxiv.org/pdf/2611.08070).

## Setup

Kaggle: attach the competition dataset through **Add Input**. Keep competition data
out of Git. Locally, `requirements.txt` lists the packages the notebooks import.
