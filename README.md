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
| 11 | report -> weak labels for all 4,407 studies (rule-based, multilingual) | written, not yet run |
| 11b | LLM report labels via [model-adapt](https://github.com/Z-bros/model-adapt-overcome), compared with 11 | written, not yet run |
| 12 | leakage-aware 5-fold split (report-hash + PatientID groups) | written, not yet run |
| 13 | all studies -> `(3, 16, 224, 224)` uint8 cache via [medprep](https://github.com/Z-bros/Med_Im_Prepper), orientation + laterality normalised | written, not yet run |
| 14 | weak-label training (masked BCE), OOF evaluation on the 58 expert labels | written, not yet run |
| 15 | offline inference -> `submission.csv` with schema checks | written, not yet run |

`kaggle-rsna-competition.ipynb` is a separate competition notebook; the laterality
and slice-ordering ideas in 13 come from it. The report extractor in 11 is copied
verbatim from my competition notebook *rsna-knee-read-the-report-then-the-knee*.

### Verified locally for 11-15 (no data, nothing executed on Kaggle yet)

- Every code cell parses.
- 11: the extractor's self-check (negation, medial/lateral side, silent report) passes on
  synthetic sentences.
- 13/15: the orientation and laterality self-test passes with the local medprep; the
  preprocessing cell is byte-identical in 13 and 15.
- 11b: the generated model-adapt config validates with `model_adapt.core.load_config`.
- 14: the masked-BCE head learns synthetic signal and ignores fully masked targets.

No model results exist yet for 11-15. Record them here only after a Kaggle run.

## Run order on Kaggle

Attach the competition data to every notebook. "Input" means **Add Input -> Notebook
output** of the earlier notebook. Each notebook finds its inputs by file name.

| Order | Notebook | Accelerator / internet | Extra inputs | Main outputs |
|---|---|---|---|---|
| 1 | 11 | CPU / off | - | `weak_labels_v1.csv`, `weak_label_agreement_v1.csv` |
| 2 | 11b (optional) | GPU T4 / **on** | 11 | model-adapt run dir, `llm_vs_regex_v1.csv` |
| 3 | 12 | CPU / off | 11 | `folds_v1.csv`, `study_headers_v1.csv` |
| 4 | 13 | CPU / **on** (installs medprep) | - | `cache/*.npz`, `preprocess_manifest_v1.csv` |
| 5 | 14 | GPU / **on** (ResNet-18 weights) | 11, 12, 13 | `model/`, `oof_gold_v1.csv`, `metrics_v1.json` |
| 6 | 15 | GPU / **off** | 14, medprep dataset | `submission.csv` |

Notes:
- 13: run once with `LIMIT = 20` and check the audit cell and montage, then with
  `LIMIT = None`. It resumes where it stopped. Expect about 10.6 GB of output.
- 15: upload the Med_Im_Prepper `medprep/` folder (or a wheel built with
  `python -m pip wheel . --no-deps -w dist`) as a Kaggle dataset, because internet is off.

## Design decisions and known limitations

- **Supervision.** 14 trains on expert labels where they exist (weight 3) and on the
  report extractor's scores elsewhere. Targets a report never mentions are masked
  (weight 0). Labels derived from reports are noisy: the extractor agreed with the
  expert labels at macro AUC 0.862 on the same 58 studies in the earlier
  competition run. 11 re-measures this.
- **Evaluation** uses only the 58 expert labels, pooled out of fold. With 9-35
  positives per finding, intervals are wide. Findings with a single class in a fold
  are reported as undefined (NaN) and excluded from that fold's mean.
- **Patient-level separation** cannot be read from the CSVs. 12 groups studies that
  share an identical report and, where the DICOM `PatientID` is usable, studies of one
  patient. If the IDs turn out to be blank or unique per study, separation is only
  study-level, and 12's printed counts will show it.
- **Leakage control.** Nothing in preprocessing is fitted on data (per-volume
  normalisation). Feature standardisation is fitted on training folds only. The
  training length is chosen on weakly labelled studies, never on the 58.
- **DICOM.** Slices are ordered by patient coordinate. The modality LUT is applied via
  medprep. MONOCHROME1, pixel padding, multiframe data and oblique orientations are
  rejected per series and logged. Volumes are flipped (never rotated) to a fixed
  patient-axis orientation, and right knees are mirrored to a left-knee convention.
  Within 20 mm of the midline the side stays unresolved, is counted, and is not
  flipped.
- **Model.** The encoder is frozen ImageNet ResNet-18 with slice-mean pooling, a
  deliberately simple baseline. There is no augmentation, because features are computed
  once.

## Roadmap

1. Run 11-15 on Kaggle; record the metrics here (macro OOF AUC, per finding, against 0.526).
2. Fine-tune the encoder with training-only augmentation (medprep
   `MedicalPipeline(training=True)`), using the same folds and evaluation.
3. **Fine-tuning / distillation via model-adapt** (needs a training milestone in that
   repository first): LoRA on an LLM report teacher, then report-teacher ->
   image-student distillation.
4. **Weight-space programming (research).** A general-purpose LLM reads or edits a
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
