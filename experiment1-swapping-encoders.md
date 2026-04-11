# Experiment 1 — Encoder Ablation Study

## Motivation

Previous work in multilingual coreference resolution (including the CRAC shared task and the CorePipe system) suggests that model performance is strongly influenced by the choice of contextual encoder. However, it remains unclear which encoder properties are most beneficial for Czech coreference resolution specifically.

The goal of this experiment is to isolate the effect of the encoder while keeping the rest of the coreference system unchanged.

## Research Question

How does the choice of pretrained encoder influence Czech coreference resolution performance?

## Hypotheses
- H1: Multilingual encoders outperform Czech-only encoders due to cross-lingual transfer.
- H2: Slavic-oriented encoders improve performance because of linguistic similarity.
- H3: Larger encoder capacity correlates with higher coreference resolution performance.

## Experimental Design

The baseline coreference system will be used as a fixed experimental framework.

Only the encoder component will be modified.

All other factors remain constant:

- model architecture
- training procedure
- dataset
- evaluation metric
- preprocessing pipeline

This ensures a controlled ablation study.

## Encoder Categories

Encoders will be selected from three groups:

1. Multilingual Encoders

General-purpose multilingual language models.

Examples:

- multilingual BERT
- XLM-R Base
- XLM-R Large

2. Czech-Specific Encoders

Models pretrained primarily on Czech data.

Examples:

- Czech BERT variants
- CZERT / RobeCzech (or comparable models)

3. Slavic Encoders

Models trained on multiple Slavic languages.

Examples:

- SlavicBERT or related models
- Implementation Considerations
- Encoder Compatibility

Direct replacement of HuggingFace encoders may lead to severe performance degradation due to:

tokenization differences,
hidden size mismatch,
span alignment errors.

To ensure comparability:

an encoder wrapper will standardize output representations,
encoder outputs will be projected to a fixed hidden dimension expected by the baseline model,
token–subtoken alignment must be verified.

## Training Stability

Different encoders require different optimization settings.

Learning rate will be tuned individually for each encoder while keeping other parameters fixed.

## Evaluation Setup

Training data:

Czech portion of the CorefUD dataset (initially monolingual).

## Evaluation metric:

standard coreference F1 score used in CRAC evaluation.

Each experiment run represents:

baseline architecture + single encoder

## Expected Outcome

The experiment aims to determine:

how much performance variance is explained solely by encoder choice,
whether multilingual transfer benefits Czech,
whether language-specific pretraining provides measurable advantages.
Contribution to Thesis

This experiment establishes the encoder as a primary variable influencing performance and forms the foundation for later experiments involving multilingual training and system-level modifications.