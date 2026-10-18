# TermMap

Offline clinical term mapping to the supplied ICD-10-CM and SNOMED CT US dictionaries.

## Measured result and deliverables

Complete supplied-fold validation: **ICD-10-CM 47.87%, SNOMED CT US 38.71%; headline 38.71%. The 95% target is not met.** All 9,644 retained labels are counted. ICD uses the learned ranker; SNOMED uses the stronger dense baseline. The same OOF comparison selected these methods; this is model-selection validation, not an untouched hidden test.

- [Public weights and model card](https://huggingface.co/Vorun/strawhats-termmap)
- [Submission report](https://huggingface.co/Vorun/strawhats-termmap/resolve/main/TermMap_Submission_Report.pdf)
- [2:30 silent demo walkthrough](https://huggingface.co/Vorun/strawhats-termmap/resolve/main/TermMap_Demo.mp4) uses actual captured application screens.
- Validation details: `docs/VALIDATION_REPORT.json`.

The public 1,799,429,897-byte archive SHA-256 is `306e7fa53c32122e66d8b0a7ab7736af0955757decde73b748cff45b522e2eb1`. It contains frozen encoder/index weights, trained rankers and hashed support-example keys. It excludes raw clinical notes and organizer dictionary tables. Download `termmap_model_weights.zip` from the model repository without an API key.

## Method

1. Normalize the extracted entity and retain a bounded surrounding context window, marking the entity when present.
2. Encode the term and the term plus context using the frozen biomedical SapBERT encoder. Search every nonempty normalized alias in both supplied dictionaries. Long encoder inputs use overlapping windows with complete token coverage.
3. Union dense term, dense context and character 3-4 gram TF-IDF candidates. Expand ambiguous aliases to every associated legal code.
4. Add candidates from labelled example memory. Validation memory contains only the other folds. Rank with fixed small gradient-boosted trees using similarity, lexical overlap, specificity/qualifier signals and training-only memory.
5. Return one best code from each dictionary. Debug output contains the top 20 candidates and ranking scores. Scores are uncalibrated and are not probabilities that a clinical code is correct.

The complete index covers **480,306 concepts**, **775,013 unique readable normalized names**, and **786,566 name-code pairs**. All 9,644 retained labels and all 7,522 input instances participate. No `subconcept` field is used as an input or a filter. Ten symbol-only aliases are omitted from search because they normalize to empty strings; every affected code retains readable aliases, so no code is removed.

## Reproduce offline prediction

Use Python 3.13. Prepare the original dictionary tables in a separate setup environment with **pandas==3.0.1, pyarrow==21.0.0, openpyxl==3.1.5**; these versions reproduce their pinned Parquet bytes. Keep the original two XLSX files in `organizer-data/`.

```sh
python -m venv .setup
.setup/bin/python -m pip install pandas==3.0.1 pyarrow==21.0.0 openpyxl==3.1.5
.setup/bin/python -m genlink.setup_runtime --weights termmap_model_weights.zip --source organizer-data --output artifacts/runtime
```

On Windows use `.setup/Scripts/python.exe`. Then use a separate inference environment with `requirements-retrieval.txt` and install this source with `python -m pip install --no-deps -e .`. Cache dependencies during setup; inference is offline and verifies all resource/source hashes.

```sh
termmap predict --artifacts artifacts/runtime --input input.csv --output predictions.csv
```

Equivalent source invocation:

```sh
python -m genlink.retrieval_cli predict --artifacts artifacts/runtime --input input.csv --output predictions.csv
```

Input columns: `entity,chunk_text`. `clinical_entity` is accepted as an alias for `entity`. If context is provided separately, use input `chunk_id,clinical_entity` and add `--chunks chunks.csv`; the lookup must contain one `chunk_text` per `chunk_id`. Identifiers are read as strings, preserving leading zeros. For entity-only input add `--contextless`; an explicit empty `chunk_text` is also accepted.

Output preserves all input columns and row order and appends `icd10cm_code,snomed_code`. Codes remain strings. The separate `predictions_debug.jsonl` contains top-20 candidates, matched aliases, ranking scores and margins. `predictions.metrics.json` reports runtime and output SHA-256. Existing prediction output files are not overwritten. Both systems always receive a legal dictionary code; there is no abstention used to raise measured accuracy.

`--device cuda` uses a compatible GPU. `--device cpu` is supported by the implementation; CPU performance and numerical agreement must be described separately from the tested GPU environment. Previously encoded input strings can reuse frozen query vectors; no evaluation labels are used by that cache.

## Validation and leakage controls

The original supplied three folds are fixed. Each held-out row is scored by a model trained only on the other folds. Dictionary TF-IDF and embeddings use static dictionary names, not validation labels. Training-memory features for fitting exclude every example from the current example's canonical chunk, preventing self-neighbor label leakage. Held-out predictions have no gold-code injection; training positives missing from candidates are inserted only into training matrices. All final rankers use every labelled row in their own system.

Hyperparameters are fixed, with seed 42 and no held-out early stopping. Report ICD and SNOMED exact top-1 separately; the headline is their minimum. Candidate recall, top-20, rare and unseen codes, per-fold accuracy, deterministic output checks and genuine errors are included in the completed validation report. These are internal out-of-fold results, not hidden-test scores.

Two source rows with invalid ICD code `L6.0` were removed during the documented input audit, leaving 2,304 ICD and 7,340 SNOMED labels. The original source files are unchanged. See `G0_REPORT.md` for alignment hashes and exclusions.

## Build and tests

The source includes full-catalogue preparation, checkpointed dual-GPU embedding, complete hybrid retrieval, fixed-fold ranking and final prediction modules. The `kaggle/RETRIEVAL_*` scripts record the actual private notebook workflow and explicit compute gates. They are experiment launchers, not scripts to run indiscriminately in sequence. Each phase checks completion of prerequisites and records resource/source hashes.

```sh
python -m pytest tests/test_retrieval.py tests/test_rank_retrieval.py -q --basetemp work/tests
```

The runtime manifest and model hashes are checked before mapping. Interrupted retrieval can resume committed per-instance/system SQLite checkpoints with the same source and resource identity. Completed fold/final models and compute measurements are retained.

## Resources, licensing and privacy

- Encoder: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`. The official model card lists Apache-2.0 and describes UMLS 2020AA training provenance. Model source and file hashes are retained in `MODEL_SOURCE.json`. [Model card](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext), [paper](https://aclanthology.org/2021.naacl-main.334/).
- Dictionaries, folds and clinical examples are the organizer-supplied proprietary resources. No public redistribution is performed. Raw clinical notes, supplied dictionary tables and debug text remain private. Public model state uses hashed example keys and numeric support vectors; authorized evaluators reconstruct dictionary metadata from their supplied XLSX files.
- Clinical data and code were uploaded to explicitly authorized **private Kaggle datasets and a private notebook**, using T4 x2. Internet was enabled only to install/cache public dependencies and pinned public models, then disabled for clinical processing. No remote model inference API was used.
- No synthetic data is used by the final retrieval track. Earlier partial Qwen generation and stopped LoRA training are documented as abandoned measured experiments; they are not completed model components.
- Native Kaggle runtime used Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, numpy 2.1.3, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.6.1, pyarrow 23.0.1, huggingface-hub 1.29.0 and safetensors 0.8.0. Preserve the final resolved runtime lock with model artifacts.

## Demo and submission status

Run python -m genlink.demo_server --artifacts artifacts/runtime and open http://127.0.0.1:8765. The demo uses the real local engine.

The submission form is prepared but must not be sent without the team POC's final confirmation. The repository remains private until submission, then must be made public for the organizers. No hidden evaluation file was supplied.
