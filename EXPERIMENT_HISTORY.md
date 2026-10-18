# Measured experiments and architecture decision

Earlier experiments are preserved privately in the workspace and Kaggle notebook history. They are not the active predictor.

- Complete G0 audit and G1 dictionary constraints were built on all retained rows and concepts.
- Full synthetic generation was projected at 47.35 hours. Batching improvements reduced the later forecast to 37.66 hours, still beyond the available deadline; partial outputs were preserved and excluded from the final method.
- Direct Qwen3-1.7B supervised LoRA training was measured with real optimizer updates. The optimized implementation reduced update time by 47.3%, but a single five-epoch selection job still forecast 6.18-6.37 hours before the remaining folds/refits/decoding. Both profiles stopped at their compute gates and retained incomplete checkpoints.
- The user then authorized frozen biomedical retrieval plus lightweight context ranking, with the original fixed folds, complete data scope, offline processing, source/resource locks and measured compute gates retained. The unavailable ml-pipeline skill requirement was explicitly waived by the user.

No stopped adapter is represented as a completed model, and no synthetic examples are used by the active retrieval architecture. The current validation report is the authority for accuracy.
