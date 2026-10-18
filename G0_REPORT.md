# G0 full-data verification

Overall: PASS

| Check | Actual | Expected | Pass |
|---|---|---|---|
| icd_dictionary_rows | 159195 | 159195 | True |
| icd_codes | 98182 | 98182 | True |
| snomed_dictionary_rows | 984585 | 984585 | True |
| snomed_codes | 382124 | 382124 | True |
| input_rows | 9646 | 9646 | True |
| systems | ['ICD10CM', 'SNOMEDCT_US'] | ['ICD10CM', 'SNOMEDCT_US'] | True |
| invalid_rows | [{'coding_system': 'ICD10CM', 'code': 'L6.0'}, {'coding_system': 'ICD10CM', 'code': 'L6.0'}] | [{'coding_system': 'ICD10CM', 'code': 'L6.0'}, {'coding_system': 'ICD10CM', 'code': 'L6.0'}] | True |
| retained_rows | 9644 | 9644 | True |
| entity_hash | 845350f70cad58ab | 845350f70cad58ab | True |
| fold_file_hash | 93002cc6069f3464 | 93002cc6069f3464 | True |
| fold_rows | 9644 | 9644 | True |
| fold_row_ids | all row ids checked | 0–9643 | True |
| fold_system_alignment | True | True | True |
| fold_values | [0, 1, 2] | [0, 1, 2] | True |
| ICD10CM_fold_counts | [817, 749, 738] | [817, 749, 738] | True |
| SNOMEDCT_US_fold_counts | [2398, 2466, 2476] | [2398, 2466, 2476] | True |
| duplicate_instance_system | 0 | 0 | True |
| instances | 7522 | 7522 | True |
| maximum_folds_per_instance | 1 | 1 | True |
| both_systems | 2122 | 2122 | True |
| snomed_only | 5218 | 5218 | True |
| icd_only | 182 | 182 | True |
| snomed_preferred_unique | True | True | True |
| snomed_preferred_code_coverage | 382124 | 382124 | True |
| normalization_tests | True | True | True |

## Preferred display match rates
- icd_position_0: 63.1944%
- icd_position_1: 17.9253%
- snomed_preferred: 92.0163%

## Next phase requirements
- G1 onward not run. No GPU jobs or uploads launched.
- Required ml-pipeline:ml-pipeline skill was not found in installed skill folders. Resolve before training.
- Kaggle authentication, private destination, available GPU quota, model licences and token/throughput compute plan remain unverified.
- The supplied prompt requires a review stop after G0.
