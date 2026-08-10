# Goal Description

The objective is to achieve the highest possible score on Hackathon Challenge 3 (Arabic Dialect Identification). You have restructured the project directory into:
- `code/` (contains `solve_baseline.py` and `solve_dl.py`)
- `datasets/` (contains `train.csv` and `test.csv`)
- `instruction/` (contains problem rules and descriptions)
- `output/` (contains submissions and plans)

We need to refactor the Python scripts to be robust against this new structure, improve error handling, and ensure the pipeline is solid for the hackathon.

## User Review Required

Please review the proposed structural updates to the code below. 

## Open Questions

1. **CUDA Setup:** The previous plan noted your PyTorch installation is CPU-only. Are we still focusing exclusively on the `solve_baseline.py` for now, or should I also refactor `solve_dl.py` to be robust and ready for when you install CUDA?

## Proposed Changes

### Refactoring `code/solve_baseline.py` and `code/solve_dl.py`

I will make the following robustness improvements:

1. **Robust Path Resolution:** 
   Instead of assuming the scripts are run from the `problem3` root, I will use `os.path.dirname(__file__)` to dynamically resolve absolute paths.
   - Datasets will be loaded from `../datasets/`
   - Submissions will be saved to `../output/`
   
2. **Handling Missing `sample_submission.csv`:**
   The `solve_baseline.py` currently tries to read `sample_submission.csv` from the current directory. To make it strictly robust, I will generate the submission DataFrame dynamically using the `id` column from `test.csv` (e.g., `pd.DataFrame({'id': test_df['id'], 'dialect': test_preds})`).

3. **Data Validation and Error Handling:**
   - Wrap data loading in `try...except FileNotFoundError` blocks to provide clear error messages if datasets are missing.
   - Verify that `train.csv` and `test.csv` contain the necessary columns before proceeding.

4. **Structured Logging:**
   Replace basic `print()` statements with Python's standard `logging` module to provide timestamps and severity levels (INFO, WARNING, ERROR).

---

#### [MODIFY] [solve_baseline.py](file:///e:/modeling-hackathon-stage-2/problem3/code/solve_baseline.py)
Implement dynamic paths (`../datasets/train.csv`, `../output/submission_baseline.csv`), add structured logging, handle missing files gracefully, and dynamically generate the submission CSV format.

#### [MODIFY] [solve_dl.py](file:///e:/modeling-hackathon-stage-2/problem3/code/solve_dl.py)
Implement the same dynamic paths and logging improvements as above, while gracefully warning if CUDA is unavailable.

## Verification Plan

### Automated Tests
- Run `python code/solve_baseline.py` to verify that it reads from `datasets/`, processes correctly, and outputs `submission_baseline.csv` into the `output/` folder without any "file not found" errors.

### Manual Verification
- Verify the generated submission CSV matches the exact Kaggle format.
