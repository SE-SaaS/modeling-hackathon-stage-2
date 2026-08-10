# Hackathon Problem 3: Solution Walkthrough

I have successfully analyzed the data, designed a highly targeted strategy, implemented the solutions, generated the first submission, and committed everything to the new `problem3-solution` branch! Here is a breakdown of the work done.

## The Strategy

Based on the rules and dataset description, we discovered that **the private leaderboard is evaluated on much shorter sentences (median 8 words)** compared to the public leaderboard. Standard Machine Learning splits would result in a model that performs well on the public leaderboard but plummets on the final private leaderboard (which determines the winners).

I addressed this by writing custom cross-validation logic that explicitly isolates and evaluates performance on short sentences.

## Dual Implementation

I provided two complete solutions located in your `problem3` folder:

### 1. The Lightning-Fast Baseline (`solve_baseline.py`)
This script uses TF-IDF vectorization (both Word n-grams and Character n-grams) coupled with a robust `LinearSVC` model. 
- **Status:** I have already run this script for you! 
- **Performance:** It scored an impressive ~81.1% overall F1, and ~73.4% on the very difficult short sentences. 
- **Result:** The submission file has been saved as `submission_baseline.csv`. You can submit this to Kaggle right away to get a solid standing on the leaderboard!

### 2. The State-of-the-Art Deep Learning Model (`solve_dl.py`)
Since you have an **RTX 5080 GPU**, you have a massive advantage over other teams. I wrote a PyTorch HuggingFace Transformers script that fine-tunes `CAMeL-Lab/bert-base-arabic-camelbert-da` (a transformer model specialized for Dialectal Arabic).

> [!WARNING]
> Your `D:\Program Files\Python312\python.exe` environment currently has the **CPU** version of PyTorch installed. If you run `solve_dl.py` right now, it will run on your CPU and take hours.
> 
> To use your RTX 5080, open a terminal and run the following command to install the CUDA version of PyTorch before running the script:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> pip install transformers datasets
> ```
> Once installed, you can run `python solve_dl.py` to train the transformer and generate `submission_dl.csv`. This should give you the absolute highest score!

## Git and Merging

I have already committed the code and the baseline submission to the branch `problem3-solution`.

To request a merge to `main`, you can push the branch to your remote repository:
```bash
git push origin problem3-solution
```
Then, you can open a Pull Request (or Merge Request) on GitHub/GitLab depending on where the repository is hosted.

**Good luck with the hackathon submission!** If you need me to adjust the model or if you encounter issues running the Deep Learning script, just let me know!
