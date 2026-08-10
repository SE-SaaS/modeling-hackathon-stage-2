import os
import pandas as pd
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import re

# ==========================================
# CONFIGURATION
# ==========================================

# The Modal Endpoint URL from your Quickstart Guide
MODAL_ENDPOINT_URL = "https://dyiaalqaisi99--ep-gemma-4-31b-it-server.us-west.modal.direct/v1"

# The model name you selected
MODEL_NAME = "google/gemma-4-31B-it"

# The credentials from your first image
MODAL_TOKEN = "wk-dqqmXZWb6ftx3w53Z9prex.ws-jLU4iKwl8cjUUzOvj3eNEG"

# Number of concurrent requests to make (adjust based on endpoint capacity)
CONCURRENCY = 20  

# ==========================================

client = AsyncOpenAI(
    base_url=MODAL_ENDPOINT_URL,
    api_key=MODAL_TOKEN,
)

def parse_response(response_text: str) -> int:
    """Extract the integer class from the model's response."""
    # Look for a number (0, 1, 2, or 3) in the text
    match = re.search(r'[0-3]', response_text)
    if match:
        return int(match.group())
    return 0  # Fallback to 0 (Syrian) if we can't parse it

async def classify_sentence(sem, sentence, row_id):
    prompt = (
        "You are an expert in Levantine Arabic dialects. Classify the dialect of the following sentence "
        "into one of these four categories: Syrian (0), Lebanese (1), Palestinian (2), or Jordanian (3).\n"
        "Respond ONLY with the single digit (0, 1, 2, or 3) representing the dialect, and nothing else.\n\n"
        f"Sentence: {sentence}"
    )
    
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that only outputs single digits."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Greedy decoding for classification
                max_tokens=5,    # We only need one digit
            )
            result = response.choices[0].message.content.strip()
            label = parse_response(result)
            return row_id, label
        except Exception as e:
            print(f"Error on row {row_id}: {e}")
            return row_id, 0 # Fallback

async def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, 'datasets', 'test.csv')
    output_path = os.path.join(base_dir, 'output', 'submission_api.csv')
    
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    test_df['text'] = test_df['text'].fillna('')
    
    print(f"Starting inference on {len(test_df)} rows...")
    sem = asyncio.Semaphore(CONCURRENCY)
    
    tasks = []
    for _, row in test_df.iterrows():
        tasks.append(classify_sentence(sem, row['text'], row['id']))
    
    # Run tasks with a progress bar
    results = await tqdm.gather(*tasks, desc="Classifying")
    
    # Sort results to maintain original order (just in case)
    results.sort(key=lambda x: x[0])
    ids = [r[0] for r in results]
    preds = [r[1] for r in results]
    
    # Save submission
    sub = pd.DataFrame({'id': ids, 'dialect': preds})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub.to_csv(output_path, index=False)
    
    print(f"Done! Saved submission to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
