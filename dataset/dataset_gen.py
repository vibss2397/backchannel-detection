from convokit import Corpus, download
import re
import pandas as pd
import string
from itertools import islice
from tqdm import tqdm

def create_backchannel_dataset(corpus, subset_size=None):
    """
    Creates a labeled dataset of back-channel vs. normal utterances from the 
    Switchboard corpus, applying a "safety-first" logic for voice agent applications.

    Args:
        subset_size (int, optional): The number of conversations to process. 
                                     If None, all conversations are processed. 
                                     Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame with columns 
                          ['scenario', 'previous_utterance', 'current_utterance', 'label'].
    """

    # Pure Backchannels
    BACKCHANNEL_TAGS = {
        'b',    # "uh-huh, right, yeah" - classic back-channels
        'bk',   # "Oh, okay" - acknowledging without adding content
        'ba',   # "I can imagine" - appreciating but not taking floor
        'by',   # "I'm sorry to hear that" - sympathy without interrupting
    }

    # Pure not-backchannels
    SUBSTANTIVE_TAGS = {
        'qy', 'qw', 'qo', 'qh', 'qr',  # Questions
        'sd', 'sv',                     # Statements/narratives
        'ar', 'arp',                    # Rejections
        'ad',                           # Action directives
        'na', 'ng', 'nn', 'ny', 'no',  # Answers
        'co',                           # Offers
        'fc', 'fp'                      # Conventional open/close
    }

    # Tags to be filtered out due to data quality issues.
    EXCLUDED_TAGS = {'%', 'x', 't1', 't3', '@', 'o@', '+@'}

    data_points = []
    
    conversation_iterator = corpus.iter_conversations()
    
    if subset_size is not None and subset_size > 0:
        print(f"Processing a subset of {subset_size} conversations...")
        # Use islice to take a subset of conversations
        conversation_iterator = islice(conversation_iterator, subset_size)
        progress_total = subset_size
    else:
        print("Processing all conversations...")
        progress_total = len(corpus.get_conversation_ids())

    total_added = 0
    for convo in tqdm(conversation_iterator, total=progress_total):
        if total_added >= progress_total:
            break
        try:
            # The rest of your processing logic remains the same
            all_utts = list(convo.iter_utterances(selector=lambda u: u.text != ''))
            utts = sorted(all_utts, key=lambda u: int(u.id.split('-')[-1]))
        except (ValueError, IndexError):
            continue

        for i in range(1, len(utts)):
            current_utt = utts[i]
            previous_utt = utts[i - 1]
            if current_utt.speaker == previous_utt.speaker:
                continue

            tags_in_utterance = {segment[1] for segment in current_utt.meta.get('tag', [])}
            
            # 1. Filter First: Check for any excluded tags.
            if any(tag in EXCLUDED_TAGS for tag in tags_in_utterance):
                continue

            # --- New "Safety-First" Labeling Logic ---
            label = None
            
            # Rule 1: If ANY tag is clearly substantive, it is NOT a back-channel.
            if any(tag in SUBSTANTIVE_TAGS for tag in tags_in_utterance):
                label = 0
            # Rule 2: If not substantive, check if it's a clear back-channel.
            elif any(tag in BACKCHANNEL_TAGS for tag in tags_in_utterance):
                label = 1
            # Rule 3: If it's in the gray area (neither), default to NOT back-channel for safety.
            else:
                label = 0

            total_added += 1
            data_points.append({
                'scenario': 'Switchboard',
                'previous_utterance': previous_utt.text,
                'current_utterance': current_utt.text,
                'label': label
            })

    print(f"\nProcessed {len(data_points)} valid data points from Switchboard.")
    return pd.DataFrame(data_points)

def clean_dataset_text(text):
    """
    Cleans and normalizes text from the Switchboard corpus.
    """
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove transcription artifacts
    # Remove content in curly braces e.g., {F uh, }
    text = re.sub(r'\{.*?\}', '', text)
    # Remove content in angle brackets e.g., <beep>
    text = re.sub(r'<.*?>', '', text)
    # Remove square brackets, plus signs, dashes, and slashes
    text = re.sub(r'[\[\]\+\-\/]', ' ', text)
    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_synthetic_data(csv_file_path):
    """
    Load and process the synthetic backchannel data.
    
    Args:
        csv_file_path (str): Path to the synthetic data CSV file
        
    Returns:
        pandas.DataFrame: Processed synthetic data
    """
    print(f"Loading synthetic data from {csv_file_path}...")
    try:
        synthetic_df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(synthetic_df)} synthetic data points.")
        return synthetic_df
    except Exception as e:
        print(f"Failed to load synthetic data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Load Switchboard data
    print("Loading Switchboard corpus...")
    try:
        # Download corpus (this might take a few minutes)
        corpus = Corpus(download('switchboard-corpus'))
    except Exception as e:
        print(f"Failed to download or load corpus. Error: {e}")
        print("Please check your internet connection and try again.")
        exit(0)
    print("Corpus loaded successfully.")

    # Process 1000 conversations from Switchboard
    switchboard_df = create_backchannel_dataset(corpus, subset_size=5000)
    
    # Error in creating switchboard df
    if switchboard_df.empty:
        print('Switchboard DF could not be processed, check logs...')
        exit(0)

    # Load synthetic data (update this path to your CSV file)
    synthetic_claude_csv_path = "synthetic_dataset_claude.csv"  # Update this path
    synthetic_claude_df = load_synthetic_data(synthetic_claude_csv_path)

    synthetic_gemini_csv_path = "synthetic_dataset_gemini.csv"  # Update this path
    synthetic_gemini_df = load_synthetic_data(synthetic_gemini_csv_path)

    # Combine datasets
    all_datasets = [switchboard_df]
    if not synthetic_claude_df.empty:
        print(f"Combining {len(switchboard_df)} Switchboard samples with {len(synthetic_claude_df)} synthetic samples...")
        all_datasets.append(synthetic_claude_df)
    if not synthetic_gemini_df.empty:
        print(f"Combining {len(switchboard_df)} Switchboard samples with {len(synthetic_gemini_df)} synthetic Gemini samples...")
        all_datasets.append(synthetic_gemini_df)
    if len(all_datasets) == len(switchboard_df):
        print("No synthetic data to combine, only Switchboard data will be used.")
    combined_df = pd.concat(all_datasets, ignore_index=True)

    print("\n--- Dataset Creation Complete ---")
    print(f"Total samples created: {len(combined_df)}")

    # Show dataset composition
    print("\nDataset Composition by Scenario:")
    print(combined_df['scenario'].value_counts())

    print("\nLabel Distribution:")
    print(combined_df['label'].value_counts(normalize=True))

    print("\n--- Sample Data Points ---")
    print("\nExamples of Back-channels (Label = 1):")
    print(combined_df[combined_df['label'] == 1].head())
    print("\nExamples of Normal Utterances (Label = 0):")
    print(combined_df[combined_df['label'] == 0].head())
    
    # Apply cleaning to all text
    print("\nCleaning text data...")
    combined_df['previous_utter_clean'] = combined_df['previous_utterance'].map(lambda x: clean_dataset_text(x))
    combined_df['current_utter_clean'] = combined_df['current_utterance'].map(lambda x: clean_dataset_text(x))

    # Save final dataset
    output_path = 'backchannel_dataset_cleaned.csv'
    combined_df.to_csv(output_path, index=False)
    print(f'Successfully created combined dataset: {output_path}')
    
    # Final statistics
    print("\nFinal Dataset Statistics:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Backchannels (label=1): {len(combined_df[combined_df['label'] == 1])}")
    print(f"Non-backchannels (label=0): {len(combined_df[combined_df['label'] == 0])}")
    print(f"Balance ratio: {combined_df['label'].mean():.3f}")