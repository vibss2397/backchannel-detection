from convokit import Corpus, download
import re
import pandas as pd

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
                          ['previous_utterance', 'current_utterance', 'label'].
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
    total_conversations = len(corpus.get_conversation_ids())

    if subset_size is not None and subset_size > 0:
        print(f"Processing a subset of {subset_size} conversations...")
        conversation_iterator = islice(conversation_iterator, subset_size)
        total_conversations = subset_size
    else:
        print("Processing all conversations...")

    for convo in tqdm(conversation_iterator, total=total_conversations):
        try:
            all_utts = list(convo.iter_utterances(selector=lambda u: u.text != ''))
            utts = sorted(all_utts, key=lambda u: int(u.id.split('-')[-1]))
        except (ValueError, IndexError):
            continue

        for i in range(1, len(utts)):
            current_utt = utts[i]
            previous_utt = utts[i-1]

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

            data_points.append({
                'previous_utterance': previous_utt.text,
                'current_utterance': current_utt.text,
                'label': label
            })

    print(f"\nProcessed {len(data_points)} valid data points.")
    return pd.DataFrame(data_points)

def clean_switchboard_text(text):
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


if __name__ == "__main__":
    print("Loading Switchboard corpus...")
    try:
        # Download corpus (this might take a few minutes)
        corpus = Corpus(download('switchboard-corpus'))
    except Exception as e:
        print(f"Failed to download or load corpus. Error: {e}")
        print("Please check your internet connection and try again.")
        exit(0)
    print("Corpus loaded successfully.")

    df = create_backchannel_dataset(corpus)
    # Error in creating df
    if not df or df.empty:
        print('DF could not be processed, check logs...')
        exit(0)

    print("\n--- Dataset Creation Complete ---")
    print(f"Total samples created: {len(df)}")

    print("\nLabel Distribution:")
    print(df['label'].value_counts(normalize=True))

    print("\n--- Sample Data Points ---")
    print("\nExamples of Back-channels (Label = 1):")
    print(df[df['label'] == 1].head())
    print("\nExamples of Normal Utterances (Label = 0):")
    print(df[df['label'] == 0].head())
    
    df['previous_utter_clean'] = df['previous_utterance'].map(lambda x: clean_switchboard_text(x))
    df['current_utter_clean'] = df['current_utterance'].map(lambda x: clean_switchboard_text(x))

    df.to_csv('dataset.csv')
    print('successfully created dataset.')

