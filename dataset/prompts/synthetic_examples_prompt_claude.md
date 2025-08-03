# Synthetic Backchannel Data Generation Prompt

## 20 Conversation Scenarios

1. **Twitch Gaming Stream** - Streamer + viewers, fast-paced gaming commentary
2. **Discord Friend Group** - Casual friends chatting while gaming/hanging out
3. **TikTok Live Stream** - Content creator interacting with live audience
4. **Reddit Voice Chat** - Topic-focused discussion, informal but structured
5. **NPR Interview** - Professional journalist interviewing expert
6. **Business Zoom Meeting** - Corporate team discussing projects/strategy
7. **Medical Consultation** - Doctor explaining diagnosis/treatment to patient
8. **Legal Consultation** - Lawyer advising client on legal matters
9. **Podcast Interview** - Host interviewing guest, conversational flow
10. **Late Night Talk Show** - Host chatting with celebrity guest
11. **Sports Commentary** - Commentators discussing live game action
12. **Customer Service Call** - Agent helping customer with problem
13. **Therapy Session** - Therapist providing support to client
14. **Tech Support Call** - Support agent walking through troubleshooting
15. **Family Dinner** - Multi-generational family catching up
16. **First Date** - Two people getting to know each other
17. **Friend Gossip Session** - Close friends sharing personal stories
18. **Cooking Show** - Chef teaching recipe to audience/co-host
19. **Online Tutoring** - Teacher explaining concept to student
20. **Real Estate Showing** - Agent describing property features to buyers

## Target Backchannel Words

```
["yeah", "yes", "uh-huh", "mhmm", "mm-hmm", "hmm", "oh", "ah", "uhhuh", "uh", "um", "mmmm", "yep", "wow", "right", "okay", "ok", "sure", "alright", "gotcha", "mmhmm", "great", "sweet", "ma'am", "awesome", "good morning", "i see", "got it", "that makes sense", "i hear you", "i understand", "good afternoon", "hey there", "perfect", "that's true", "good point", "exactly", "makes sense", "no problem", "indeed", "certainly", "very well", "absolutely", "correct", "of course", "k", "hey", "hello", "hi", "yo"]
```

## Task Instructions

Generate synthetic conversation data where each example contains:
1. **Previous utterance** (what the other person said)
2. **Current utterance** (the response containing the backchannel word)
3. **Label** (1 = backchannel, 0 = not backchannel)

For each of the 20 scenarios, select **realistic backchannel words** that would naturally occur in that context and create **2 examples per word**:
- **1 POSITIVE example** (label=1): The word is used as a genuine backchannel (listening cue, brief acknowledgment)
- **1 NEGATIVE example** (label=0): The word is used for turn-taking, disagreeing, or extending the conversation

**Important**: You don't need to use all 50 words in every scenario. Some backchannel words won't naturally fit certain contexts:
- Gaming streams might use "yeah", "yo", "sweet", "right", "okay" but probably not "certainly" or "good afternoon"
- Medical consultations might use "I understand", "mm-hmm", "okay", "I see", "yes" but not "yo" or "sweet"
- NPR interviews might use "absolutely", "indeed", "certainly", "mm-hmm", "I see" but not "yo" or "sweet"
- Legal consultations might use "I understand", "certainly", "yes", "absolutely" but not "yo", "sweet", or "awesome"
- Family dinners might use "yeah", "oh", "really", "awesome", "great" but might not use "indeed" or "certainly"

**Aim for 15-25 realistic words per scenario** rather than forcing unnatural usage. Quality and authenticity are more important than hitting exact numbers.

## Example Format

Here are examples showing the difference:

### POSITIVE Examples (label=1) - True Backchannels
```
Scenario: NPR Interview
Previous: "The economic data shows inflation decreased by 0.2% last month due to energy cost reductions."
Current: "mm-hmm"
Label: 1

Scenario: Medical Consultation  
Previous: "The test results show your cholesterol levels have improved significantly since your last visit."
Current: "that's great"
Label: 1

Scenario: Twitch Gaming Stream
Previous: "Chat, I'm about to attempt this insane speedrun trick that took me weeks to master."
Current: "yo"
Label: 1
```

### NEGATIVE Examples (label=0) - Turn-Taking/Not Backchannels
```
Scenario: NPR Interview
Previous: "The economic data shows inflation decreased by 0.2% last month due to energy cost reductions."
Current: "mm-hmm, but I think we need to be cautious about reading too much into a single month's data."
Label: 0

Scenario: Medical Consultation
Previous: "The test results show your cholesterol levels have improved significantly since your last visit."
Current: "that's great, but I wanted to ask about these side effects I've been experiencing."
Label: 0

Scenario: Twitch Gaming Stream
Previous: "Chat, I'm about to attempt this insane speedrun trick that took me weeks to master."
Current: "yo, you should try the wall-jump method instead, it's way faster."
Label: 0
```

## Key Patterns for Negative Examples

Make sure negative examples capture these common patterns:
- **"[backchannel word] + but/however"** - Disagreement or contrast
- **"[backchannel word] + so/and"** - Continuing the conversation 
- **"[backchannel word] + question"** - Taking conversational floor
- **"[backchannel word] + new information"** - Adding content
- **Longer responses** that happen to contain the backchannel word

## Output Format

Please generate the data in CSV format with these exact column headers:
```
scenario,previous_utterance,current_utterance,label
```

**Important Requirements:**
1. Generate realistic examples that fit each scenario's conversational style
2. Use 15-25 backchannel words per scenario (skip words that don't fit naturally)
3. Create exactly 2 examples per word you choose (1 positive, 1 negative)
4. Make conversations feel natural and realistic for each scenario
5. Vary the length and style appropriate to each context
6. Include modern slang/language patterns for digital contexts
7. Ensure clear distinction between backchannel (label=1) and turn-taking (label=0) usage
8. Use proper CSV escaping for any commas or quotes in the text

**Expected output**: Approximately 800-1,000 total examples (20 scenarios × ~20 words average × 2 examples)