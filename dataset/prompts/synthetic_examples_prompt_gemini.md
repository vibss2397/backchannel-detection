
Conversation with Gemini
ROLE: You are an expert data generator for machine learning. Your task is to create realistic, two-line conversational snippets for a back-channel detection model.

GOAL: Generate a CSV-formatted dataset with three columns: previous_utterance, current_utterance, and label.

DEFINITIONS:



Back-channel (Label = 1): A short, simple utterance that signals "I'm listening" without taking over the conversation. Examples: "uh-huh," "I see," "right."

Substantive Utterance (Label = 0): An utterance that takes the conversational floor by adding a new opinion, question, or piece of information.

RULES:



Strict CSV Format: The output must be in CSV format with a header row: previous_utterance,current_utterance,label.

One Positive, One Negative: For each "Back-channel Term," you must generate one positive example (label 1) and one "hard negative" example (label 0).

Natural Dialogue is Key: Use your best judgment. If a specific back-channel term (e.g., 'good morning') feels unnatural or out of place for a given scenario (e.g., a late-night gaming session), skip that combination and move on. Prioritize realistic, believable dialogue over generating an entry for every single combination.

TASK:

For each of the 50 "Back-channel Terms" provided below, generate your two examples (one positive, one negative), cycling through the 20 "Diverse Scenarios" to ensure the conversational style is varied and modern.

BACK-CHANNEL TERMS TO USE (50 total):"yeah", "yes", "uh-huh", "mhmm", "mm-hmm", "hmm", "oh", "ah", "uhhuh", "uh", "um", "mmmm", "yep", "wow", "right", "okay", "ok", "sure", "alright", "gotcha", "mmhmm", "great", "sweet", "ma'am", "awesome", "good morning", "i see", "got it", "that makes sense", "i hear you", "i understand", "good afternoon", "hey there", "perfect", "that's true", "good point", "exactly", "makes sense", "no problem", "indeed", "certainly", "very well", "absolutely", "correct", "of course", "k", "hey", "hello", "hi", "yo"

DIVERSE SCENARIOS TO CYCLE THROUGH (20 total):



Two friends on a Discord call planning a gaming session.

A tech podcast host interviewing a startup CEO.

A customer complaining to a support agent about a faulty product.

Two colleagues in a remote stand-up meeting discussing project blockers.

A food vlogger enthusiastically reviewing a new restaurant.

An NPR-style radio host conducting a calm interview with an author.

A parent trying to troubleshoot a WiFi issue with their tech-savvy teenager.

Two roommates debating what to order for dinner.

A fitness instructor leading a virtual workout class.

A financial advisor explaining investment options to a new client.

Two Gen Z friends gossiping about a recent social media trend.

A university professor holding virtual office hours with a student.

A DIY YouTuber explaining a complex step in a home renovation project.

A true-crime podcast hosts discussing a case theory.

A doctor explaining a diagnosis to a patient over a telehealth call.

Two people on a slightly awkward first date.

A manager giving performance feedback to an employee.

A travel agent helping a couple plan their honeymoon.

Two sports fans reacting live to a game they are watching.

A book club meeting where members are discussing the plot twist.

HIGH-QUALITY EXAMPLES OF THE FINAL OUTPUT:

Here is exactly how to structure your response.

Example for the term "yeah":

Code snippet



"So the final boss has a shield phase, you have to use the EMP grenade to drop it before you can do any damage.","yeah",1

"So the final boss has a shield phase, you have to use the EMP grenade to drop it before you can do any damage.","yeah that makes sense, I was just trying to brute force it.",0

Example for the term "wow":

Code snippet



"Our user growth last quarter was over 300% after the new feature launch.","wow",1

"Our user growth last quarter was over 300% after the new feature launch.","wow, that's an insane metric, was that all organic growth?",0

Example for the term "got it":

Code snippet



"You just need to submit your expense report through the portal by EOD Friday, and I'll approve it Monday morning.","got it",1

"You just need to submit your expense report through the portal by EOD Friday, and I'll approve it Monday morning.","got it, but I'm having trouble logging into the portal, can you reset my password?",0

Now, please begin generating the full CSV dataset based on these instructions. Start with the header row.