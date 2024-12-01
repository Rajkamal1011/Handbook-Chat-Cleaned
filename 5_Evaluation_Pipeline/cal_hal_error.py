import pandas as pd

# File path
file_path = "/home/rajkamal/Desktop/New_ERA/NLP/RAG/Evaluation/comet/vague_testset_DPR_MS_openai_no_as_far_as.xlsx"

# Load the Excel file
df = pd.read_excel(file_path)

#df = df.head(300)

# Bad retrieval phrases
# Define the list of "Bad Retrieval" texts
bad_retrieval_texts = [
    "does not contain", "does not specify", "does not provide", "doesn't contain",
    "doesn't specify", "doesn't provide", "cannot answer", "does not mention",
    "doesn't mention", "cannot determine", "does not explicitly",
    "doesn't explicitly", "does not answer", "doesn't answer", "does not specifically",
    "doesn't specifically", "does not include", "doesn't include", "not explicitly",
    "not specify", "not provide", "not contain", "not mention", "not answer","not specifically","cannot provide","I'm sorry",
    "no mention","no indication","no specific","no specific mention",
    "no direct mention"]
    
#Added - no mention, no specific, no specific mention, no indication for Pipeline 3 and 4, since, openai was using this phrases a lot 

# Filter rows that do not contain any of the bad retrieval texts
#filtered_df = df[~df['Src-Ans-en'].str.contains('|'.join(bad_retrieval_texts), case=False, na=False)]
#filtered_df = df[~df['Textual Responses'].str.contains('|'.join(bad_retrieval_texts), case=False, na=False)]
filtered_df = df[~df['Textual Responses'].str.contains('|'.join(bad_retrieval_texts), case=False, na=False)]
#filtered_df = df[~df['response'].str.contains('|'.join(bad_retrieval_texts), case=False, na=False)]

# Calculate the average COMET score
#avg_comet_score = filtered_df['COMET Scores'].mean()
#median_comet_score = filtered_df['COMET Scores'].median()
#std_dev_comet_score = filtered_df['COMET Scores'].std()

# Calculate total entries and bad retrieval error
total_entries = len(df)
valid_entries = len(filtered_df)
bad_retrieval_error = (total_entries - valid_entries) / total_entries
hallucination_error = 1 - bad_retrieval_error

# Print the results
#print(f"Avg COMET Score: {avg_comet_score}")
#print(f"Median COMET Score: {median_comet_score}")
#print(f"Standard Deviation of COMET Score: {std_dev_comet_score}")
print(f"x = {valid_entries} (Count of total entries that don't have bad retrieval texts)")
#print(f"BAD RETRIEVAL ERROR = {bad_retrieval_error:.2%}")
print(f"HALLUCINATION ERROR = {hallucination_error:.2%}")


