import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dictionary to map number words to digits
number_words_to_digits = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

# Function to replace number words with digits
def replace_number_words(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    # Replace each number word with its corresponding digit
    for word, digit in number_words_to_digits.items():
        text = re.sub(rf'\b{word}\b', digit, text, flags=re.IGNORECASE)
    return text

# Function to extract numbers from a given text based on the rules provided
def extract_numbers(text):
    # If the input is NaN or not a string, return an empty set
    if pd.isna(text) or not isinstance(text, str):
        return set()
    
    # This regex captures numbers like 5, 5.5, 42,000 (treated as 42000), but splits phone-like numbers
    number_pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+"
    
    # Find all numbers in the text
    numbers = re.findall(number_pattern, text)
    
    # Clean and convert numbers to float/int, remove commas for large numbers
    processed_numbers = set()
    for number in numbers:
        # Remove commas from numbers like '42,000'
        clean_number = number.replace(',', '')
        
        # Convert the number to float or int depending on its format
        if '.' in clean_number:
            processed_numbers.add(float(clean_number))
        else:
            processed_numbers.add(int(clean_number))
    
    return processed_numbers

# Function to calculate Average Mismatch Error
def calculate_avg_mismatch_error(numbers_in_answer, numbers_in_response):
    if not numbers_in_answer:
        return '#'
    
    avg_mismatch_error = 0
    for number in numbers_in_answer:
        if number not in numbers_in_response:
            avg_mismatch_error += 1
    
    avg_mismatch_error /= len(numbers_in_answer)
    return avg_mismatch_error

# Function to calculate Exact Mismatch Error
def calculate_exact_mismatch_error(numbers_in_answer, numbers_in_response):
    if not numbers_in_answer:
        return '#'
    
    for number in numbers_in_answer:
        if number not in numbers_in_response:
            return 1
    return 0

# Load the Excel files
#Evaluation of RAG from scratch pipeline
#gold_testset_path = './comet/tabular_TESTSET_FINAL.xlsx'
gold_testset_path = './textual_TESTSET_FINAL_gold.xlsx'
#comet_evaluation_path = './comet/textual_COMET_evaluation.xlsx'
#gold_testset_path = './comet/textual_TESTSET_FINAL.xlsx'
#comet_evaluation_path = './comet/tabular_COMET_evaluation.xlsx'
#comet_evaluation_path = './comet/textual_COMET_openai.xlsx'
#comet_evaluation_path = './comet/tabular_COMET_openai.xlsx'
#comet_evaluation_path = './comet/textual_COMET_gemma_better.xlsx'
#comet_evaluation_path = './comet/textual_gold_testset_openai_no_as_far_as.xlsx'
#comet_evaluation_path = './comet/textual_gold_testset_DPR_openai.xlsx'
#comet_evaluation_path = './comet/tabular_testset_DPR_openai.xlsx'
#comet_evaluation_path = './comet/textual_gold_COMET_openai_no_as_far_as.xlsx'
#comet_evaluation_path = './comet/tabular_COMET_openai_no_as_far_as.xlsx'
comet_evaluation_path = './comet/textual_gold_COMET_DPR_MS_openai_no_as_far_as.xlsx'
#comet_evaluation_path = './comet/tabular_COMET_DPR_MS_openai_no_as_far_as.xlsx'



#comet_evaluation_path = './comet/tabular_COMET_lamini.xlsx'
#comet_evaluation_path = './comet/textual_COMET_lamini.xlsx'
print(gold_testset_path, comet_evaluation_path)

gold_df = pd.read_excel(gold_testset_path)
comet_df = pd.read_excel(comet_evaluation_path)

# Add new columns for average and exact mismatch errors if they don't exist
if 'Average Mismatch Error' not in comet_df.columns:
    comet_df['Average Mismatch Error'] = None

if 'Exact Mismatch Error' not in comet_df.columns:
    comet_df['Exact Mismatch Error'] = None

#bad_retrieval_texts = ["does not","doesn't contain", "doesn't specify", "doesn't provide", "cannot answer"]

bad_retrieval_texts = [
    "does not contain", "does not specify", "does not provide", "doesn't contain",
    "doesn't specify", "doesn't provide", "cannot answer", "does not mention",
    "doesn't mention", "cannot determine", "does not explicitly",
    "doesn't explicitly", "does not answer", "doesn't answer", "does not specifically",
    "doesn't specifically", "does not include", "doesn't include", "not explicitly",
    "not specify", "not provide", "not contain", "not mention", "not answer","not specifically","cannot provide","I'm sorry"]

# Iterate over the COMET evaluation DataFrame
for index, row in comet_df.iterrows():
    src_ans_en = row['Src-Ans-en']
    #src_ans_en = row['Textual Responses']
    #src_ans_en = row['Tabular Responses']
    
    if any(bad_text in src_ans_en.lower() for bad_text in bad_retrieval_texts):
    	comet_df.at[index, 'Average Mismatch Error'] = '#'
    	comet_df.at[index, 'Exact Mismatch Error'] = '#'
    	continue;
    
    '''
    if src_ans_en == "As far as my knowledge, the handbook doesn't contain an answer to this query.":
    		comet_df.at[index, 'Average Mismatch Error'] = '#'
    		comet_df.at[index, 'Exact Mismatch Error'] = '#'
    		continue'''
    
    # Preprocess the response to replace numeric words with digits
    src_ans_en = replace_number_words(src_ans_en)
    
    # Fetch the corresponding question and answer from the gold testset
    matching_row = gold_df.loc[index]
    gold_answer = matching_row['answer']
    question = matching_row['question']

    # Step 1: Extract numbers from the gold answer
    numbers_in_answer = extract_numbers(gold_answer)
    
    # Step 2: Extract numbers from the question
    numbers_in_question = extract_numbers(question)

    # Step 3: Remove numbers from the answer that are present in the question
    numbers_in_answer.difference_update(numbers_in_question)

    # Step 4: Extract numbers from the response (Src-Ans-en)
    numbers_in_response = extract_numbers(src_ans_en)

    # Step 5: Calculate Average Mismatch Error and Exact Mismatch Error
    avg_mismatch_error = calculate_avg_mismatch_error(numbers_in_answer, numbers_in_response)
    exact_mismatch_error = calculate_exact_mismatch_error(numbers_in_answer, numbers_in_response)

    # Store the results in the COMET DataFrame
    comet_df.at[index, 'Average Mismatch Error'] = avg_mismatch_error
    comet_df.at[index, 'Exact Mismatch Error'] = exact_mismatch_error

# Save the updated comet_df back to the Excel file
comet_df.to_excel(comet_evaluation_path, index=False)

print("Average and Exact Mismatch Errors calculated and stored successfully.")

# Function to process numeric values in a column (ignoring '#')
def process_numeric_column(column_data):
    # Collect all non-# values (convert to numeric, coerce invalid to NaN, drop NaN)
    numeric_values = pd.to_numeric(column_data, errors='coerce').dropna()

    # Compute statistics
    mean_value = np.mean(numeric_values)
    median_value = np.median(numeric_values)
    std_value = np.std(numeric_values)

    return numeric_values, mean_value, median_value, std_value

# Function to plot histogram for a list of values
def plot_histogram(data, title, xlabel):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=10, color='blue', edgecolor='black')
    plt.title(f"Histogram of {title}")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# Process "Average Mismatch Error" column
avg_mismatch_values, avg_mean, avg_median, avg_std = process_numeric_column(comet_df['Average Mismatch Error'])

# Print statistics for "Average Mismatch Error"
print(f"Average Mismatch Error - Mean: {avg_mean}, Median: {avg_median}, Std: {avg_std}")

# Plot histogram for "Average Mismatch Error"
plot_histogram(avg_mismatch_values, "Average Mismatch Error", "Average Mismatch Error")

# Process "Exact Mismatch Error" column
exact_mismatch_values, exact_mean, exact_median, exact_std = process_numeric_column(comet_df['Exact Mismatch Error'])

# Print statistics for "Exact Mismatch Error"
print(f"Exact Mismatch Error - Mean: {exact_mean}, Median: {exact_median}, Std: {exact_std}")

# Plot histogram for "Exact Mismatch Error"
plot_histogram(exact_mismatch_values, "Exact Mismatch Error", "Exact Mismatch Error")

