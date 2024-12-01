import pandas as pd
from easygoogletranslate import EasyGoogleTranslate
from comet import download_model, load_from_checkpoint
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_float32_matmul_precision('medium')


# Step 1: Load data from Excel files
textual_responses_file = "tabular_testset_openai_no_as_far_as.xlsx"  # File with generated responses
textual_testset_file = "tabular_TESTSET_FINAL.xlsx"  # File with English questions and correct answers
output_excel_path = "tabular_COMET_openai_no_as_far_as.xlsx"

# Load the generated responses and the correct answers
df_responses = pd.read_excel(textual_responses_file)
df_correct = pd.read_excel(textual_testset_file)

# Extract relevant columns
source_ans_en = df_responses['Tabular Responses'].tolist()
#source_ans_en = df_responses['response'].tolist()  # Generated responses (English)
correct_ans_en = df_correct['answer'].tolist()  # Correct answers (English)

# Convert all responses to strings (this handles numbers correctly)
source_ans_en = [str(answer) for answer in source_ans_en]
correct_ans_en = [str(answer) for answer in correct_ans_en]

# Step 2: Translate using Google Translate API
translator = EasyGoogleTranslate(
    source_language='en',
    target_language='hi',
    timeout=10
)

# Translate the correct answers into Hindi (Reference Answer Hindi)
ref_ans_hindi = []
for sentence in correct_ans_en:
    translation = translator.translate(str(sentence))  # Ensure the input is a string
    ref_ans_hindi.append(translation)

# Translate the generated responses into Hindi (Translation Answer Hindi)
trans_ans_hindi = []
for sentence in source_ans_en:
    translation = translator.translate(str(sentence))  # Ensure the input is a string
    trans_ans_hindi.append(translation)

# Step 3: Create DataFrame with original English and translated Hindi answers
df_translation = pd.DataFrame({
    "Src-Ans-en": source_ans_en,  # Source answer (Generated response in English)
    "Ref-Ans-hin": ref_ans_hindi,  # Reference answer in Hindi
    "Trans-Ans-hin": trans_ans_hindi  # Translated generated answer in Hindi
})

# Write the translations to an Excel file
df_translation.to_excel(output_excel_path, index=False)
print("Hindi translations have been added to the Excel file.")

# Step 4: Evaluate translations with COMET
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

# Step 3: Move the model to cuda:1
model.to("cuda:0")

# Prepare data for COMET scoring
data = []
row_count = len(df_translation)
for i in range(row_count):
    data.append({
        "src": str(df_translation["Src-Ans-en"].iloc[i]),
        "mt": str(df_translation["Trans-Ans-hin"].iloc[i]),
        "ref": str(df_translation["Ref-Ans-hin"].iloc[i])
    })

# Get COMET scores
model_output = model.predict(data, batch_size=1, gpus=1)
comet_scores = model_output.scores

# Step 5: Save COMET scores back to the Excel file
df_translation["COMET Scores"] = comet_scores
df_translation.to_excel(output_excel_path, index=False)
print("COMET scores have been added to the Excel file.")

# Step 6: Visualize COMET Scores
plt.figure(figsize=(14, 7))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(comet_scores, bins=20, edgecolor='black', color='skyblue')
plt.title('Histogram of COMET Scores')
plt.xlabel('COMET Scores (Higher is better)')
plt.ylabel('# of responses')
plt.grid(True)

# Box-Whisker Plot
plt.subplot(1, 2, 2)
plt.boxplot(comet_scores, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Box-Whisker Plot of COMET Scores')
plt.xlabel('COMET Scores')

plt.tight_layout()

# Save the plot
plt.savefig('tabular_comet_scores_openai_no_as_far_as_visualization.png')
plt.show()

# Print summary statistics
print(f"Average COMET Score: {np.mean(comet_scores):.4f}")
print(f"Median COMET Score: {np.median(comet_scores):.4f}")
print(f"Standard Deviation of COMET Scores: {np.std(comet_scores):.4f}")

