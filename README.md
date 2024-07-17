# Hindi Conversation Analysis
This repo aims to analyze a hindi conversation between an agent and customer
## Analysis Process
- Dividing the whole conversation into individual statements
- Counting frequencies of different words to get the most valuable statement
- For the purpose of sentiment analysis, positive and negative sets are created
- Weights of each word of each statement are analyzed with these sets to determine the sentiment of each statement
- For individual speaker, we analyze the change in sentiment

## Additional Analysis with HuggingFace Model
A separate anlaysis with the help of huggingface model named `txlm-roberta-hindi-sentiment` is carried out in the `sentiment_analysis.ipynb` notebook.
