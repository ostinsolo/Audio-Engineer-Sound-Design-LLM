# QA Information Retrieval System using Fine-Tuned DistilBERT



## Project Overview

This project focuses on developing and fine-tuning a question-answering (QA) information retrieval system to help customers find information about various medical equipments. The system is built using the [multi-qa-distilbert-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1) sentence-transformer model and fine-tuned on a custom dataset of 324 documents and 963 queries that emulate real customer inquiries about the company's products.

To develop the system, we first generate a document corpus from tabular data and create synthetic queries using a large language model (LLM) within our local Python computing environment. We then evaluate pre-trained sentence-transformer models from HuggingFace, perform Bayesian optimization for hyperparameter tuning, and fine-tune the most promising model using our synthesized document-query dataset.

The fine-tuned DistilBERT model consistently outperformed other pre-trained models on a holdout test set across various performance metrics, showcasing the effectiveness of the fine-tuning process and the model's adaptability to the specific domain and data characteristics.

## Project Resources

Below, you'll find information on where to locate specific files:

- [`notebooks/qa_system.ipynb`](notebooks/qa_system.ipynb): This Jupyter Notebook contains the majority of the code used in the project, including data cleaning, query generation using OLLaMA, significance testing, Bayesian optimization, and fine-tuning of the selected model.

- [`docs/`](docs/): The "docs" folder contains the following important documents:
  - [`Capstone_Final_Report.pdf`](docs/Capstone_Final_Report.pdf): The final report for this capstone project, detailing the project's objectives, methodology, results, and conclusions.
  - [`Capstone_3_Presentation.pptx`](docs/Capstone_3_Presentation.pptx): The PowerPoint slides used for the capstone project presentation.
  - [`Capstone_3_Presentation.pdf`](docs/Capstone_3_Presentation.pdf): A PDF version of the capstone project presentation slides.
  - [`Project Proposal.docx`](docs/Project_Proposal.docx): The initial project proposal document, outlining the project's objectives and intended deliverables.


Feel free to explore these resources to gain a deeper understanding of the project's implementation, findings, and deliverables.

## Data

The project utilizes two datasets that were created from a portion of the company's catalog:

1. `probes.csv`: This dataset contains information about various ultrasound probes, including the probe model, manufacturer, compatible systems, array type, frequency range, and stock availability.
2. `systems.csv`: This dataset contains information about different ultrasound systems, including the system model, manufacturer, and compatible probes.

## Methodology

### Converting Tabular Data into Documents

To prepare the data for the QA system, the tabular data from `probes.csv` and `systems.csv` was converted into human-readable text documents. Each row, column, and cell was transcribed into sentences using a python script.

### Generating Synthetic Queries using LLM

A large language model (LLM), llama3, was utilized to generate a diverse set of synthetic questions based on the information provided in each document. The prompt template provided examples of concise, answerable questions based on the information within each document. The model was instructed to generate three questions per document and structure its response as a numbered list.

```python
prompt = """
Generate three concise questions that can be answered using the following information, similar to the example provided. 
Provide only the questions, numbered as follows:

1. [insert question here]
2. [insert question here]
3. [insert question here]

Example 1:
Information: The manufacturer of the C3 probe is ATL.
Questions:
1. Who is the manufacturer of the C3 probe?
2. Who makes the C3 transducer?
3. Is the C3 probe made by ATL?

Information: {document}
Questions: 
"""
```

- A regular expression function was created to look for a sequence of a number followed by a period, whitespace, and then capture the text until the first question mark. This allowed us to parse the outputs, map each question onto its corresponding document and tags, and create a training dataset containing question-document pairs.

- Another LLM prompt was employed to score each question-document pair based on its relevance. Any question-document pairs deemed irrelevant were manually reviewed and re-scored if necessary. These pairs were then merged back into the final dataset.

The final fine-tuning dataset comprised 963 queries and a corpus of 324 documents.

### Automated Question Correction using RAG

To further improve the quality of our dataset and reduce manual review, we implemented an additional layer using Retrieval Augmented Generation (RAG) with a local LLM. This step automatically corrects filtered questions that were initially deemed irrelevant or low-quality.

We have implemented two methods for this RAG system:

1. **Custom RAG Implementation using LangChain and LLaMA**:

   This method uses LangChain with a local LLaMA model to create a custom RAG system.

   ```python
   from langchain import PromptTemplate, LLMChain
   from langchain.llms import LlamaCpp
   from langchain.embeddings import HuggingFaceEmbeddings
   from langchain.vectorstores import FAISS
   from langchain.chains import RetrievalQA

   # Setup RAG system
   embeddings = HuggingFaceEmbeddings()
   vector_store = FAISS.from_texts(documents, embeddings)
   retriever = vector_store.as_retriever()

   llm = LlamaCpp(model_path="path/to/llama/model.bin")
   qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

   # Function to correct and re-evaluate questions
   def correct_and_reevaluate_question(question, document):
       context = qa_chain.run(question)
       
       correction_prompt = PromptTemplate(
           input_variables=["question", "context"],
           template="Given the context: {context}\n\nImprove or correct this question: {question}"
       )
       
       correction_chain = LLMChain(llm=llm, prompt=correction_prompt)
       corrected_question = correction_chain.run(question=question, context=context)
       
       # Re-evaluate corrected question (implementation depends on your scoring method)
       new_score = evaluate_relevance(corrected_question, document)
       
       return corrected_question, new_score

   # Process filtered questions
   for question, document in filtered_pairs:
       corrected_question, new_score = correct_and_reevaluate_question(question, document)
       if new_score > relevance_threshold:
           final_dataset.append((corrected_question, document))
   ```

2. **Using PrivateGPT**:

   This method leverages PrivateGPT, an open-source project that provides a production-ready AI system for asking questions about documents using LLMs, with a focus on privacy and offline capabilities.

   To set up PrivateGPT:

   a. Clone the PrivateGPT repository:
      ```bash
      git clone https://github.com/zylon-ai/private-gpt.git
      cd private-gpt
      ```

   b. Follow the installation instructions in the PrivateGPT documentation.

   c. Ingest your document corpus into PrivateGPT.

   d. Use PrivateGPT's API to correct and re-evaluate questions:

   ```python
   import requests

   def correct_and_reevaluate_question_privategpt(question, document):
       # Assuming PrivateGPT API is running on localhost:8001
       url = "http://localhost:8001/v1/completions"
       
       prompt = f"""Given the context of the following document:
       {document}

       Improve or correct this question: {question}"""

       payload = {
           "prompt": prompt,
           "max_tokens": 100
       }
       
       response = requests.post(url, json=payload)
       corrected_question = response.json()['choices'][0]['text'].strip()
       
       # Re-evaluate corrected question (implementation depends on your scoring method)
       new_score = evaluate_relevance(corrected_question, document)
       
       return corrected_question, new_score

   # Process filtered questions
   for question, document in filtered_pairs:
       corrected_question, new_score = correct_and_reevaluate_question_privategpt(question, document)
       if new_score > relevance_threshold:
           final_dataset.append((corrected_question, document))
   ```

Both methods significantly reduce the need for manual review and improve the overall quality of the synthetic queries in our dataset. The choice between these methods depends on specific project requirements, such as privacy concerns, offline capabilities, and integration with existing systems.

After applying one of these automated correction processes, we merge the corrected questions back into the dataset:

```python
# Merge corrected questions back into the dataset
df_final = pd.concat([df_final, pd.DataFrame(final_dataset)])
```

### Evaluating Pre-trained Embedding Models

Three pre-trained embedding models from Hugging Face were evaluated: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1), and [multi-qa-distilbert-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1).

To compare the performance of the pre-trained embedding models across different question types, a Levene's test was conducted to evaluate the homogeneity of variances across models, and the null hypothesis was rejected. Subsequently, a Friedman test (non-parametric repeated measures ANOVA) was performed, followed by Nemenyi post-hoc tests. Out of the eight question categories, six yielded significant differences.

The overall trend showed that the `multi-qa-distilbert-cos-v1` model performed equally as high, if not the highest, in terms of Mean Reciprocal Rank (MRR), followed by the other question-answering model, `multi-qa-mpnet-base-dot-v1`. The generalized model, `all-mpnet-base-v2`, performed the worst. Consequently, the `multi-qa-distilbert-cos-v1` model was selected for fine-tuning due to its superior performance across most question types.




### Bayesian Optimization

Bayesian optimization was employed to find the optimal hyperparameters for fine-tuning the multi-qa-distilbert-cos-v1 model. The search space included `per_gpu_batch_size`, `weight_decay`, `learning_rate`, `warmup_steps`, and `num_epochs`. The optimization process aimed to maximize the MRR metric on the validation set.

### Fine-tuning

The `multi-qa-distilbert-cos-v1` model was fine-tuned using the best hyperparameters obtained from Bayesian optimization. The `MultipleNegativesRankingLoss` function was used as the loss function, which encourages the model to assign higher probabilities to relevant documents and lower probabilities to irrelevant ones. Various performance metrics, such as accuracy, recall, MRR, and NDCG (Normalized Discounted Cumulative Gain), were observed over the training process lasting 4 epochs.

### Model Performance and Comparison

The fine-tuned DistilBERT model was evaluated on a holdout test set and compared against three other pre-trained models: `multi-qa-distilbert-cos-v1`, `multi-qa-mpnet-base-dot-v1`, and `all-mpnet-base-v2`. The fine-tuned model consistently outperformed the other models across all performance metrics, demonstrating the effectiveness of the fine-tuning process and the model's ability to adapt to the specific domain and characteristics of the company's data.



## Conclusion

This project successfully developed a QA information retrieval system using a fine-tuned DistilBERT model to answer customer queries about ultrasound probes and systems. The system's performance highlights the importance of tailoring the model to the specific domain and data characteristics. Future work could explore expanding the QA system to handle more complex queries and integrate large language model chains with structured database queries for even more accurate and comprehensive responses.

## Libraries & Dependencies

- Python 3.x
- numpy
- pandas
- torch
- sklearn
- matplotlib
- sentence-transformers
- statsmodels
- scipy
- langchain
- bayesian-optimization
- requests (for PrivateGPT implementation)

## Using Ollama with QA-DistillBert

To utilize this QA-DistillBert example, follow these steps:

1. **Install Ollama:**
   - Visit [Ollama.com](https://ollama.com) to download and install Ollama for your operating system.

2. **Run the LLaMA Model:**
   - Open your terminal and execute:
     ```bash
     ollama run llama3.2
     ```

3. **Start Ollama Server:**
   - After running the model, close the Ollama application from the top menu bar on your Mac.
   - Start the Ollama server:
     ```bash
     ollama serve
     ```

4. **Navigate to QA-DistillBert Folder:**
   - Change to the `QA-DistillBert` directory:
     ```bash
     cd QA-DistillBert
     ```

5. **Follow the Instructions:**
   - Refer to the sections below for detailed instructions on dataset preparation, model training, and deployment.

## Key Aspects of QA-DistillBert

1. **Data Preparation:** Converting tabular data into human-readable text documents and generating synthetic queries using Ollama's LLaMA model.

2. **Model Selection and Evaluation:** Evaluating pre-trained embedding models from Hugging Face and using statistical tests to compare model performance.

3. **Hyperparameter Tuning:** Employing Bayesian optimization for fine-tuning the DistilBERT model.

4. **Fine-tuning Process:** Using the `MultipleNegativesRankingLoss` function and monitoring various performance metrics.

5. **Performance Evaluation:** Comparing the fine-tuned model against pre-trained models on a holdout test set.
