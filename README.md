# Audio-Engineer-Sound-Design-LLM

This repository contains resources and code for developing a Language Learning Model (LLM) focused on audio engineering, sound design, and music production. The project aims to collect relevant data, prepare datasets, and fine-tune a model capable of answering various musical and audio-related questions.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
3. [Data Collection Tools](#data-collection-tools)
4. [Models and Libraries](#models-and-libraries)
   - [How DistilBERT Works](#how-distilbert-works)
   - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
   - [Visual Representations](#visual-representations)
5. [Projects to Watch and Learn From](#projects-to-watch-and-learn-from)
   - [Understanding ONNX and Its Benefits](#understanding-onnx-and-its-benefits)
6. [CPU Optimization](#cpu-optimization)
7. [Building Optimized TensorFlow Wheel](#building-optimized-tensorflow-wheel)
8. [TensorFlow Wheel](#tensorflow-wheel)
9. [Step-by-Step Guide](#step-by-step-guide)
   - [Data Collection](#1-data-collection)
   - [Transcription](#2-transcription)
   - [Dataset Preparation](#3-dataset-preparation)
   - [Model Training](#4-model-training)
   - [Deployment](#5-deployment)
   - [CPU Optimization](#6-cpu-optimization)
10. [Dataset Preparation and DistilBERT Training Example](#dataset-preparation-and-distilbert-training-example)
    - [Using Ollama with QA-DistillBert](#using-ollama-with-qa-distillbert)
    - [Key Aspects of QA-DistillBert](#key-aspects-of-qa-distillbert)
    - [Automated Question Correction using RAG](#automated-question-correction-using-rag)
11. [Contributing](#contributing)
12. [License](#license)
13. [References](#references)

## Project Structure

- `data_collection/`: Scripts for collecting audio transcripts and manuals
- `dataset_preparation/`: Code for preprocessing and preparing the dataset
- `model/`: Scripts for training and fine-tuning the LLM
- `utils/`: Utility functions for audio processing

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Follow the instructions in each directory to collect data, prepare the dataset, and train the model

## Data Collection Tools

We use the following tools for collecting and processing data:

- [youtube-subtitles-downloader](https://github.com/eliascotto/youtube-subtitles-downloader): A Node.js tool for downloading YouTube subtitles
- [PLWIZ](https://github.com/SomeOrdinaryBro/PLWIZ): For downloading video and Instagram content
- [transcribe](https://github.com/vivekuppal/transcribe) or [vibe](https://thewh1teagle.github.io/vibe/): For accurate transcription of audio content

## Models and Libraries

We incorporate the following models and libraries for reasoning, tokenization, and deployment:

- [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert): A lightweight BERT model for efficient tokenization and processing
- [Retrieval Augmented Generation (RAG)](https://huggingface.co/docs/transformers/model_doc/rag): A framework that combines retrieval-based methods with generative models to enhance response accuracy and context relevance
- [LLaMA](https://github.com/facebookresearch/llama): A large language model for advanced reasoning tasks
- [TensorFlow.js](https://www.tensorflow.org/js): For deploying and running the model in JavaScript environments

### How DistilBERT Works

DistilBERT is a smaller, faster, and lighter version of BERT that retains much of the original model's performance. Here's a simple explanation of how it works:

1. **Knowledge Distillation**: DistilBERT is trained using a process called knowledge distillation, where a smaller model (the student) learns to mimic a larger model (the teacher, in this case, BERT).

2. **Reduced Architecture**: It has fewer layers (6 instead of 12 in BERT) and fewer attention heads, resulting in a model that's about 40% smaller and 60% faster.

3. **Preserved Performance**: Despite its reduced size, DistilBERT retains about 97% of BERT's language understanding capabilities.

4. **Efficient Tokenization**: It uses the same tokenization process as BERT, allowing for efficient processing of text input.

5. **Fine-tuning**: Like BERT, DistilBERT can be fine-tuned for specific tasks such as classification, named entity recognition, or question answering.

For a more detailed explanation, watch this video on DistilBERT:

[![YouTube](http://i.ytimg.com/vi/90mGPxR2GgY/hqdefault.jpg)](https://www.youtube.com/watch?v=90mGPxR2GgY)

### Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) combines retrieval-based methods with generative models to enhance the model's ability to generate accurate and contextually relevant responses. By fetching relevant documents or data snippets during the generation process, RAG improves the quality and factual accuracy of the output.

#### Key Features of RAG:

1. **Dual Components**: RAG consists of a retriever and a generator. The retriever fetches relevant documents from a large corpus based on the input query, and the generator uses these documents to produce a coherent and accurate response.

2. **Enhanced Knowledge Utilization**: By integrating external information, RAG overcomes the limitations of models that rely solely on their pre-trained knowledge, allowing them to provide up-to-date and specific information.

3. **Improved Accuracy**: The retrieval mechanism ensures that the generated responses are grounded in actual data, reducing the occurrence of hallucinations and increasing the reliability of the answers.

4. **Flexibility**: RAG can be adapted to various domains by modifying the retrieval corpus, making it suitable for specialized applications like audio engineering, sound design, and music production.

#### Implementation in This Project:

In this project, we use RAG to enhance the LLM's ability to answer complex and specific questions related to audio engineering and sound design. The retrieval component accesses a curated dataset of audio engineering manuals, tutorials, and relevant documentation, while the generation component constructs accurate and informative responses.

For a deeper understanding, watch this video on RAG:

[![YouTube](http://i.ytimg.com/vi/rhZgXNdhWDY/hqdefault.jpg)](https://www.youtube.com/watch?v=rhZgXNdhWDY)

#### Resources:

- [Hugging Face RAG Documentation](https://huggingface.co/docs/transformers/model_doc/rag): Comprehensive guide on implementing and using RAG models.
- [RAG Paper](https://arxiv.org/abs/2005.11401): Original paper introducing RAG and its methodologies.

### Visual Representations

To better understand the architecture and functioning of transformer models like DistilBERT, refer to the following images:

1. **Transformer Model Architecture**

![Transformer Model Architecture](Transformers%20Model.png)

This image illustrates the overall architecture of a transformer model, showing the encoder and decoder components, as well as the attention mechanisms.

2. **Word Embeddings**

![Word Embeddings](Embeddings.png)

This image demonstrates how words are converted into numerical representations (embeddings) that can be processed by the model.

3. **BERT vs GPT/LLaMA**

![BERT vs GPT/LLaMA](BERT%20vs%20GPT_LLaMA.png)

This image compares the architecture and approach of BERT-based models (like DistilBERT) with GPT and LLaMA models, highlighting their differences in structure and application.

These visual aids help in grasping the complex structure and operations of transformer-based models like DistilBERT, as well as understanding how they compare to other popular language models.

## Projects to Watch and Learn From

As we develop our Audio-Engineer-Sound-Design-LLM, it's valuable to study and learn from similar projects that have successfully implemented and optimized question-answering models. Here are two noteworthy projects:

1. [multi-qa-MiniLM-distill-onnx-L6-cos-v1](https://huggingface.co/rawsh/multi-qa-MiniLM-distill-onnx-L6-cos-v1/tree/main): This project showcases a distilled version of MiniLM optimized for question-answering tasks and converted to ONNX format for improved inference performance.

2. [multi-qa-distilbert-cos-v1-onnx](https://huggingface.co/onnx-models/multi-qa-distilbert-cos-v1-onnx): This is an ONNX-ported version of the DistilBERT model fine-tuned for question-answering tasks, demonstrating how to optimize larger models for deployment.

### Understanding ONNX and Its Benefits

ONNX (Open Neural Network Exchange) is an open format designed to represent machine learning models. It defines a common set of operators and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

#### Why ONNX can be useful for our project:

1. **Improved Inference Time**: ONNX Runtime can significantly speed up model inference, which is crucial for real-time audio processing and question-answering tasks.

2. **Cross-Platform Compatibility**: ONNX models can be run on various hardware and operating systems, making our Audio-Engineer-Sound-Design-LLM more versatile and deployable across different environments.

3. **Optimization**: ONNX Runtime includes various optimization techniques that can automatically improve model performance without changing the model's architecture.

4. **Quantization**: ONNX supports model quantization, which can reduce model size and improve inference speed with minimal impact on accuracy. This is particularly useful for deploying models on edge devices or in resource-constrained environments.

5. **Integration with Hardware Accelerators**: ONNX models can easily leverage hardware accelerators like GPUs, TPUs, and specialized AI chips, potentially boosting performance for audio processing tasks.

#### Implementing ONNX in Our Project

To leverage ONNX in our Audio-Engineer-Sound-Design-LLM project, we can follow these steps:

1. **Convert to ONNX**: After training our DistilBERT or custom model, convert it to ONNX format using tools like `torch.onnx`.

2. **Optimize with ONNX Runtime**: Use ONNX Runtime to apply automatic optimizations to our model.

3. **Quantize if Necessary**: If deployment size or speed is a concern, apply quantization techniques to reduce model size while maintaining accuracy.

4. **Benchmark and Compare**: Test the ONNX model against our original model to measure improvements in inference time and resource usage.

By incorporating ONNX into our project, we can potentially achieve faster inference times, which is crucial for real-time audio processing and quick responses in our question-answering system. This optimization can lead to a more responsive and efficient Audio-Engineer-Sound-Design-LLM.

For more information on implementing ONNX in your projects, refer to the [ONNX official documentation](https://onnx.ai/get-started.html).

## CPU Optimization

For CPU optimization during training, we employ a custom-compiled wheel that leverages AVX2 and FMA instructions. This optimized wheel is designed to work seamlessly with Python 3.11, although Python 3.9 is the most commonly used version for machine learning and deep learning applications due to its broad compatibility.

To use the optimized wheel:

1. Ensure you have Python 3.9 installed
2. Install the `wheel` package: `pip install wheel`
3. Install the custom wheel (located in the `wheels/` directory)

## Building Optimized TensorFlow Wheel

To optimize TensorFlow for your specific CPU architecture and Python version, you can build a custom wheel. This process allows you to enable specific CPU instructions like AVX, AVX2, and FMA for improved performance. Follow these steps to build an optimized TensorFlow wheel:

### Prerequisites

- Bazel build system
- Python development environment
- Git

### Steps to Build

1. **Clone TensorFlow Repository:**
   ```bash
   git clone https://github.com/tensorflow/tensorflow.git
   cd tensorflow
   git checkout v2.17.0
   ```

2. **Configure Build:**
   Run the configuration script and answer the prompts. When asked about optimization flags, use:
   ```
   -march=native -mavx -mavx2 -mfma 
   ```

3. **Build the Wheel:**
   Use the following extensive command for compilation with specific optimizations:

   ```bash
   bazel build --config=opt \
     --copt=-march=skylake \
     --copt=-mtune=skylake \
     --copt=-mavx \
     --copt=-mavx2 \
     --copt=-mfma \
     --copt=-msse4.2 \
     --copt=-mpopcnt \
     --copt=-maes \
     --copt=-mf16c \
     --copt=-Wno-error=unknown-warning-option \
     --define=no_tensorflow_py_deps=true \
     --define=xnn_enable_avxvnniint8=false \
     --define=with_cuda=false \
     --define=with_rocm=false \
     --define=with_xla_support=false \
     //tensorflow/tools/pip_package:wheel \
     --repo_env=WHEEL_NAME=tensorflow_cpu \
     --verbose_failures
   ```

   This command builds a CPU-only version of TensorFlow with optimizations for Skylake architecture and includes AVX, AVX2, and FMA instructions.

4. **Create the Wheel:**
   After the build completes, create the wheel file:
   ```bash
   ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
   ```

5. **Install the Wheel:**
   Install the created wheel for your Python environment:
   ```bash
   pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl
   ```

### Notes

- Replace `skylake` with your specific CPU architecture if different.
- Adjust the TensorFlow version (`v2.17.0`) as needed for your project requirements.
- Building TensorFlow can take a significant amount of time and computational resources.
- Ensure your system meets the hardware requirements for the specified optimizations.

By following these steps, you can create a custom TensorFlow wheel optimized for your specific CPU architecture and Python version, potentially improving performance for your audio engineering and sound design LLM project.

## TensorFlow Wheel

The custom TensorFlow wheel for this project is too large to be included directly in the repository. You can download it from the [Releases page](https://github.com/ostinsolo/Audio-Engineer-Sound-Design-LLM/releases/tag/v0.0.2).

After downloading, place the wheel file in the `wheel/` directory of your local repository clone.

## Step-by-Step Guide

Follow this guide to set up the project, collect data, prepare the dataset, train the model, and deploy it.

### 1. Data Collection

**Tools Needed:**

- [youtube-subtitles-downloader](https://github.com/eliascotto/youtube-subtitles-downloader)
- [PLWIZ](https://github.com/SomeOrdinaryBro/PLWIZ)

**Steps:**

1. **Download YouTube Subtitles:**
   - Use `youtube-subtitles-downloader` to fetch subtitles from relevant YouTube videos.
   - Example command:
     ```bash
     youtube-subtitles-downloader --url "https://www.youtube.com/watch?v=example" --output data_collection/subtitles/
     ```

2. **Download Videos and Instagram Content:**
   - Use `PLWIZ` to download video content from YouTube and Instagram.
   - Example command:
     ```bash
     plwiz download --platform youtube --url "https://www.youtube.com/watch?v=example" --output data_collection/videos/
     ```

### 2. Transcription

**Tools Needed:**

- [transcribe](https://github.com/vivekuppal/transcribe) or [vibe](https://thewh1teagle.github.io/vibe/)

**Steps:**

1. **Transcribe Audio Content:**
   - Use `transcribe` or `vibe` to convert audio from videos into text.
   - Example command with `transcribe`:
     ```bash
     transcribe --input data_collection/videos/ --output data_collection/transcripts/
     ```

### 3. Dataset Preparation

**Directory:** `dataset_preparation/`

**Steps:**

1. **Preprocess Transcripts:**
   - Clean and format the transcription data.
   - Remove any unnecessary noise or artifacts from the text.

2. **Organize Data:**
   - Structure the data into training, validation, and testing sets.
   - Example:
     ```
     dataset_preparation/
     ├── train/
     ├── validation/
     └── test/
     ```

3. **Classification and Labeling:**
   
   **a. Define Labels:**
   - Determine the categories or topics relevant to your audio engineering and sound design context (e.g., mixing, mastering, sound synthesis).
   
   **b. Automatic Labeling with DistilBERT:**
   - Utilize DistilBERT to automate the classification of transcripts into predefined labels.
   - **Setup:**
     - Ensure you have the `transformers` library installed:
       ```bash
       pip install transformers
       ```
   
   - **Example Script:**
     - Create a script named `label_dataset.py` in the `dataset_preparation/` directory.
     
     `````language:dataset_preparation/label_dataset.py
     from transformers import pipeline
     import pandas as pd

     # Initialize the DistilBERT classifier
     classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

     # Load the transcript data
     transcripts = pd.read_csv('transcripts.csv')

     # Define a function to classify each transcript
     def classify_text(text):
         result = classifier(text[:512])[0]  # Truncate to 512 tokens if necessary
         return result['label']

     # Apply classification
     transcripts['label'] = transcripts['text'].apply(classify_text)

     # Save the labeled dataset
     transcripts.to_csv('labeled_transcripts.csv', index=False)
     ````
   
   - **Usage:**
     - Navigate to the `dataset_preparation/` directory and run the script:
       ```bash
       python label_dataset.py
       ```

   **c. Export to CSV:**
   - Ensure that the labeled data is saved in a CSV format (`labeled_transcripts.csv`) for easy integration with the training pipeline.

   **d. Automate the Process:**
   - To make the labeling process as automatic as possible, consider integrating the labeling script into your data pipeline or using scheduling tools to process new transcripts as they become available.

### 4. Model Training

**Directory:** `model/`

**Steps:**

1. **Configure Training Parameters:**
   - Set parameters like learning rate, batch size, and number of epochs in the training scripts.

2. **Train the Model:**
   - Use the prepared dataset to train the LLM.
   - Example command:
     ```bash
     python model/train.py --config model/config.yaml
     ```
   
3. **Fine-Tune the Model:**
   - Adjust the model based on validation performance to improve accuracy.

### 5. Deployment

**Tools Needed:**

- [TensorFlow.js](https://www.tensorflow.org/js)

**Steps:**

1. **Export the Trained Model:**
   - Convert the model to a format compatible with TensorFlow.js.
   - Example command:
     ```bash
     tensorflowjs_converter --input_format=tf_saved_model model/exported_model/ model/tfjs_model/
     ```
   
2. **Integrate with JavaScript Application:**
   - Use TensorFlow.js to load and run the model in a web environment.
   - Example code snippet:
     ```javascript
     const model = await tf.loadGraphModel('path/to/tfjs_model/model.json');
     ```

### 6. CPU Optimization

**Steps:**

1. **Install Optimized Wheel:**
   - Ensure Python 3.9 is installed.
   - Navigate to the project directory and install the wheel.
     ```bash
     pip install wheels/custom_wheel.whl
     ```
   
2. **Verify Installation:**
   - Run a test script to ensure the optimized wheel is functioning correctly.
     ```bash
     python utils/verify_optimization.py
     ```

## Dataset Preparation and DistilBERT Training Example

For an excellent example of how to prepare a dataset and train DistilBERT for a Question-Answering (QA) Retrieval System, we have created a customized setup in the `QA-DistillBert` folder. This setup demonstrates a comprehensive approach to developing a QA system using fine-tuned DistilBERT models integrated with Ollama.

### Using Ollama with QA-DistillBert

To utilize the QA-DistillBert example, follow these steps:

1. **Install Ollama:**
   - Visit [Ollama.com](https://ollama.com) to download and install Ollama for your operating system.

2. **Run the LLaMA Model:**
   - Open your terminal and execute the following command to run LLaMA:
     ```bash
     ollama run llama3.2
     ```

3. **Start Ollama Server:**
   - After running the model, close the Ollama application from the top menu bar on your Mac.
   - Then, start the Ollama server by running:
     ```bash
     ollama serve
     ```

4. **Navigate to QA-DistillBert Folder:**
   - Instead of using the external GitHub repository, we have created our own `QA-DistillBert` folder with updates and custom configurations.
   - Navigate to the `QA-DistillBert` directory:
     ```bash
     cd QA-DistillBert
     ```

5. **Follow the Customized Instructions:**
   - Inside the `QA-DistillBert` folder, you will find updated scripts and configurations tailored to work seamlessly with Ollama and our project's specific requirements.
   - Refer to the README.md inside the `QA-DistillBert` folder for detailed instructions on dataset preparation, model training, and deployment.

### Key Aspects of QA-DistillBert:

1. **Data Preparation:**
   - Converting tabular data into human-readable text documents.
   - Generating synthetic queries using Ollama's LLaMA model to create a diverse set of questions based on the document information.

2. **Model Selection and Evaluation:**
   - Evaluating pre-trained embedding models from Hugging Face, including `multi-qa-distilbert-cos-v1`.
   - Using statistical tests (Levene's test, Friedman test, Nemenyi post-hoc tests) to compare model performance across different question types.

3. **Hyperparameter Tuning:**
   - Employing Bayesian optimization to find optimal hyperparameters for fine-tuning the DistilBERT model.

4. **Fine-tuning Process:**
   - Fine-tuning the `multi-qa-distilbert-cos-v1` model using the `MultipleNegativesRankingLoss` function.
   - Monitoring various performance metrics during the training process.

5. **Performance Evaluation:**
   - Comparing the fine-tuned model against pre-trained models on a holdout test set.
   - Demonstrating significant improvements in performance metrics after fine-tuning.

For those interested in exploring text classification using DistilBERT, which can be a valuable step in automating dataset creation and cleaning, we recommend this tutorial: [Building a Text Classification Model using DistilBERT](https://medium.com/@prakashram1327/building-a-text-classification-model-using-distilbert-703c1409696c). This resource provides insights into how to prepare and process data for text classification tasks, which could be adapted to help automate the generation and categorization of questions for our audio engineering and sound design dataset. By implementing similar techniques, we could potentially streamline the process of creating a large, diverse set of questions without the need for manual writing of thousands of entries.

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

This customized `QA-DistillBert` setup serves as a robust reference for implementing similar QA retrieval systems, showcasing best practices in data preparation, model selection, and fine-tuning techniques, all integrated with Ollama for enhanced performance.

For more details, explore the `QA-DistillBert` folder within this repository.


## Contributing

Contributions are welcome! Please feel free to submit more files but NOT Pull Requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. [youtube-subtitles-downloader](https://github.com/eliascotto/youtube-subtitles-downloader): A Node.js tool for downloading YouTube subtitles, used in data collection.
2. [PLWIZ](https://github.com/SomeOrdinaryBro/PLWIZ): A tool for downloading video and Instagram content, used in data collection.
3. [transcribe](https://github.com/vivekuppal/transcribe): A tool for accurate transcription of audio content.
4. [vibe](https://thewh1teagle.github.io/vibe/): An alternative tool for audio transcription.
5. [DistilBERT - Hugging Face](https://huggingface.co/docs/transformers/model_doc/distilbert): Documentation for DistilBERT, the lightweight BERT model used in this project.
6. [Retrieval Augmented Generation (RAG) - Hugging Face](https://huggingface.co/docs/transformers/model_doc/rag): Documentation for RAG, used to enhance response accuracy and context relevance.
7. [LLaMA - Facebook Research](https://github.com/facebookresearch/llama): Large language model used for advanced reasoning tasks.
8. [TensorFlow.js](https://www.tensorflow.org/js): Library used for deploying and running the model in JavaScript environments.
9. [DistilBERT Explained Video](https://www.youtube.com/watch?v=90mGPxR2GgY): Educational video explaining the workings of DistilBERT.
10. [RAG Explained Video](https://www.youtube.com/watch?v=rhZgXNdhWDY): Educational video explaining Retrieval Augmented Generation.
11. [Hugging Face RAG Documentation](https://huggingface.co/docs/transformers/model_doc/rag): Comprehensive guide on implementing and using RAG models.
12. [RAG Paper](https://arxiv.org/abs/2005.11401): Original research paper introducing RAG and its methodologies.
13. [Ollama](https://ollama.com): Tool used for running and serving LLaMA models.
14. [TensorFlow GitHub Repository](https://github.com/tensorflow/tensorflow): Source code for TensorFlow, used in building optimized wheels.
15. [Audio-Engineer-Sound-Design-LLM Releases](https://github.com/ostinsolo/Audio-Engineer-Sound-Design-LLM/releases/tag/v0.0.2): Project releases, including custom TensorFlow wheels.
16. [LangChain.js GitHub Repository](https://github.com/langchain-ai/langchainjs): JavaScript library for building applications with large language models.
