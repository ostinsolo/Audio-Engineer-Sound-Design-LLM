# Audio-Engineer-Sound-Design-LLM

This repository contains resources and code for developing a Language Learning Model (LLM) focused on audio engineering, sound design, and music production. The project aims to collect relevant data, prepare datasets, and fine-tune a model capable of answering various musical and audio-related questions.

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
     
     ````language:dataset_preparation/label_dataset.py
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

For an excellent example of how to prepare a dataset and train DistilBERT for a Question-Answering (QA) Retrieval System, we can look at the project by Dennis van Dang. This project demonstrates a comprehensive approach to developing a QA system using fine-tuned DistilBERT models.

Key aspects of the project include:

1. **Data Preparation:**
   - Converting tabular data into human-readable text documents.
   - Generating synthetic queries using a large language model (LLaMA) to create a diverse set of questions based on the document information.

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

This project serves as an excellent reference for implementing similar QA retrieval systems, showcasing best practices in data preparation, model selection, and fine-tuning techniques.

For more details, you can explore the [QA-Retrieval-System repository](https://github.com/dennisvdang/QA-Retrieval-System/tree/main) on GitHub.
## Contributing

Contributions are welcome! Please feel free to submit more files but NOT Pull Requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

