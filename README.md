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
- [LLaMA](https://github.com/facebookresearch/llama): A large language model for advanced reasoning tasks
- [TensorFlow.js](https://www.tensorflow.org/js): For deploying and running the model in JavaScript environments

## CPU Optimization

We use a custom-compiled wheel with AVX2 and FMA enabled for CPU optimization during training. This wheel is compatible with Python 3.9, which is currently the most compatible Python version for machine learning and deep learning technologies.

To use the optimized wheel:

1. Ensure you have Python 3.9 installed
2. Install the `wheel` package: `pip install wheel`
3. Install the custom wheel (located in the `wheels/` directory)

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

3. **Classification (Optional):**
   - If needed, classify data into different categories such as topic, difficulty, or genre.
   - Use scripts in `dataset_preparation/` for automated classification.

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
