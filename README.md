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

## Models

We incorporate the following models for reasoning and tokenization:

- LLaMA: A large language model for advanced reasoning tasks
- DistilBERT: A lightweight BERT model for efficient tokenization and processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
