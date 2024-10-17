# AI-Generated Ableton Live Voice Commands

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ostinsolo/Audio-Engineer-Sound-Design-LLM/blob/main/ai_gen/auto_utterance_action.ipynb)

**Note on Colab Usage:** The Colab link above has been added for easy viewing and testing of the notebook structure. However, please be aware that the current implementation uses Ollama, a local LLM service, which cannot run directly in the Colab environment. To fully execute this code, you'll need to run it locally with Ollama installed, or modify the LLM implementation to use a cloud-based service accessible from Colab.

This project aims to generate a dataset of natural language utterances and corresponding action sequences for controlling Ableton Live using voice commands. The generated data can be used to train a machine learning model for voice-controlled music production.

## Project Overview

The main components of this project are:

1. **Data Generation**: Using AI to create diverse and realistic voice command utterances and their corresponding action sequences in Ableton Live.
2. **Data Processing**: Converting the generated data into a format suitable for machine learning models.
3. **Ableton Live Integration**: Utilizing Ableton Live's data structures and actions to ensure the generated commands are relevant and executable.

## Key Files

- `auto_utterance_action.ipynb`: Jupyter notebook containing the main logic for generating utterances and action sequences.
- `ableton_data.py`: Contains structured data about Ableton Live, including audio effects, instruments, actions, and templates.
- `create_csv_from_ai_output.py`: Script to convert the generated data into a CSV file for further processing or model training.
- `requirements_for_ai_gen.txt`: List of Python packages required for this project.

## How It Works

1. The system uses a language model (Ollama LLM) to generate natural language utterances based on Ableton Live's structure and capabilities.
2. For each utterance, a corresponding action sequence is generated, representing the steps needed to execute the command in Ableton Live.
3. The generated data is then processed and saved to a CSV file, which can be used for training a machine learning model.

## Usage

1. Ensure all required packages are installed: `pip install -r requirements_for_ai_gen.txt`
2. Run the `auto_utterance_action.ipynb` notebook to generate the dataset.
3. The resulting CSV file (`ableton_utterances_and_actions.csv`) will contain the generated utterances and their corresponding action sequences.

### Using Ollama with AI-Generated Ableton Live Voice Commands

To utilize this project with Ollama, follow these steps:

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

4. **Use the Project:**
   - With Ollama running, you can now use the `auto_utterance_action.ipynb` notebook to generate the dataset as described in the usage instructions above.

## Development Status

**Important Note:** This system is currently under active development and has not been thoroughly tested yet. We welcome contributions and testing to identify and resolve potential errors or inconsistencies in the generated data.

## Future Development

This dataset can be used to train a machine learning model that can interpret natural language commands and convert them into actionable sequences in Ableton Live, enabling voice-controlled music production.

As the project evolves, we aim to:
1. Improve the accuracy and diversity of generated utterances and actions
2. Expand the coverage of Ableton Live features
3. Implement and test the voice command interpretation model

## Contributing

Contributions to improve the data generation process, expand the Ableton Live data structures, or enhance the overall system are welcome. We especially encourage:

- Testing the generated datasets for accuracy and consistency
- Identifying and reporting any errors or unexpected outputs
- Suggesting improvements to the data generation algorithms
- Expanding the Ableton Live data structures to cover more features

Please submit a pull request or open an issue to discuss proposed changes or report any findings from your testing.
