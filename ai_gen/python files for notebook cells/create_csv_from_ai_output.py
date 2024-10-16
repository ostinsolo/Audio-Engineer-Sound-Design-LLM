import pandas as pd
from tqdm.auto import tqdm
from generated_utterances_and_actions import generated_data

# Enable tqdm for pandas operations
tqdm.pandas()

def create_csv_from_ai_output(input_data, output_file):
    # Convert the input data to a pandas DataFrame
    df = pd.DataFrame(input_data)
    
    # Convert the Action_Order list to a comma-separated string
    df['Action_Order'] = df['Action_Order'].progress_apply(lambda x: ', '.join(x))
    
    # Write the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"CSV file '{output_file}' has been created successfully.")

def main():
    input_file = 'generated_utterances_and_actions.py'
    output_file = 'ableton_utterances_and_actions.csv'

    create_csv_from_ai_output(generated_data, output_file)

if __name__ == "__main__":
    main()

