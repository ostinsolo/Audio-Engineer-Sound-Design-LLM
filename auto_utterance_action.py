import json
import random
import re
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Load the Ableton data
with open('ableton_data.json', 'r') as f:
    ableton_data = json.load(f)

# Load the Ollama LLM
llm = OllamaLLM(model="llama3.2")

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["ableton_data"],
    template="""
Generate a unique utterance for an Ableton Live task and its corresponding action order. Use the provided Ableton data to ensure relevance and accuracy.

Ableton Data:
{ableton_data}

Format the output as follows:
Utterance: [insert utterance here]
Action Order: ["step 1", "step 2", ...]

Rules:
1. If the utterance contains "track" with a number, the first action should be the specific track number.
2. If the utterance mentions an audio effect, instrument, or device type, include it in the action order.
3. Use one of the common actions in the utterance.
4. You can use or adapt one of the utterance templates, or create a new utterance based on the provided data.

Now generate a new, unique utterance and action order:
"""
)

def extract_utterance_and_action(response):
    utterance_match = re.search(r'Utterance: (.+)', response)
    action_order_match = re.search(r'Action Order: (\[.+\])', response)
    
    utterance = utterance_match.group(1) if utterance_match else None
    action_order = eval(action_order_match.group(1)) if action_order_match else None
    
    return utterance, action_order

def generate_utterance_and_action():
    # Randomly select elements from the Ableton data to encourage variety
    audio_effect = random.choice(ableton_data['audio_effects'])
    device_type = random.choice(ableton_data['device_types'])
    instrument = random.choice(ableton_data['instruments'])
    action = random.choice(ableton_data['common_actions'])
    template = random.choice(ableton_data['utterance_templates'])
    
    # Create a simplified version of the Ableton data to pass to the AI
    simplified_data = {
        'selected_audio_effect': audio_effect,
        'selected_device_type': device_type,
        'selected_instrument': instrument,
        'selected_action': action,
        'selected_template': template
    }
    
    prompt = prompt_template.format(ableton_data=json.dumps(simplified_data, indent=2))
    output = llm(prompt)
    return extract_utterance_and_action(output)

def generate_action_order(utterance, template):
    action_order = template.copy()
    
    # Handle track creation separately
    if "create track" in utterance or "new track" in utterance:
        return handle_track_creation(utterance, action_order)
    
    # If a device is mentioned, ensure it starts with "search device"
    if any(device in utterance for device in ["{audio_effect}", "{instrument}", "{device_type}"]):
        if action_order[0] != "search device":
            action_order.insert(0, "search device")
    
    # For Control actions, generate a random value
    if "Control" in utterance:
        value_index = action_order.index("{value}")
        action_order[value_index] = str(random.randint(0, 100))
    
    # Ensure track-related actions start with the track number
    if "track" in utterance and not action_order[0].startswith(("track", "create track")):
        track_number = extract_track_number(utterance)
        action_order.insert(0, f"track {track_number}")
    
    return action_order

def handle_track_creation(utterance, action_order):
    # For track creation, we don't need to add a track number
    if "create track" not in action_order:
        action_order.insert(0, "create track")
    
    # Extract track type and instrument if present
    track_type = extract_track_type(utterance)
    instrument = extract_instrument(utterance)
    
    if track_type:
        action_order.insert(1, track_type)
    if instrument:
        action_order.append(f"add {instrument}")
    
    return action_order

def extract_track_number(utterance):
    match = re.search(r'track (\d+)', utterance)
    return match.group(1) if match else "1"  # Default to track 1 if no number found

def extract_track_type(utterance):
    track_types = ["audio", "MIDI", "return", "master", "group"]
    for track_type in track_types:
        if track_type.lower() in utterance.lower():
            return track_type
    return "audio"  # Default to audio if no type specified

def extract_instrument(utterance):
    # This function would contain logic to extract the instrument name from the utterance
    # For simplicity, let's assume it just checks for the presence of "instrument" keyword
    if "instrument" in utterance:
        return "{instrument}"
    return None

def main():
    num_generations = 100  # You can adjust this number
    generated_data = []

    for _ in tqdm(range(num_generations), desc="Generating utterances and actions"):
        utterance, action_order = generate_utterance_and_action()
        if utterance and action_order:
            generated_data.append({"Utterance": utterance, "Action_Order": action_order})

    # Save the results
    with open("generated_utterances_and_actions.json", "w") as f:
        json.dump(generated_data, f, indent=2)

    print(f"Generated {len(generated_data)} utterances and action orders.")
    print("Results saved to generated_utterances_and_actions.json")

if __name__ == "__main__":
    main()
