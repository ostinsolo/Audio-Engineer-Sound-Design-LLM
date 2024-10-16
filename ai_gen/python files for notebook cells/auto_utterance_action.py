import random
import re
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Load the Ableton data
from ableton_data import *

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
1. Track-related actions always start with "track {track_number}" unless it's a global action.
2. Device-related actions always start with "search device" followed by the device name.
3. For instruments, the order is: "search device", "{instrument}", "{device_type}", then the action.
4. Audio effects are treated separately from instruments.
5. Control actions include one of the speed modifiers, followed by a value between 0 and 100.
6. When creating a track, don't mention a track number (it doesn't exist yet).
7. Use one of the common actions in the utterance.
8. You can use or adapt one of the utterance templates, or create a new utterance based on the provided data.

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
    audio_effect = random.choice(audio_effects)
    instrument = random.choice(instruments)
    device_type = random.choice(device_types[instrument])
    action_category = random.choice(list(actions.keys()))
    action = random.choice(actions[action_category])
    template = random.choice(utterance_templates)
    speed_modifier = random.choice(sum(speed_modifiers.values(), []))
    
    # Create a simplified version of the Ableton data to pass to the AI
    simplified_data = {
        'selected_audio_effect': audio_effect,
        'selected_instrument': instrument,
        'selected_device_type': device_type,
        'selected_action': action,
        'selected_template': template,
        'selected_speed_modifier': speed_modifier,
        'action_order_templates': action_order_templates
    }
    
    prompt = prompt_template.format(ableton_data=str(simplified_data))
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
    with open("generated_utterances_and_actions.py", "w") as f:
        f.write("generated_data = " + str(generated_data))

    print(f"Generated {len(generated_data)} utterances and action orders.")
    print("Results saved to generated_utterances_and_actions.py")

if __name__ == "__main__":
    main()
