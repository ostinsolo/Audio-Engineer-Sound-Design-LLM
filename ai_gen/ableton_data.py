# ableton_data.py

# List of all available audio effects in Ableton Live
audio_effects = [
    "Amp",
    "Audio Effect Racks",
    "Auto Filter",
    "Auto Pan",
    "Beat Repeat",
    "Cabinet",
    "Channel Eq",
    "Chorus-Ensemble",
    "Compressor",
    "Corpus",
    "Delay",
    "Drum Buss",
    "Drum Racks",
    "Dynamic Tube",
    "Echo",
    "Envelope Follower",
    "EQ Eight",
    "EQ Three",
    "Erosion",
    "FilterDelay",
    "Frequency Shifter",
    "Gate",
    "Gated Delay",
    "Glue Compressor",
    "Grain Delay",
    "Hybrid Reverb",
    "Impulse",
    "LFO",
    "Limiter",
    "Looper",
    "Multiband Dynamics",
    "Overdrive",
    "Pedal",
    "Phaser",
    "Pitch Hack",
    "Ping Pong Delay",
    "Redux",
    "Resonator",
    "Reverb",
    "Roar",
    "Saturator",
    "Scale",
    "Sampler",
    "Simpler",
    "Shifter",
    "Shaper",
    "Spectrum",
    "Spectral Blur",
    "Tuner",
    "Utility",
    "Vinyl Distortion",
    "Vocoder"
]

# Mapping of instruments to their available device types (presets)
device_types = {
    "Analog": [
        "Bass",
        "Brass",
        "Effects",
        "Piano & Keys",
        "Synth Keys",
        "Synth Lead",
        "Synth Pad",
        "Synth Percussion"
    ],
    "Collision": [
        "Ambient & Evolving",
        "Guitar & Plucked",
        "Mallets",
        "Piano & Keys"
    ],
    "Drift": [
        "Bass",
        "Brass",
        "Effects",
        "Mallets",
        "Pad",
        "Percussive",
        "Piano & Keys",
        "Plucked",
        "Strings",
        "Synth Keys",
        "Synth Lead",
        "Synth Rhythmic"
    ],
    "Drum Synth": [
        "Drums",
        "Percussion"
    ],
    "Electric": [
        "Piano & Keys"
    ],
    "External Instrument": [],
    "Instrument Rack": [
        "Ambient & Evolving",
        "Bass",
        "Brass",
        "Drums",
        "Effects",
        "Guitar & Plucked",
        "Mallets",
        "Pad",
        "Percussive",
        "Piano & Keys",
        "Strings",
        "Synth",
        "Synth Keys",
        "Synth Lead",
        "Synth Pad",
        "Synth Rhythmic",
        "Synth Voices",
        "Voices",
        "Winds"
    ],
    "Meld": [
        "Bass",
        "Effects",
        "Pad",
        "Percussive",
        "Synth Keys",
        "Synth Lead",
        "Synth Rhythmic"
    ],
    "Operator": [
        "Ambient & Evolving",
        "Bass",
        "Brass",
        "Effects",
        "Guitar & Plucked",
        "Mallets",
        "Piano & Keys",
        "Synth Keys",
        "Synth Lead",
        "Synth Pad",
        "Synth Percussion",
        "Synth Rhythmic",
        "Synth Voices",
        "Winds"
    ],
    "Sampler": [
        "Ambient & Evolving",
        "Bass",
        "Effects",
        "Synth Keys",
        "Synth Lead",
        "Synth Pad",
        "Synth Rhythmic"
    ],
    "Simpler": [
        "Ambient & Evolving",
        "Bass",
        "Effects",
        "Pad",
        "Piano & Keys",
        "Synth Keys",
        "Synth Lead",
        "Synth Rhythmic"
    ],
    "Tension": [
        "Ambient & Evolving",
        "Bass",
        "Effects",
        "Guitar & Plucked",
        "Mallets",
        "Piano & Keys",
        "Strings",
        "Synth Keys",
        "Synth Pad"
    ],
    "Wavetable": [
        "Ambient & Evolving",
        "Bass",
        "Brass",
        "Effects",
        "Guitar & Plucked",
        "Mallets",
        "Percussive",
        "Piano & Keys",
        "Synth Keys",
        "Synth Lead",
        "Synth Pad",
        "Synth Rhythmic"
    ]
}

# List of all available instruments in Ableton Live
instruments = [
    "Analog",
    "Collision",
    "Drift",
    "Drum Synth",
    "Electric",
    "External Instrument",
    "Instrument Rack",
    "Meld",
    "Operator",
    "Sampler",
    "Simpler",
    "Tension",
    "Wavetable"
]

# List of parameters that can be adjusted for various devices
parameters = [
    "volume",
    "pan",
    "send",
    "return",
    "attack",
    "decay",
    "sustain",
    "release",
    "cutoff",
    "resonance",
    "drive",
    "mix",
    "feedback",
    "depth",
    "rate",
    "time",
    "frequency",
    "gain",
    "threshold",
    "ratio",
    "knee",
    "makeup",
    "activate",
    "enable",
    "deactivate",
    "disable",
    "dry/wet"
]

# Types of tracks available in Ableton Live
track_types = [
    "audio",
    "MIDI",
    "return",
    "master",
    "group"
]

# Categorized actions for different elements in Ableton Live
actions = {
    "track_creation_actions": [
        "add",
        "create"
    ],
    "track_actions": [
        "remove",
        "arm",
        "solo",
        "unsolo",
        "mute",
        "unmute",
        "rename",
        "delete",
        "duplicate",
        "freeze",
        "unfreeze",
        "group",
        "bypass",
        "unbypass",
        "set volume",
        "set pan",
        "set color"
    ],
    "project_actions": [
        "create scene",
        "delete scene",
        "duplicate scene",
        "set tempo",
        "set time signature",
        "set loop start",
        "set loop end",
        "record",
        "stop",
        "play",
        "pause",
        "continue",
        "resume"
    ],
    "device_actions": [
        "add",
        "remove",
        "change",
        "enable",
        "disable",
        "bypass",
        "unbypass",
        "set parameter"
    ],
    "clip_actions": [
        "create",
        "delete",
        "duplicate",
        "rename",
        "loop",
        "unloop",
        "quantize",
        "warp",
        "unwarp",
        "transpose",
        "reverse",
        "view",
        "record",
        "stop",
        "play",
        "pause",
        "freeze",
        "unfreeze"
    ],
    "view_actions": [
        "show Arranger",
        "show Session",
        "show Device Chain"
    ],
    "mapping_actions": [
        "Map",
        "Delete Map",
        "Set Min Range",
        "Set Max Range",
        "Control"
    ],
    "value_actions": [
        "set",
        "increase",
        "decrease"
    ],
    "speed_modifiers": {
        "instant": ["now", "immediately"],
        "fast": ["fast", "quick", "rapidly"],
        "slow": ["slow", "gradually", "smoothly"]
    }
}

# Templates for generating action orders (sequences of actions)

# Action Order Rules:
# 1. Track creation actions don't include a track number (the track doesn't exist yet)
# 2. Other track-related actions always start with "track {track_number}"
# 3. Device-related actions always start with "track {track_number}", "search device", followed by the device name
# 4. For instruments, the order is: "track {track_number}", "search device", "{instrument}", "{device_type}"
# 5. Audio effects are treated separately from instruments
# 6. Value actions include the device name, parameter, action (set/increase/decrease), speed modifier, and value
# 7. Project actions are global and don't require a track number
# 8. Clip actions always include a track number and clip number
# 9. View actions change the current view in Ableton Live and don't require a track number
# 10. Mapping actions involve assigning controls to parameters and may not require a track number

# Placeholder Explanations:
# {track_creation_actions}: Actions for creating new tracks (e.g., add, create)
# {track_number}: The number of the track (e.g., 1, 2, 3)
# {track_actions}: Any action from the track_actions list (e.g., arm, solo, mute)
# {audio_effect}: Any audio effect from the audio_effects list
# {instrument}: Any instrument from the instruments list
# {device_type}: Any device type from the device_types dictionary for the specified instrument
# {project_actions}: Any action from the project_actions list (e.g., create scene, set tempo)
# {track_type}: Any track type from the track_types list (e.g., audio, MIDI)
# {clip_number}: The number of the clip (e.g., 1, 2, 3)
# {clip_actions}: Any action from the clip_actions list (e.g., create, delete, loop)
# {device_name}: The name of the device being adjusted
# {parameter}: The parameter of the device being adjusted
# {value_actions}: Actions that change values (set, increase, decrease)
# {speed_modifiers}: Any speed modifier from the speed_modifiers dictionary
# {value}: A numeric value, typically between 0 and 100
# {number}: A numeric value, used for Control or Map operations
# {view_actions}: Any action from the view_actions list (e.g., show Arranger, show Session)
# {mapping_actions}: Any action from the mapping_actions list (e.g., Map, Delete Map)

# Action Types and Usage:
# 1. Track creation actions: 
#    - Create new tracks
#    - Don't require a track number
#
# 2. Track actions: 
#    - Performed on specific tracks
#    - Require a track number
#
# 3. Project actions: 
#    - Global actions that affect the entire project
#    - Don't require a track number
#
# 4. Device actions: 
#    - Performed on devices within tracks
#    - Require a track number and device name
#
# 5. Clip actions: 
#    - Performed on clips within tracks
#    - Require a track number and clip number
#
# 6. Value actions:
#    - Modify parameters of devices on tracks
#    - Require track number, device name, parameter, action, speed modifier, and value
#
# 7. View actions: 
#    - Change the current view in Ableton Live
#    - Don't require a track number
#
# 8. Mapping actions: 
#    - Involve assigning controls to parameters
#    - May or may not require a track number

# Updated action order templates
action_order_templates = [
    ["{track_creation_actions}", "{track_type}"],
    ["track {track_number}", "{track_actions}"],
    ["track {track_number}", "search device", "{audio_effect}"],
    ["track {track_number}", "search device", "{instrument}", "{device_type}"],
    ["{project_actions}"],
    ["track {track_number}", "clip {clip_number}", "{clip_actions}"],
    ["track {track_number}", "duplicate", "search device", "{audio_effect}"],
    ["track {track_number1}", "track {track_number2}", "group"],
    ["track {track_number}", "create_send", "search device", "{audio_effect}"],
    ["track master", "search device", "{audio_effect}"],
    ["track {track_number}", "search device", "{device_name}", "{parameter}", "{value_actions}", "{speed_modifiers}", "{value}"],
    ["Map {number}", "{parameter}", "{value}"],
    ["Delete", "Map {number}"],
    ["{view_actions}"],
    
    # Action orders for Extended Utterance Templates
    ["{track_creation_actions}", "{track_type}"],  # "Please {track_creation_actions} a new {track_type} track"
    ["track {track_number}", "{track_actions}"],  # "Could you {track_actions} the track number {track_number}?"
    ["track {track_number}", "search device", "{audio_effect}"],  # "Kindly add the {audio_effect} effect to track {track_number}"
    ["track {track_number}", "search device", "{instrument}", "{device_type}"],  # "Insert a {instrument} with {device_type} into track {track_number}"
    ["{project_actions}"],  # "Execute the project action: {project_actions}"
    ["track {track_number}", "clip {clip_number}", "{clip_actions}"],  # "Perform the {clip_actions} action on clip {clip_number} in track {track_number}"
    ["track {track_number}", "duplicate", "search device", "{audio_effect}"],  # "Duplicate track {track_number} and apply the {audio_effect} effect"
    ["track {track_number1}", "track {track_number2}", "group"],  # "Group together tracks {track_number1} and {track_number2}"
    ["track {track_number}", "create_send", "search device", "{audio_effect}"],  # "Set up a send from track number {track_number} to {audio_effect}"
    ["track master", "search device", "{audio_effect}"],  # "Apply the {audio_effect} effect to the master track"
    ["track {track_number}", "search device", "{device_name}", "{parameter}", "{value_actions}", "{speed_modifiers}", "{value}"],  # "Please {value_actions} the {parameter} of {device_name} on track {track_number} to {value} {speed_modifiers}"
    ["Map {number}", "{parameter}", "{value}"],  # "Assign control {number} to {parameter} with a value of {value}"
    ["Delete", "Map {number}"],  # "Remove mapping number {number}"
    ["{view_actions}"],  # "Change the view to {view_actions}"
    ["{track_creation_actions}", "{track_type}"],  # "Initiate {track_creation_actions} for a {track_type} track"
    ["track {track_number}", "{track_actions}"],  # "I want to {track_actions} track {track_number}"
    ["track {track_number}", "search device", "{audio_effect}", "enable"],  # "Enable the {audio_effect} on track {track_number}"
    ["track {track_number}", "search device", "{instrument}", "{device_type}", "disable"],  # "Disable the {instrument} with {device_type} on track {track_number}"
    ["{project_actions}"],  # "Let's {project_actions}"
    ["track {track_number}", "clip {clip_number}", "{clip_actions}"],  # "Could you {clip_actions} clip {clip_number} on track {track_number}?"
    ["track {track_number}", "duplicate", "search device", "{audio_effect}"],  # "Make a copy of track {track_number} and add {audio_effect}"
    ["track {track_number1}", "track {track_number2}", "group"],  # "Combine tracks {track_number1} and {track_number2} into a group"
    ["track {track_number}", "create_send", "search device", "{audio_effect}"],  # "Establish a send from track {track_number} to {audio_effect}"
    ["track master", "search device", "{audio_effect}"],  # "Integrate the {audio_effect} into the master track"
    ["track {track_number}", "search device", "{device_name}", "{parameter}", "{value_actions}", "{speed_modifiers}", "{value}"],  # "Adjust the {parameter} of {device_name} on track {track_number} by {value_actions} to {value} {speed_modifiers}"
    ["Map {number}", "{parameter}", "{value}"],  # "Map {number} to control {parameter} with value {value}"
    ["Delete", "Map {number}"],  # "Delete the mapping number {number}"
    ["{view_actions}"],  # "Switch to {view_actions} view"
    ["{track_creation_actions}", "{track_type}"],  # "Start {track_creation_actions} a {track_type} track"
    ["track {track_number}", "{track_actions}"],  # "Please {track_actions} on track {track_number}"
    ["track {track_number}", "search device", "{audio_effect}"],  # "Add an {audio_effect} to track {track_number}"
    ["track {track_number}", "search device", "{instrument}", "{device_type}"],  # "Load {instrument} with {device_type} into track {track_number}"
    ["{project_actions}"],  # "Carry out {project_actions}"
    ["track {track_number}", "clip {clip_number}", "{clip_actions}"],  # "Execute {clip_actions} on clip {clip_number} of track {track_number}"
    ["track {track_number}", "duplicate", "search device", "{audio_effect}"],  # "Clone track {track_number} and insert {audio_effect}"
    ["track {track_number1}", "track {track_number2}", "group"],  # "Merge tracks {track_number1} and {track_number2}"
    ["track {track_number}", "create_send", "search device", "{audio_effect}"],  # "Create a new send from track {track_number} targeting {audio_effect}"
    ["track master", "search device", "{audio_effect}"],  # "Deploy the {audio_effect} effect to the master track"
    ["track {track_number}", "search device", "{device_name}", "{parameter}", "{value_actions}", "{speed_modifiers}", "{value}"],  # "Modify {device_name} {parameter} on track {track_number} by {value_actions} to {value} {speed_modifiers}"
    ["Map {number}", "{parameter}", "{value}"],  # "Link control {number} with {parameter} set to {value}"
    ["Delete", "Map {number}"],  # "Remove the mapping identified by {number}"
    ["{view_actions}"],  # "Navigate to {view_actions} view"
    ["{track_creation_actions}", "{track_type}"],  # "Initiate the creation of a {track_type} track using {track_creation_actions}"
    ["track {track_number}", "{track_actions}"],  # "Perform {track_actions} on track {track_number}"
    ["track {track_number}", "search device", "{audio_effect}"],  # "Insert the {audio_effect} into track {track_number}"
    ["track {track_number}", "search device", "{instrument}", "{device_type}"],  # "Load a {instrument} with the following device type: {device_type} into track {track_number}"
    ["{project_actions}"],  # "Perform the project-level action: {project_actions}"
    ["track {track_number}", "clip {clip_number}", "{clip_actions}"],  # "Execute the clip action {clip_actions} on clip {clip_number} within track {track_number}"
    ["track {track_number}", "duplicate", "search device", "{audio_effect}"],  # "Duplicate the existing track {track_number} and add {audio_effect}"
    ["track {track_number1}", "track {track_number2}", "group"],  # "Group the specified tracks {track_number1} and {track_number2}"
    ["track {track_number}", "create_send", "search device", "{audio_effect}"],  # "Establish a send from track {track_number} directed to {audio_effect}"
    ["track master", "search device", "{audio_effect}"],  # "Apply {audio_effect} to the master track"
    ["track {track_number}", "search device", "{device_name}", "{parameter}", "{value_actions}", "{speed_modifiers}", "{value}"],  # "Adjust {device_name}'s {parameter} on track {track_number} by {value_actions} to reach {value} {speed_modifiers}"
    ["Map {number}", "{parameter}", "{value}"],  # "Assign control {number} to adjust {parameter} with a value setting of {value}"
    ["Delete", "Map {number}"],  # "Delete the current mapping {number}"
    ["{view_actions}"],  # "Change the interface to {view_actions}"
    ["{track_creation_actions}", "{track_type}"],  # "Please {track_creation_actions} a {track_type} track"
    ["track {track_number}", "{track_actions}"],  # "Kindly {track_actions} on track {track_number}"
    ["track {track_number}", "search device", "{audio_effect}"],  # "Add {audio_effect} effect to track {track_number}"
    ["track {track_number}", "search device", "{instrument}", "{device_type}"],  # "Load {instrument} with device type {device_type} into track {track_number}"
    ["{project_actions}"],  # "Execute the following project action: {project_actions}"
    ["track {track_number}", "clip {clip_number}", "{clip_actions}"],  # "Perform {clip_actions} on clip number {clip_number} in track {track_number}"
    ["track {track_number}", "duplicate", "search device", "{audio_effect}"],  # "Make a duplicate of track {track_number} and incorporate {audio_effect}"
    ["track {track_number1}", "track {track_number2}", "group"],  # "Group tracks numbered {track_number1} and {track_number2}"
    ["track {track_number}", "create_send", "search device", "{audio_effect}"],  # "Set up a send from track {track_number} to {audio_effect}"
    ["track master", "search device", "{audio_effect}"],  # "Deploy {audio_effect} on the master track"
    ["track {track_number}", "search device", "{device_name}", "{parameter}", "{value_actions}", "{speed_modifiers}", "{value}"],  # "Please {value_actions} {parameter} on {device_name} in track {track_number} to {value} {speed_modifiers}"
    ["Map {number}", "{parameter}", "{value}"],  # "Map control number {number} to {parameter} with value {value}"
    ["Delete", "Map {number}"],  # "Remove mapping {number}"
    ["{view_actions}"],  # "Switch view to {view_actions}"
    ["{track_creation_actions}", "{track_type}"],  # "Create a {track_type} track by {track_creation_actions}"
    ["track {track_number}", "{track_actions}"],  # "Please {track_actions} of track {track_number}"
    ["track {track_number}", "search device", "{audio_effect}"],  # "Add the {audio_effect} to the specified track {track_number}"
    ["track {track_number}", "search device", "{instrument}", "{device_type}"],  # "Load a {instrument} with the {device_type} into track {track_number}"
    ["{project_actions}"],  # "Carry out the project action: {project_actions}"
    ["track {track_number}", "clip {clip_number}", "{clip_actions}"],  # "Execute the {clip_actions} on clip {clip_number} in track {track_number}"
    ["track {track_number}", "duplicate", "search device", "{audio_effect}"],  # "Duplicate track {track_number} and add {audio_effect}"
    ["track {track_number1}", "track {track_number2}", "group"],  # "Group tracks {track_number1} & {track_number2}"
    ["track {track_number}", "create_send", "search device", "{audio_effect}"],  # "Create a send from track {track_number} targeting {audio_effect}"
    ["track master", "search device", "{audio_effect}"],  # "Add {audio_effect} effect to the master track"
    ["track {track_number}", "search device", "{device_name}", "{parameter}", "{value_actions}", "{speed_modifiers}", "{value}"],  # "Modify {device_name} {parameter} on track {track_number} by {value_actions} to {value} {speed_modifiers}"
    ["Map {number}", "{parameter}", "{value}"],  # "Assign control {number} to {parameter} with a value of {value}"
    ["Delete", "Map {number}"],  # "Delete the mapping {number}"
    ["{view_actions}"]  # "Change the current view to {view_actions}"
]

# Updated utterance templates
utterance_templates = [
    "{track_creation_actions} {track_type} track",
    "{track_actions} track {track_number}",
    "Add {audio_effect} to track {track_number}",
    "Add {instrument} with {device_type} to track {track_number}",
    "{project_actions}",
    "{clip_actions} clip {clip_number} on track {track_number}",
    "Duplicate track {track_number} and add {audio_effect}",
    "Group tracks {track_number1} and {track_number2}",
    "Create send from track {track_number} to {audio_effect}",
    "Add {audio_effect} to master track",
    "{value_actions} {device_name} {parameter} on track {track_number} to {value} {speed_modifiers}",
    "Map {number} {parameter} to {value}",
    "Delete Map {number}",
    "{view_actions}",

    # Extended Utterance Templates
    "Please {track_creation_actions} a new {track_type} track",
    "Could you {track_actions} the track number {track_number}?",
    "Kindly add the {audio_effect} effect to track {track_number}",
    "Insert a {instrument} with {device_type} into track {track_number}",
    "Execute the project action: {project_actions}",
    "Perform the {clip_actions} action on clip {clip_number} in track {track_number}",
    "Duplicate track {track_number} and apply the {audio_effect} effect",
    "Group together tracks {track_number1} and {track_number2}",
    "Set up a send from track number {track_number} to {audio_effect}",
    "Apply the {audio_effect} effect to the master track",
    "Please {value_actions} the {parameter} of {device_name} on track {track_number} to {value} {speed_modifiers}",
    "Assign control {number} to {parameter} with a value of {value}",
    "Remove mapping number {number}",
    "Change the view to {view_actions}",
    "Initiate {track_creation_actions} for a {track_type} track",
    "I want to {track_actions} track {track_number}",
    "Enable the {audio_effect} on track {track_number}",
    "Disable the {instrument} with {device_type} on track {track_number}",
    "Let's {project_actions}",
    "Could you {clip_actions} clip {clip_number} on track {track_number}?",
    "Make a copy of track {track_number} and add {audio_effect}",
    "Combine tracks {track_number1} and {track_number2} into a group",
    "Establish a send from track {track_number} to {audio_effect}",
    "Integrate the {audio_effect} into the master track",
    "Adjust the {parameter} of {device_name} on track {track_number} by {value_actions} to {value} {speed_modifiers}",
    "Map {number} to control {parameter} with value {value}",
    "Delete the mapping number {number}",
    "Switch to {view_actions} view",
    "Start {track_creation_actions} a {track_type} track",
    "Please {track_actions} on track {track_number}",
    "Add an {audio_effect} to track {track_number}",
    "Load {instrument} with {device_type} into track {track_number}",
    "Carry out {project_actions}",
    "Execute {clip_actions} on clip {clip_number} of track {track_number}",
    "Clone track {track_number} and insert {audio_effect}",
    "Merge tracks {track_number1} and {track_number2}",
    "Create a new send from track {track_number} targeting {audio_effect}",
    "Deploy the {audio_effect} effect to the master track",
    "Modify {device_name} {parameter} on track {track_number} by {value_actions} to {value} {speed_modifiers}",
    "Link control {number} with {parameter} set to {value}",
    "Remove the mapping identified by {number}",
    "Navigate to {view_actions} view",
    "Initiate the creation of a {track_type} track using {track_creation_actions}",
    "Perform {track_actions} on track {track_number}",
    "Insert the {audio_effect} into track {track_number}",
    "Load a {instrument} with the following device type: {device_type} into track {track_number}",
    "Perform the project-level action: {project_actions}",
    "Execute the clip action {clip_actions} on clip {clip_number} within track {track_number}",
    "Duplicate the existing track {track_number} and add {audio_effect}",
    "Group the specified tracks {track_number1} and {track_number2}",
    "Establish a send from track {track_number} directed to {audio_effect}",
    "Apply {audio_effect} to the master track",
    "Adjust {device_name}'s {parameter} on track {track_number} by {value_actions} to reach {value} {speed_modifiers}",
    "Assign control {number} to adjust {parameter} with a value setting of {value}",
    "Delete the current mapping {number}",
    "Change the interface to {view_actions}",
    "Please {track_creation_actions} a {track_type} track",
    "Kindly {track_actions} on track {track_number}",
    "Add {audio_effect} effect to track {track_number}",
    "Load {instrument} with device type {device_type} into track {track_number}",
    "Execute the following project action: {project_actions}",
    "Perform {clip_actions} on clip number {clip_number} in track {track_number}",
    "Make a duplicate of track {track_number} and incorporate {audio_effect}",
    "Group tracks numbered {track_number1} and {track_number2}",
    "Set up a send from track {track_number} to {audio_effect}",
    "Deploy {audio_effect} on the master track",
    "Please {value_actions} {parameter} on {device_name} in track {track_number} to {value} {speed_modifiers}",
    "Map control number {number} to {parameter} with value {value}",
    "Remove mapping {number}",
    "Switch view to {view_actions}",
    "Create a {track_type} track by {track_creation_actions}",
    "Please {track_actions} of track {track_number}",
    "Add the {audio_effect} to the specified track {track_number}",
    "Load a {instrument} with the {device_type} into track {track_number}",
    "Carry out the project action: {project_actions}",
    "Execute the {clip_actions} on clip {clip_number} in track {track_number}",
    "Duplicate track {track_number} and add {audio_effect}",
    "Group tracks {track_number1} & {track_number2}",
    "Create a send from track {track_number} targeting {audio_effect}",
    "Add {audio_effect} effect to the master track",
    "Modify {device_name} {parameter} on track {track_number} by {value_actions} to {value} {speed_modifiers}",
    "Assign control {number} to {parameter} with a value of {value}",
    "Delete the mapping {number}",
    "Change the current view to {view_actions}"
]
