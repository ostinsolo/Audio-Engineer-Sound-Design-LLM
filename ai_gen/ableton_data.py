# ableton_data.py
from utterance_actions_template import action_order_templates, utterance_templates
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
    ["{view_actions}"]
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
    "{view_actions}"
]
