# Templates for generating action orders (sequences of actions)
# Rules:
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




