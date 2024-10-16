# Templates for generating action orders (sequences of actions)
# Rules:
# 1. Track-related actions always start with "track {track_number}" unless it's a global action
# 2. Device-related actions always start with "search device" followed by the device name
# 3. For instruments, the order is: "search device", "{instrument}", "{device_type}", then the action
# 4. Audio effects are treated separately from instruments
# 5. Control actions include one of the speed modifiers from the speed_modifiers dictionary,
#    followed by a value between 0 and 100
# 6. When creating a track, don't mention a track number (it doesn't exist yet)
# Explanation of placeholders:
# {track_number}: The number of the track (e.g., 1, 2, 3)
# {track_action}: Any action from the track_actions list
# {audio_effect}: Any audio effect from the audio_effects list
# {device_action}: Any action from the device_actions list
# {instrument}: Any instrument from the instruments list
# {device_type}: Any device type from the device_types dictionary for the specified instrument
# {project_action}: Any action from the project_actions list
# {track_type}: Any track type from the track_types list
# {clip_number}: The number of the clip (e.g., 1, 2, 3)
# {clip_action}: Any action from the clip_actions list
# {speed_modifier}: Any speed modifier from the speed_modifiers dictionary
# {value}: A numeric value, typically between 0 and 100
# {action}: Any action that can be mapped (e.g., volume, pan, send level)
# {number}: A numeric value, typically used for macro or mapping numbers
action_order_templates = [
    ["track {track_number}", "{track_action}"],
    ["track {track_number}", "search device", "{audio_effect}", "{device_action}"],
    ["track {track_number}", "search device", "{instrument}", "{device_type}", "{device_action}"],
    ["{project_action}", "{track_type}"],
    ["track {track_number}", "clip {clip_number}", "{clip_action}"],
    ["track {track_number}", "duplicate", "search device", "{audio_effect}", "add"],
    ["track {track_number1}", "track {track_number2}", "group", "search device", "{audio_effect}", "add"],
    ["track {track_number}", "create_send", "search device", "{audio_effect}"],
    ["track {track_number}", "freeze"],
    ["track {track_number}", "unfreeze", "flatten"],
    ["create_track", "{track_type}", "search device", "{instrument}", "{device_type}"],
    ["track master", "search device", "{audio_effect}", "add"],
    ["create_return_track", "search device", "{audio_effect}", "add"],
    ["track {track_number1}", "search device", "{audio_effect}", "copy", "track {track_number2}"],
    ["group_tracks", "{instrument}"],
    ["create_rack", "search device", "{audio_effect1}", "add", "search device", "{audio_effect2}", "add", "search device", "{audio_effect3}", "add"],
    ["search device", "{instrument}", "adjust_macro", "{number}"],
    ["Control {number}", "{speed_modifier}", "{value}"],
    ["Map {number}", "{action}"],
    ["Delete", "Map {number}"]
]

# Templates for generating natural language utterances
utterance_templates = [
    "{track_action} track {track_number}",
    "{device_action} {audio_effect} on track {track_number}",
    "{device_action} {instrument} with {device_type} on track {track_number}",
    "{project_action} {track_type} track",
    "{clip_action} clip {clip_number} on track {track_number}",
    "Duplicate track {track_number} and add {audio_effect}",
    "Group tracks {track_number1} and {track_number2} with {audio_effect}",
    "Create send from track {track_number} to {audio_effect}",
    "Freeze track {track_number}",
    "Unfreeze and flatten track {track_number}",
    "Create new {track_type} track with {instrument} using {device_type}",
    "Add {audio_effect} to master track",
    "Create return track with {audio_effect}",
    "Copy {audio_effect} from track {track_number1} to track {track_number2}",
    "Group all {instrument} tracks",
    "Create rack with {audio_effect1}, {audio_effect2}, and {audio_effect3}",
    "Adjust macro {number} on {instrument}",
    "Set Control {number} to {value} {speed_modifier}",
    "Map {number} to {action}",
    "Delete Map {number}"
]



