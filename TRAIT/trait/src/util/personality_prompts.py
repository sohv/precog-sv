import json
import os

def get_system_prompt(personality):
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to TRAIT directory and find the JSON file
    json_path = os.path.join(current_dir, "..", "..", "..", "personality_prompt_BFI.json")
    personality_description_data=json.load(open(json_path))
    personality_description_dict={}
    for personality_trait in personality_description_data["behavior"]:
        for hl in ["high", "low"]:
            key=f"{hl} {personality_trait}"
            personality_description_dict[key]=personality_description_data["behavior"][personality_trait][hl]
    for key in personality_description_dict:
        desc_str=""
        for i, desc in enumerate(personality_description_dict[key]):
            desc_str+=f"{i+1}. {desc} "
        personality_description_dict[key]=desc_str
        
    # Debug: print available keys if the requested personality is not found
    if personality not in personality_description_dict:
        print(f"Error: '{personality}' not found in personality descriptions.")
        print(f"Available keys: {list(personality_description_dict.keys())}")
        raise KeyError(f"Personality '{personality}' not found")
        
    system_prompt=f"You are an assistant with {personality}. Following statements are descriptions of {personality}.\n{personality_description_dict[personality]}"
    return system_prompt                