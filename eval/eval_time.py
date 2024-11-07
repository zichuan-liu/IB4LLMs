import json
import numpy as np
import re  
  
def convert_to_seconds(time_str):  
    # Use regular expressions to find hours, minutes, and seconds  
    pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?'  
    match = re.match(pattern, time_str)  
      
    if not match:  
        return None  # Pattern not matched  
  
    # Extract hours, minutes, and seconds from the matched groups  
    hours, minutes, seconds = match.groups(default='0')  
      
    # Convert all to integers  
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)  
      
    # Calculate total seconds  
    total_seconds = hours * 3600 + minutes * 60 + seconds  
      
    return total_seconds  
  


# Read from the file and calculate the not matched rate
if __name__ == "__main__":
    # Just one example
    file_path = "../repeat_results/PAIR_vib_llama2_/result.jsonl"
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
            if len(data)==120:
                break

    time_ref = []
    for d in data:
        time_ref.append(convert_to_seconds(d['eta']))
    print(time_ref)
    print("Time: {:.3f}$\pm${:.3f}".format(np.mean(time_ref), np.std(time_ref)))