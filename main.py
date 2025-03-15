import os
import requests
import csv
import time
import replicate
from webvtt import WebVTT
from tenacity import retry, stop_after_attempt, wait_exponential

def truncate_string(string, max_length):
    return string[:max_length]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_replicate_with_retry(model, input_data):
    return replicate.run(model, input=input_data)

def meeting_summary():
    input_dir = "/Users/kashishmanoch/Documents/recordingsales5"
    max_input_length = 30000
    with open('meeting_summaries.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File',  'Participants', 'client type', 'capture', 'call type', 'use_cases', 'existing solutions', 'objections'])
        
        for filename in os.listdir(input_dir):
             if filename.endswith(".vtt"):
                file_path = os.path.join(input_dir, filename)
                
                with open(file_path, 'r') as file:
                    conversation_string = file.read()
                
                conversation_string = truncate_string(conversation_string, max_input_length)

                prompts = {
                    
                    "participants": "Please extract the name of the participants from this call including the companies they are representing, only the names and noting else. Please be very succinct, use bullet points & no bull shit. Please be very succinct, no bull shit. don't mention any extra information other than the participants. Be very very very succinct.",
                    
                    "Client type": "In the following call, please provide in bullet points the name of the client's company and their core business segment (what exactly do they do for their patients). Also mention whether it's a Small business or a mid-market business. Please be very succinct, use bullet points & no bull shit.",
                    
                    "capture": "In the following call, how does the prospective client capture the incoming lead data (mention in bullet points) from what all Lead sources? Please be very succinct, use bullet points & no bull shit. don't mention any extra information other than lead sources. Please be very succinct, no bull shit. don't mention any extra information other than the lead sources. Be very very very succinct.",

                    "call type": "In the following call, in bullet points, please determine whether this is a 1/ first sales call between LeadSquared Salesperson & prospective client, or 2/ detailed demo call where the prospective client is deeply understanding the product talking to a LeadSquared Salesperson, or 3/ LeadSquared internal team call between different team members of Leadsquared and no client presence, or 4/ A scoping call where client is requesting some additional features or 5/ 3rd party partnership call where LeadSquared is talking to prospective partner software companie or 6/ general chit chat personal call. Please be very succinct, no bull shit. don't mention any extra information other than the call type in short. Be very very very succinct.",

                    "use_cases": "In the following call, in bullet points starting with a 1-2 word summary of the point followed by its explaination, please provide the use-cases (applications/pain-points) in bullet points that the prospective customer is trying to solve with LeadSquared, please quote excerpts from the conversation as evidence. Please keep the bullet points very very very succinct, use bullet points & no bull shit, don't mention any extra information other than the use-cases.",
               
                    "existing_solutions": "In the following call, in bullet points, please extract the existing softwares that the prospective customer is using to run their business, with some information on their use-case in short succinct bullet points, don't mention any extra information only the name of the software tool and it's use-case. Please quote excerpts from the conversation as evidence",
               
                    "objections": "In the following call, in bullet points starting with a 1-2 word summary of the point followed by its explaination, please extract the objections or concerns that the prospective customer is having for not buying LeadSquared SaaS in bullet points, Please quote excerpts from the conversation. Please be very very very succinct, use bullet points & no bull shit."

                }
                
                results = [filename]
                
                for prompt in prompts.items():
                    input_data = {
                        "top_p": 0.9,
                        "prompt": f"{prompt}\n\n{conversation_string}",
                        "min_tokens": 0,
                        "max_tokens": 210,
                        "temperature": 0.3,
                        #"prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant tasked with extracting specific information about {prompt_key} from the given conversation. Provide your response in a concise, bullet-pointed list. Each point should be brief and directly relevant to the requested information. Do not include any additional explanations or context beyond the bullet points.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant, if you find out that the requested information in the prompt is not available in the call recording transcript, then simply state no relecant information found. so please be honest and direct. <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                        "presence_penalty": 1.15,
                        "length_penalty": 2
                    }
                    
                    output = run_replicate_with_retry("mistralai/mixtral-8x7b-instruct-v0.1", input_data)
                    
                    results.append("".join(output))
                    time.sleep(6)
                
                writer.writerow(results)
                csvfile.flush()  # Force write to disk
                print(f"Processed {filename}")

    print("Meeting analysis generated successfully.")

if __name__ == "__main__":
    meeting_summary()





'''
input = {
    "top_p": 0.9,
    "prompt": "From the following call, extract the name of the prospective client and their business type:  ",
    "min_tokens": 0,
    "temperature": 0.6,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're the ai agent of the sales head of US Healthcare for LeadSqaured, a US focused CRM SaaS company competing with HubSpot & Salesforce <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "presence_penalty": 1.15
}

for event in replicate.stream(
    "meta/meta-llama-3-70b-instruct",
    input=input
):
    print(event, end="done dana done")



'''
