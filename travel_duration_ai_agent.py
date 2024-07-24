#!pip install langchain
#!pip install googlemaps

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

MAPS_API_KEY = os.environ['MAPS_API_KEY']

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.base import Chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from typing import Dict, List
import json
import googlemaps
from datetime import datetime, timedelta

# initialize the LLM
llm = OpenAI(temperature=0)

validate_input_template = PromptTemplate(
    input_variables=["user_input"],
    template="""
    Validate if the following input is a proper travel duration query containing both origin and destination, if so, also provide whether it mentions travel mode:
    
    {user_input}
    
    Respond with a JSON object with the following structure:
    {{
        "is_valid": bool,
        "reason": string,
        "origin": string or null,
        "destination": string or null,
        "has_mode": bool
    }}
    
    If the input is valid, set is_valid to true, provide the extracted origin and destination, whether it has travel mode information ('driving'/'walking'/'bicycling'/'transit').
    If the input is invalid, set is_valid to false and provide a reason why it's invalid.
    """
)

validate_input_chain = LLMChain(llm=llm, prompt=validate_input_template, output_key="validation_result")

classify_destination_template = PromptTemplate(
    input_variables=["destination"],
    template="""
    Classify the following destination as either a 'general' or 'specific' place:

    Destination: {destination}

    Respond with a JSON object with the following structure:
    {{
        "classification": "general" or "specific",
        "confidence": float between 0 and 1,
        "reason": string explaining the classification
    }}

    A 'general' place might be a chain store, a type of business.
    A 'specific' place would be a specific address or a unique landmark or location.

    Examples:
    - "Walgreens" would be a "general" place
    - "McDonald's" would be a "general" place
    - "1 Market St, San Francisco, CA" would be a "specific" place
    - "Eiffel Tower" would be a "specific" place
    - "Empire State Building, 350 5th Ave, New York, NY 10118" would be a "specific" place
    """
)

classify_destination_chain = LLMChain(llm=llm, prompt=classify_destination_template, output_key="destination_classification")

extract_locations_template = PromptTemplate(
    input_variables=["user_input"],
    template="Extract the full origin, destination, and travel mode ('driving'/'walking'/'bicycling'/'transit') from this query: {user_input}\nProvide the answer in JSON format with keys 'origin', 'destination', and 'mode'."
)

extract_locations_chain = LLMChain(llm=llm, prompt=extract_locations_template, output_key="locations")

def get_travel_duration(origin, destination, mode):
    gmaps = googlemaps.Client(MAPS_API_KEY)
    now = datetime.now()
    try:
        directions_result = gmaps.directions(origin, destination, mode, departure_time=now)
    except:
        print('Route not found.')
        return (None, None)
    duration_result = directions_result[0].get('legs')[0].get('duration')
    return (duration_result['text'], (now + timedelta(seconds=duration_result['value'])).strftime('%Y/%m/%d %H:%M %p'))

# custom Chain for processing locations and getting duration
class LocationProcessingChain(Chain):
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        locations = json.loads(inputs["locations"])
        origin = locations['origin']
        destination = locations['destination']
        mode = locations['mode']
        duration, ETA = get_travel_duration(origin, destination, mode)
        return {"travel_info": f"Origin: {origin}, Destination: {destination}, Mode: {mode}, Duration: {duration}, ETA: {ETA}"}

    @property
    def input_keys(self) -> List[str]:
        return ["locations"]

    @property
    def output_keys(self) -> List[str]:
        return ["travel_info"]

final_response_template = PromptTemplate(
    input_variables=["travel_info"],
    template="Based on the following travel information: {travel_info}\nProvide a natural language response about the travel duration. If no duration or ETA found, say 'no route was found from origin to destination'. "
)

final_response_chain = LLMChain(llm=llm, prompt=final_response_template, output_key="final_response")

# combine the chains
overall_chain = SequentialChain(
    chains=[
        validate_input_chain,
        extract_locations_chain,
        LocationProcessingChain(),
        final_response_chain
    ],
    input_variables=["user_input"],
    output_variables=["validation_result", "locations", "travel_info", "final_response"],
    verbose=False
)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

def chatbot():
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print("Welcome to the Travel Duration Chatbot!")
    print("Ask me about travel durations between locations.")
    print("Type 'exit' to end the conversation.")

    need_new_input = True
    while True:
        if need_new_input:
            user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        # validate the input
        result = validate_input_chain({"user_input": user_input})
        validation_result = json.loads(result["validation_result"])
        destination_result = classify_destination_chain({"destination": validation_result['destination']})
        destination_classification_result = json.loads(destination_result['destination_classification'])
        
        if not validation_result["is_valid"]:
            # re-prompt the user
            print(f"Chatbot: {validation_result['reason']}. Could you please rephrase your question?")
            print("For example: 'How long does it take to drive from New York to Boston?'")
            continue
        elif not validation_result["has_mode"]:
            # ask for travel mode:
            travel_mode = ''
            travel_mode_mapping = {str(i+1): mode for i, mode in enumerate(['driving', 'walking', 'bicycling', 'transit'])}
            while travel_mode not in travel_mode_mapping:
                print(f"Chatbot: How do you want to get to {validation_result['destination']}? \
                \nPlease enter one of the numbers: 1 driving; 2 walking; 3 bicycling; 4 transit")
                travel_mode = input("\nYou: ").strip()
            user_input = user_input + ' travel mode: ' + travel_mode_mapping[travel_mode]
            need_new_input = False
            continue
        elif destination_classification_result['classification'] == 'general':
            user_input, need_new_input = perform_nearby_search(validation_result["origin"], validation_result["destination"], user_input)
            continue
        else:
            # use overall_chain for travel duration queries
            result = overall_chain({"user_input": user_input})
            response = result["final_response"]
        
        print(f"Chatbot: {response}")
        need_new_input = True

import requests
def perform_nearby_search(origin, destination, original_user_input):
    origin_latlong = get_latlong(origin)
    if origin_latlong:
        nearest = nearby_search(origin_latlong, destination)
        if nearest:
            confirm = input(f"Chatbot: Did you mean to go to the nearest {nearest['name']} at {nearest['address']}? (yes/no): ")
            if confirm.lower() == 'yes':
                new_user_input = original_user_input + f" {nearest['name']} at {nearest['address']}"
                return new_user_input, False
            else:
                print("Chatbot: I see. Could you please provide a more specific destination?")
                return original_user_input, True
        else:
            print(f"Chatbot: I couldn't find a nearby {validation_result['destination']}. Could you please provide a more specific destination?")
            return original_user_input, True
    else:
        print(f"Chatbot: I couldn't find a nearby {validation_result['destination']}. Could you please provide a more specific destination?")
        return original_user_input, True

def nearby_search(location, keyword):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius=1500&keyword={keyword}&key={MAPS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")
    if data['status'] == 'OK' and len(data['results']) > 0:
        nearest = data['results'][0]
        return {
            'name': nearest['name'],
            'address': nearest['vicinity'],
            'location': f"{nearest['geometry']['location']['lat']},{nearest['geometry']['location']['lng']}"
        }
    return {}

def get_latlong(address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={MAPS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        return f"{location['lat']},{location['lng']}"
    return ""

# run the chatbot
if __name__ == "__main__":
    chatbot()