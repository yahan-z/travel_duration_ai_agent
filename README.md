# travel_duration_ai_agent
## Description
The AI agent is designed to answer questions about current travel duration between two locations for a mode of transport of the user's choosing. Large Language Models are powerful but themselves alone have knowledge cutoffs and do not have direct access to real-time data. For the specific travel duration inquiry task, the Google Maps API is integrated with an LLM using the LangChain framework to search for real-time travel duration.

## API Key Requirement
A Google Maps Python API key is required to perform a real-time Google Maps search. More details about Google Maps Python API and how to generate an API key can be found at

https://github.com/googlemaps/google-maps-services-python/

Create an .env text file with the following line:
```
MAPS_API_KEY=YOUR_API_KEY
```
Important: This key should be kept secret on your server.

## Python dependencies:
```
pip install langchain
pip install googlemaps
```
