import os
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load environment variables from .env
load_dotenv()

# 2. Define the OpenWeather tool
@tool
def get_current_weather(city: str) -> str:
    """Get the current weather conditions and temperature for a given city name."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY is missing from the .env file."

    url = "https://openweathermap.org"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return f"Could not fetch weather for {city}: {data.get('message', 'Unknown error')}"

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        # FIXED: Added [0] because data["weather"] is a list
        description = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        return (
            f"Weather in {data['name']}: {description.capitalize()}, "
            f"Temperature: {temp}°C (feels like {feels_like}°C), "
            f"Humidity: {humidity}%, Wind Speed: {wind_speed} m/s."
        )
    except Exception as e:
        return f"API request failed: {str(e)}"


# 3. Initialize Gemini (removed temperature=0 to avoid UserWarning)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 4. Create the Agent
tools = [get_current_weather]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful weather assistant. Always use the get_current_weather tool when asked about the weather.",
)

# 5. Run the Agent
if __name__ == "__main__":
    user_query = "What is the weather like in Jaipur and Mumbai right now? Which one is warmer?"
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]}
    )
    
    # FIXED: Use .text to print clean text without metadata dictionaries
    print(result["messages"][-1].text)
