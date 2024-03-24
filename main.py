"""
Main entry point of the program
"""

from user_interface import UserInterface
from entity_recognizer import EntityRecognizer
from chat_agent import ChatAgent

def program():
    """
    Program contains the main logic that is executed.

    It uses the following objects:
    1. UserInterface - to use Streamlit for a intuitive chat interface
    2. EntityRecognizer - to check and extract location entities
    3. ChatAgent - to connect and use to LLMs and Tools
    """

    # ChatAgent includes OpenWeatherMap tool
    agent = ChatAgent()

    entity_recognizer = EntityRecognizer()

    # The UserInterface component uses the EntityRecognizer
    # internally in managing session-specific chat history
    user_interface = UserInterface(agent, entity_recognizer)
    
    user_interface.run()

if __name__ == '__main__':
    program()