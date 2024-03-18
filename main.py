from user_interface import UserInterface
from entity_recognizer import EntityRecognizer
from chat_agent import ChatAgent

def program():
    
    # ChatAgent includes OpenWeatherMap tool
    agent = ChatAgent()

    entity_recognizer = EntityRecognizer()

    # The UserInterface component uses the EntityRecognizer
    # internally in managing session-specific chat history
    user_interface = UserInterface(agent, entity_recognizer)
    
    user_interface.run()

if __name__ == '__main__':
    program()