from user_interface import UserInterface
from entity_recognizer import EntityRecognizer
from chat_agent import ChatAgent

def program():
    
    agent = ChatAgent()
    entity_recognizer = EntityRecognizer()
    user_interface = UserInterface(agent, entity_recognizer)
    
    user_interface.run()

if __name__ == '__main__':
    program()