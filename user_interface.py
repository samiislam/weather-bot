"""
UserInterface allows using Streamlit as the Chat Agent interface
"""

import streamlit as st
from chat_agent import ChatAgent
from entity_recognizer import EntityRecognizer

class UserInterface:
    """
    UserInterface uses Streamlit as the Chat Agent interface
    """

    def __init__(self, agent: ChatAgent, entity_recognizer: EntityRecognizer):
        """
        Initializes the object

        The method saves a reference to the ChatAgent
        and EntityRecognizer objects for use

        Parameters
        ----------
        agent : ChatAgent
            The ChatAgent used by the UI to
            obtain Chat history and retrieve
            a response from an user query

        entity_recognizer: EntityRecognizer
            The EntityRecognizer used to identify
            and extract location entities present
            in user queries
        """

        self.__agent = agent
        self.__entity_recognizer = entity_recognizer

        st.set_page_config(page_title="Chat with the weather", page_icon="🦜")
        st.title("🦜🔗 Hi! I am your WeatherBot")

        # Initialize chat history for UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize chat history for ChatBot
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = self.__agent.chat_history

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    def run(self):
        """
        Run the main loop to allow for
        user input and agent response.
        The user interface is updated
        accordingly with the chat history.

        The strategy to use only the last
        known location, when the user does 
        not specify one, is defined here.
        """

        # React to user input
        if user_query := st.chat_input(placeholder="🦜 Ask me anything about the weather!"):
        
            # Display user message in chat message container
            st.chat_message("user").markdown(user_query)
            
            # Add user message to chat history for UI
            st.session_state.messages.append({"role": "user", "content": user_query})

            if self.__entity_recognizer.locationEntityDetected(user_query):
                # New location detected. Chat history not required
                st.session_state.chat_history.clear()
            else:
                # Find any location from the chat history since the user did not input one
                location = self.__entity_recognizer.findLocation(st.session_state.chat_history.messages)
                
                if len(location) > 0:
                    # Since there is a location in the chat history
                    # only use that instead of chat history
                    st.session_state.chat_history.clear()
                    st.session_state.chat_history.add_user_message(location)

            # Add user message to chat history for ChatBot
            st.session_state.chat_history.add_user_message(user_query)

            # Display assistant response in chat message container
            assistant = st.chat_message("assistant")
            
            # Get the agent response
            response = self.__agent.respond({"messages": st.session_state.chat_history.messages})

            # Show the agent response
            assistant.markdown(response)

            # Add assistant response to chat history for UI
            st.session_state.messages.append({"role": "assistant", "content": response})