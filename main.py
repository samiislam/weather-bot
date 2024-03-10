import streamlit as st
from entity_recognizer import EntityRecognizer
from chat_agent import ChatAgent

def program():
    
    agent = ChatAgent()
    entity_recogniser = EntityRecognizer()
        
    st.set_page_config(page_title="Chat with the weather", page_icon="🦜")
    st.title("🦜🔗 Hi! I am your WeatherBot")

    # Initialize chat history for UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize chat history for ChatBot
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = agent.chat_history

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_query := st.chat_input(placeholder="🦜 Ask me anything about the weather!"):
        
        # Display user message in chat message container
        st.chat_message("user").markdown(user_query)
        
        # Add user message to chat history for UI
        st.session_state.messages.append({"role": "user", "content": user_query})

        if entity_recogniser.locationEntityDetected(user_query):
            # New location detected. Chat history not required
            st.session_state.chat_history.clear()
        else:
            # Find any location from the chat history since the user did not input one
            location = entity_recogniser.findLocation(st.session_state.chat_history.messages)
            
            if len(location) > 0:
                # Since there is a location in the chat history
                # only use that instead of chat history
                st.session_state.chat_history.clear()
                st.session_state.chat_history.add_user_message(location)

        # Add user message to chat history for ChatBot
        st.session_state.chat_history.add_user_message(user_query)

        # Display assistant response in chat message container
        assistant = st.chat_message("assistant")
        
        response = agent.respond({"messages": st.session_state.chat_history.messages})

        assistant.markdown(response)

        # Add assistant response to chat history for UI
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    program()