from langchain.agents import initialize_agent, load_tools
from langchain.memory import ChatMessageHistory

from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from pyowm.commons import exceptions

class ChatAgent:

    def __init__(self):
    
        """Configure a chat chain."""

        # Load the instruct model
        llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.01)

        # Load the tool to query weather information and forecast from OpenWeatherMap
        tools = load_tools(["openweathermap-api"], llm)

        # Create an chained agent
        agent_chain = initialize_agent(tools=tools, llm=llm, verbose=True, handle_parsing_errors=True, max_iterations=4)

        # Create a prompt to be able to pass in chat history
        prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="messages")])

        self.__agent_chain = prompt | agent_chain
    

    @property
    def chat_history(self) -> ChatMessageHistory:
        return ChatMessageHistory()
    
    def respond(self, messages) -> str:
        response = "Unfortunately I was not able to find any data for you at the moment."

        try:
            formatted_response = self.__agent_chain.invoke(messages)
            response = formatted_response['output']
        except exceptions.NotFoundError:
            response = "Sorry but I was not able to detect a valid location."

        return response