"""
ChatAgent allows the communication to the LLM and tools used.
"""

from langchain.agents import initialize_agent, load_tools
from langchain.memory import ChatMessageHistory
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

from pyowm.commons import exceptions

class ChatAgent:
    """
    ChatAgent allows the communication 
    to LLM and the OpenWeatherMap tool.
    """

    def __init__(self):
        """
        Initializes the object

        The method loads a LLM to use for the chat.
        It also loads the external tool to use for
        the weather forecast.

        A ChatAgent is then created by chaining the
        LLM, tool and a prompt that is required to save
        chat history.
        """

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
        """
        Create and return a ChatMessageHistory

        Returns
        -------
        ChatMessageHistory
            The chat message history to store
            and use.
        """
        
        return ChatMessageHistory()
    
    def respond(self, messages: dict[str, list[BaseMessage]]) -> str:
        """
        Input the user queries to the 
        LLM and retrieve a response.

        The list of message contain:
        - either a single message with a location entity
        - or 2 messages:
          - 1 with only the location entity
          - the second one is the current user query
            that does not contain a location entity

        Parameters
        ----------
        messages : dict[str, list[BaseMessage]]
            A list of chat messages

        Returns
        -------
        str
            The response from the LLM
        """

        response = "Unfortunately I was not able to find any data for you at the moment."

        try:
            formatted_response = self.__agent_chain.invoke(messages)
            response = formatted_response['output']
        except exceptions.NotFoundError:
            response = "Sorry but I was not able to detect a valid location."

        return response