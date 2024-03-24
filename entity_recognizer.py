"""
EntityRecognizer allows querying and extracting a location
name based on a Named Entity Recognition model
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from langchain_core.messages import BaseMessage

class EntityRecognizer:
    """
    EntityRecognizer allows querying and extracting a location
    name based on a BERT-based Named Entity Recognition model
    """

    def __init__(self):
        """
        Initializes the object

        The load pretrained BERT-base model
        to use as a tokenizer and LLM. It
        then creates a pipeline with a simple
        strategy to use the model.
        """

        # Use BERT based LLM
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")

        # Setup pipeline to use for Named Entity Recognition
        # Use aggregation strategy 'simple' to include location names like "New York" as a single location
        self.__ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='simple')

    def locationEntityDetected(self, user_message: str) -> bool:
        """
        Find a location in a user message

        Parameters
        ----------
        user_message : str
            The user message

        Returns
        -------
        bool
            True if location is found, False otherwise
        """

        locationFound = False

        # Use model to find a location entity
        results = self.__ner(user_message)

        if len(results) > 0:
            for result in results:
                # Entity group used to extract full name for cities like "New York" and "San Francisco"
                # Only use location entity if the model is above 90% sure it's a location
                if result['entity_group'] == 'LOC' and result['score'] > 0.9:
                    locationFound = True
                    break

        return locationFound

    def findLocation(self, user_messages: list[BaseMessage]) -> str:
        """
        Find a location in a user message and return it.
        Only keep searching till the first location is found.

        Parameters
        ----------
        user_messages :  list[BaseMessage]
            A list of user messages

        Returns
        -------
        str
            A location if found. Otherwise an
            empty string
        """

        location = ""

        # Iterate over all chat messages
        for user_message in user_messages:
            # Use model to find a location entity in a chat message
            results = self.__ner(user_message.content)
            
            if len(results) > 0:
                for result in results:
                    # Entity group used to extract full name for cities like "New York" and "San Francisco"
                    # Only use location entity if the model is above 90% sure it's a location
                    if result['entity_group'] == 'LOC' and result['score'] > 0.9:
                        location = result['word']

                        # Only use the first location found in a chat message
                        break

            if len(location) > 0:
                # Only use the first location found in a chat message in the list of messages
                break

        return location