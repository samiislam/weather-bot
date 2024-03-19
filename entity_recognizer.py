from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from langchain_core.messages import BaseMessage

class EntityRecognizer:

    def __init__(self):

        # Use BERT based LLM
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")

        # Setup pipeline to use for Named Entity Recognition
        # Use aggregation strategy 'simple' to include location names like "New York" as a single location
        self.__ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='simple')

    def locationEntityDetected(self, statement: str) -> bool:
        locationFound = False

        # Use model to find a location entity
        results = self.__ner(statement)

        if len(results) > 0:
            for result in results:
                # Entity group used to extract full name for cities like "New York" and "San Francisco"
                # Only use location entity if the model is above 90% sure it's a location
                if result['entity_group'] == 'LOC' and result['score'] > 0.9:
                    locationFound = True
                    break

        return locationFound

    def findLocation(self, statements: list[BaseMessage]) -> str:
        location = ""

        # Iterate over all chat messages
        for statement in statements:
            # Use model to find a location entity in a chat message
            results = self.__ner(statement.content)
            
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