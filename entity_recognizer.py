from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from langchain_core.messages import BaseMessage

class EntityRecognizer:

    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")

        self.__ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='simple')

    def locationEntityDetected(self, statement: str) -> bool:
        locationFound = False

        results = self.__ner(statement)

        if len(results) > 0:
            for result in results:
                if result['entity_group'] == 'LOC' and result['score'] > 0.9:
                    locationFound = True
                    break

        return locationFound

    def findLocation(self, statements: list[BaseMessage]) -> str:
        location = ""

        for statement in statements:
            results = self.__ner(statement.content)
            
            if len(results) > 0:
                for result in results:
                    if result['entity_group'] == 'LOC' and result['score'] > 0.9:
                        location = result['word']
                        break

            if len(location) > 0:
                break

        return location