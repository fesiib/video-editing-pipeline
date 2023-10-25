from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from langchain.chains import LLMChain

from LangChainPipeline.PromptTemplates.parser_index_prompt import get_parser_prompt_chat as get_parser_prompt
from LangChainPipeline.PydanticClasses.IndexedReferences import IndexedReferences

class IndexedIntentParserChain():

    def __init__(
        self,
        verbose,
    ):
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")
        self.parser = PydanticOutputParser(pydantic_object=IndexedReferences)

        self.prompt_template = get_parser_prompt({
            "format_instructions": self.parser.get_format_instructions(),
        })    

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
            output_parser=self.parser,
        )
        print("Initialized IndexedIntentParserChain")

    def run(self, command):
        if command == "":
            return IndexedReferences()
        
        try:
            references = self.chain.predict(command=command)
        except:
            print("ERROR: Failed to parse: ", command)
            return IndexedReferences()
        return references