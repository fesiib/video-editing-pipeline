from llama_index import (
    KeywordTableIndex,
    VectorStoreIndex,

    SimpleDirectoryReader,
    Document,
    #LLMPredictor,
    ServiceContext
)
from llama_index.schema import MetadataMode
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI

import ast
import random
from typing import List

#langchain.llm_cache = InMemoryCache()

def get_vector_store_index():

    VIDEO_METADATA = "metadata/4LdIvyfzoGY_10.txt"

    metadata = []

    with open(VIDEO_METADATA) as f:
        documents_visual = []
        documents_text = []
        for line in f:
            interval = ast.literal_eval(line.rstrip())
            document_visual = Document(
                text=interval["synth_caption"].strip() + ", " 
                    + interval["dense_caption"].strip() + ", " 
                    + interval["action_pred"],
                metadata={
                    "start": interval["start"],
                    "end": interval["end"],
                },
                metadata_seperator=" - ",
                metadata_template="{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
            document_text = Document(
                text=interval["transcript"].strip(),
                metadata={
                    "start": interval["start"],
                    "end": interval["end"],
                },
                metadata_seperator=" - ",
                metadata_template="{value}",
                text_template="{metadata_str}:\n{content}",
            )
            documents_visual.append(document_visual)
            documents_text.append(document_text)
        

    # index = random.choice([1, 2, 3])

    # document = documents_text[index]

    # print("The LLM sees this: \n", document.get_content(metadata_mode=MetadataMode.LLM))
    # print("The Embedding model sees this: \n", document.get_content(metadata_mode=MetadataMode.EMBED))

    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents_visual + documents_text)

    print("preparing index...")

    # define LLM
    llm = OpenAI(temperature=0.1, model="gpt-4")
    service_context = ServiceContext.from_defaults(llm=llm)

    print("building index...")

    # build index
    # index = KeywordTableIndex.from_documents(nodes, service_context=service_context)

    index = VectorStoreIndex(nodes, service_context=service_context)
    print("index built!")
    return index


def try_llama_query_engine():
    index = get_vector_store_index()
    # # get response from query
    query_engine = index.as_query_engine()

    while True:
        text = input("Enter query: ")
        response = query_engine.query(text)
        print(response)


def main():
    pass

if __name__ == "__main__":
    main()