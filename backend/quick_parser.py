import spacy

### This file contains functions for quickly parsing NL queries.
### Specifically, to recognize adverbs in the NL query.
### Each adverb should be classified as one of the following:
###     - time
###     - space
###     - frequency

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

# TODO: If latency allows:
def extract_adverbs_of_space_gpt3(query):
    response = {}
    return response

def extract_adverbs_of_space(query):
    tokens = nlp(query)
    space_adverbs = []
    for token in tokens:
        print(token.text, token.pos_, token.dep_, token.head.dep_)
        # if token.pos_ == "ADV" and (token.dep_ in ("advmod", "prep", "pcomp") or token.head.dep_ in ("prep", "pcomp")):
        #     space_adverbs.append((token.text, token.idx))
        if token.pos_ == "ADV" or token.pos_ == "ADP":
            space_adverbs.append({
                "text": token.text,
                "offset": token.idx
            })
    return {
        "temporal": [],
        "spatial": space_adverbs,
        "edit": [],
        "parameters": [],
    }
        

def main():
    text = "Add in hand emoji for delicious in top left corner whenever he says the word “delicious” for a split second"
    text = "Slightly boost up sizzling and chopping sounds whenever he’s not talking for an extended period."
    text = "Reduce the prominence of white in these parts of the video due to overexposure"
    space_adverbs = extract_adverbs_of_space(text)
    
    print(space_adverbs)

if __name__ == "__main__":
    main()