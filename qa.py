from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import re
import dateutil.parser
import argparse

stopwords = set(open("stopwords.txt").read().splitlines())


def get_words(query):
    """
    >>> sorted(get_words("What is the mass of Saturn?"))
    ['mass', 'saturn']
    """

    words = set(re.split(r"[\s\.\?\!]+", query.lower()))

    return words - stopwords


def search(query):
    """
    Uses the wbsearchentities action to return entities matching a description.

    >>> search("John S. Pistole")[0]['id']
    'Q1701660'
    """

    result = requests.get(
        f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json"
    ).json()

    return result["search"]


def get_label(entity):
    """
    >>> get_label("http://www.wikidata.org/entity/Q613726")
    'yottagram'

    >>> get_label("Q613726")
    'yottagram'
    """

    entity = entity.split("/")[-1]

    result = requests.get(
        f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity}&props=labels&languages=en&format=json"
    ).json()

    return result["entities"][entity]["labels"]["en"]["value"]


def get_prop_value(entity, prop):
    """
    >>> get_prop_value("Q193", "P2067")
    '568360 yottagram'
    """
    result = requests.get(
        f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity}&props=claims&language=en&format=json"
    ).json()

    try:
        claim = result["entities"][entity]["claims"][prop][0]["mainsnak"]
    except KeyError:
        return None

    if "amount" in claim["datavalue"]["value"]:
        value = claim["datavalue"]["value"]["amount"].lstrip("+")
    elif "time" in claim["datavalue"]["value"]:
        value = claim["datavalue"]["value"]["time"]
    elif "id" in claim["datavalue"]["value"]:
        value = get_label(claim["datavalue"]["value"]["id"])
    else:
        value = claim["datavalue"]["value"]

    try:
        value += " " + get_label(claim["datavalue"]["value"]["unit"])
    except:
        pass

    return value


def search_prop(query):
    """
    Returns the property matching a query

    >>> search_prop("mass")['id']
    'P2067'
    >>> search_prop("color")['id']
    'P462'
    >>> search_prop("hair color")['id']
    'P1884'
    """
    result = requests.get(
        f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&type=property&language=en&format=json"
    ).json()

    return result["search"][0]


def answer(question):
    """
    >>> answer("What is the mass of Saturn?")
    'Mass of Saturn (sixth planet from the Sun and the second-largest planet in the Solar System, after Jupiter) is 568360 yottagram.'

    >>> answer("What is the birthdate of Barack H. Obama?")
    'Birthdate of Barack Obama (44th president of the United States) is +1961-08-04T00:00:00Z.'

    >>> answer("What is the official website of Anderson, IN?")
    'Official Website of Anderson (county seat of Madison County, Indiana, United States) is http://www.cityofanderson.com/.'
    """

    m = re.match("What is the (.*) of (.*)\?", question)

    if m:
        prop = m.group(1)
        query = m.group(2)
    else:
        return

    prop_id = search_prop(prop)["id"]

    results = search(query)

    answers = []

    for result in results:
        value = get_prop_value(result["id"], prop_id)
        if value:
            answers.append(
                f"{prop.title()} of {result['label']} ({result['description']}) is {value}."
            )

    return "\n".join(answers)


def generate(model, tokenizer, instruction, knowledge, dialog):
    if knowledge != "":
        knowledge = "[KNOWLEDGE] " + knowledge
    dialog = " EOS ".join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rename files to a standard format")
    ap.add_argument(
        "--large",
        action="store_true",
        help="Use large model",
    )
    ap.add_argument(
        "--wikidata",
        action="store_true",
        help="Download knowledge from Wikidata",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output",
    )
    args = ap.parse_args()

    if args.large:
        model_name = "microsoft/GODEL-v1_1-large-seq2seq"
    else:
        model_name = "microsoft/GODEL-v1_1-base-seq2seq"

    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    dialog = []

    while True:
        # Instruction for a chitchat task
        instruction = (
            f"Instruction: given a dialog context, you need to response empathically."
        )
        # Leave the knowldge empty
        query = input("You: ")
        dialog.append(query)

        knowledge = ""
        if args.wikidata:
            for word in get_words(query):
                for result in search(word):
                    try:
                        knowledge += f"{result['label']}: {result['description']}\n"
                    except KeyError:
                        pass
        if args.verbose:
            print(f"Knowledge: {knowledge}")

        response = generate(model, tokenizer, instruction, knowledge, dialog)
        print(f"Computer: {response}")
