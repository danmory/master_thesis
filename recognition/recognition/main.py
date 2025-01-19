import spacy

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

## SPACY:
# def main():
#     nlp = spacy.load("ru_core_news_sm")
    
#     with open("recognition/contract.txt", "r", encoding="utf-8") as file:
#         text = file.read()
   
#     doc = nlp(text)
#     for ent in doc.ents:
#         print(ent.text, ent.label_)

# Tourch:
def main():
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained("DeepPavlov/rubert-base-cased")

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    with open("recognition/contract.txt", "r", encoding="utf-8") as file:
        text = file.read()

    entities = ner_pipeline(text)

    for entity in entities: # type: ignore
        print(entity)

if __name__ == "__main__":
    main()
