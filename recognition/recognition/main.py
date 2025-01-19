import spacy

def main():
    nlp = spacy.load("ru_core_news_sm")
    
    with open("recognition/contract.txt", "r", encoding="utf-8") as file:
        text = file.read()
   
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.label_)

if __name__ == "__main__":
    main()
