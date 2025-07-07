import spacy
nlp = spacy.load("en_core_web_sm")

# Sample review
text = "My iPhone 13 Pro's battery drains faster than my old Samsung Galaxy. Apple support was unhelpful."

# NER and sentiment
doc = nlp(text)
product_entities = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]
# Output: ['iPhone 13 Pro', 'Samsung Galaxy', 'Apple']

# Rule-based sentiment
positive_words = ["great", "love", "recommend"]
negative_words = ["drains", "unhelpful", "faster"]
sentiment = "positive" if sum(token.text in positive_words for token in doc) > sum(token.text in negative_words for token in doc) else "negative"
# Output: "negative"