import numpy as np
import spacy

class SubjectiveTest:
    def __init__(self, data, noOfQues):
        self.question_patterns = [
            "Explain in detail ",
            "Define ",
            "Write a short note on ",
            "What do you mean by "
        ]
        self.summary = data
        self.noOfQues = int(noOfQues)  # Ensure noOfQues is an integer
        self.nlp = spacy.load("en_core_web_sm")

    def get_key_phrases(self, doc):
        chunks = list(doc.noun_chunks)
        key_phrases = sorted(chunks, key=lambda x: len(x), reverse=True)
        return key_phrases[:self.noOfQues]

    def generate_test(self):
        doc = self.nlp(self.summary)
        key_phrases = self.get_key_phrases(doc)
        questions = []
        answers = []
        for i in range(len(key_phrases)):
            phrase = key_phrases[i].text
            sentence = next(sent.text for sent in doc.sents if phrase in sent.text)
            question = f"{self.question_patterns[i % len(self.question_patterns)]}{phrase}."
            questions.append(question)
            answers.append(sentence)
        return questions, answers
