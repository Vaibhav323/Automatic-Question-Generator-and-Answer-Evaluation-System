import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import spacy

class ObjectiveTest:
    def __init__(self, data, noOfQues):
        self.summary = data
        self.noOfQues = int(noOfQues)  # Ensure noOfQues is an integer
        self.nlp = spacy.load("en_core_web_sm")

    def get_key_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
        scores = X.sum(axis=1).flatten().tolist()[0]
        ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1]]
        return ranked_sentences[:self.noOfQues]

    def generate_trivial_question(self, sentence):
        doc = self.nlp(sentence)
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        if nouns:
            main_noun = max(nouns, key=len)
            question = sentence.replace(main_noun, "____", 1)
            return {
                "Question": question,
                "Answer": main_noun,
                "Key": len(main_noun),
                "Similar": self.answer_options(main_noun)
            }
        return None

    @staticmethod
    def answer_options(word):
        synonyms = set()
        for syn in wn.synsets(word, pos=wn.NOUN):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)[:8]

    def generate_test(self):
        key_sentences = self.get_key_sentences()
        questions = []
        answers = []
        for sentence in key_sentences:
            qa_pair = self.generate_trivial_question(sentence)
            if qa_pair:
                questions.append(qa_pair["Question"])
                answers.append(qa_pair["Answer"])
        return questions, answers
