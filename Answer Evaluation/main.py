import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import requests
from werkzeug.wrappers import Request, Response
from flask import Flask ,redirect ,url_for ,request , render_template ,flash
import sqlite3
from gensim import models
import numpy as np
import pandas as pd

csv_filename = 'Qcsv.csv'
df = pd.read_csv(csv_filename)
df['QuestionID'] = df.index + 1
print(df)

def GrammerChecker(answer):
    req = requests.get("https://api.textgears.com/check.php?text=" + answer + "&key=JmcxHCCPZ7jfXLF6")
    no_of_errors = len(req.json()['errors'])

    #print(no_of_errors)

    if no_of_errors > 5 :
        g = 0
    else:
        g = 1
    return g

#key Word matching
def KeyWordmatching(X, Y_lst):
    result = 0
    X_list = word_tokenize(X)

    # sw contains the list of stopwords
    sw = stopwords.words('english')
    l1 = []; l2 = []

    # remove stop words from string
    X_set = {w for w in X_list if not w in sw}

    for Y in Y_lst:
        Y_list = word_tokenize(Y)
        Y_set = {w for w in Y_list if not w in sw}
        # form a set containing keywords of both strings
        rvector = X_set.union(Y_set)
        for w in rvector:
            if w in X_set:
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in Y_set:
                l2.append(1)
            else:
                l2.append(0)
        c = 0

        # cosine formula
        # Compute the dot product of l1 and l2 and divide by the product of their magnitudes
        dot_product = np.dot(l1, l2)
        magnitude_l1 = np.linalg.norm(l1)
        magnitude_l2 = np.linalg.norm(l2)
        cosine = dot_product / (magnitude_l1 * magnitude_l2)

        result += cosine

    cosine = result / 3
    kval = 0
    if cosine > 90:
        kval = 1
    elif cosine > 80:
        kval = 2
    elif cosine > 60:
        kval = 3
    elif cosine > 40:
        kval = 4
    elif cosine > 20:
        kval = 5
    else:
        kval = 6
    return kval


#length of string
def CheckLenght(client_answer):
    
    client_ans = len(client_answer.split())
    #return client_ans
    kval1 = 0
    if client_ans > 50:
        kval1 = 1
    elif client_ans > 40:
        kval1 = 2
    elif client_ans > 30:
        kval1 = 3
    elif client_ans > 20:
        kval1 = 4
    elif client_ans > 10:
        kval1 = 5
    else:
        kval1 = 6
    return kval1

#Synonyam

class DocSim:
    def __init__(self, w2v_model, stopwords=None):
        self.w2v_model = w2v_model
        self.stopwords = stopwords if stopwords is not None else []

    def vectorize(self, doc: str) -> np.ndarray:
        """
        Identify the vector values for each word in the given document
        :param doc:
        :return:
        """
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim
    
    
    def calculate_similarity(self, source_doc, target_docs=None, threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if not target_docs:
            return []

        if isinstance(target_docs, str):
            target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        results = []
        result=[]
        for doc in target_docs:
            target_vec = self.vectorize(doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            result.append(sim_score)
            if sim_score > threshold:
                results.append({"score": sim_score, "doc": doc})
            # Sort results by score in desc order
            results.sort(key=lambda k: k["score"], reverse=True)

        return result



app = Flask(__name__)
email = "null"
name="null"
roll="null"


questions = df.to_dict(orient='records')

@app.route('/')
def Base_qstn_paper_set():
    return render_template('index.html',questions=questions)



@app.route('/foo', methods=['POST', 'GET'])
def foo():
    if request.method == 'POST':
        question_index = int(request.form['question_index'])
        first = request.form['answer{}'.format(question_index)]
        second = request.form['answer{}'.format(question_index + 1)]

        name = request.form['name']
        roll = request.form['roll']
        email = request.form['emailID']
        print(name)
        print(first)

        #ans = {"a1": first, "a2": second, "a3": name,"a4":roll, "email": email}

       
        
        googlenews_model_path = './GoogleNews-vectors-negative300.bin'
        stopwords_path = "./stopword.txt"
        

        model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
        with open(stopwords_path, 'r') as fh:
            stopwords = fh.read().split(",")
        ds = DocSim(model,stopwords)
        print("hello")
        #source_doc = "Python has been an object-oriented language since it existed. Because of this, creating and using classes and objects are downright easy. This chapter helps you become an expert in using Python's object-oriented programming support"
        #source_doc1=first
        #source_doc2=second
        def short(source_doc1,target_answer):
            target_docs = [target_answer]
            sim_scores = ds.calculate_similarity(source_doc1, target_docs)
            key_match=KeyWordmatching(source_doc1,target_docs)
            key_Error=GrammerChecker(source_doc1)
            
            marks1 =  ((sum(sim_scores) / len(sim_scores)) * 70)+ (10/key_match) + (20 * key_Error) 
            return marks1
            
            
        def essay(source_doc2,target_answer):
            target_docs = [target_answer]   
            sim_scores = ds.calculate_similarity(source_doc2, target_docs)
            key_match=KeyWordmatching(source_doc2,target_docs)
            key_Error=GrammerChecker(source_doc2)
            key_length=CheckLenght(source_doc2)
            marks2 =  ((sum(sim_scores) / len(sim_scores)) * 60)+ (10/key_match) + (20 * key_Error) + (10/key_length)
            return marks2

        question_row = df.iloc[question_index - 1]   
        mark1= short(first,question_row['Answer'])
        mark2= essay(second,question_row['Answer'])
        mark1=mark1/2
        mark1=round(mark1)
        mark2=round(mark2)
        mark2=mark2/2
        print(mark1)
        #print(mark2)
        #print(key_match)
        #print(sim_scores)
        with sqlite3.connect("employee.db") as con:  
            cur = con.cursor()  
            cur.execute("INSERT into Employees (name, roll, email, marks_short, marks_des ) values (?,?,?,?,?)",(name,roll,email,mark1,mark2))  
            con.commit()  

            print("After Insert in DB")
            msg = "Employee successfully Added"          
                
        #print(ans.key)
            

        
    return redirect(url_for('Recorded')) 
    #return render_template('index.html')

@app.route('/Recorded')
def Recorded():
    return render_template('base.html')


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 8000, app)