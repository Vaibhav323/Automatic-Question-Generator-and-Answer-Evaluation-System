# Automatic Question Generator and Descriptive Answer Evaluation System
## Overview
The **Automatic Question Generator and Descriptive Answer Evaluation System** is designed to streamline the process of creating educational assessments and evaluating student responses. This system leverages Natural Language Processing (NLP) and Machine Learning techniques to automatically generate questions and assess descriptive answers, making learning more effective and personalized.
## Features

- **Automatic Question Generation (AQG):** Generates various types of questions, including fill-in-the-blanks and subjective questions, based on the input text.
- **Descriptive Answer Evaluation (DAE):** Evaluates descriptive answers based on several criteria, including keyword matching, cosine similarity, grammar checking, and length checking.
- **Efficiency and Time-Saving:** Automates the creation and evaluation of assessments, saving significant time for educators.
- **Customization and Adaptability:** Customizes generated questions and evaluation criteria to meet specific learning objectives.
- **Scalability:** Can handle assessments for a wide range of class sizes, making it suitable for institutions of varying scales.
## System Architecture

1. **Automatic Question Generator (AQG):**
    
    - Utilizes NLP techniques to tokenize and tag input text.
    - Applies rules and patterns to generate relevant questions from the provided content.
2. **Descriptive Answer Evaluation (DAE):**
    
    - Evaluates student responses using methods such as keyword matching, cosine similarity, grammar checking, and more.
    - Scores are normalized and aggregated to reflect the overall quality of answers.
## Dependencies

- Python 3.x
- NLTK
- NumPy
- Pandas
- Scikit-learn

## How It Works

### Automatic Question Generator (AQG)

1. **Tokenization and POS Tagging:** The system tokenizes input sentences and tags each word based on its part of speech.
2. **Trivial Sentence Filtering:** Sentences that do not meet set criteria (e.g., length, structure) are excluded from question generation.
3. **Chunking and Parsing:** The system extracts meaningful phrases to form questions.
4. **Question Generation:** Formulates questions based on identified patterns and relationships in the text.

### Descriptive Answer Evaluation (DAE)

1. **Short Answer Evaluation:**
    
    - Uses keyword matching and cosine similarity to assess answers.
    - Checks grammar and sentence length to ensure quality.
2. **Essay Answer Evaluation:**
    
    - Analyzes longer responses using advanced NLP techniques.
    - Adjusts scores based on the relevance and structure of the content.
## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automatic-question-evaluation-system.git
cd automatic-question-evaluation-system
```   
2. Install dependencies:
```bash
pip install -r requirements.txt
```   
3. **Setup Answer Evaluation:**

- After extracting the repository files, download the "GoogleNews-vectors-negative300.bin" file from [this link](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300).
- Extract the downloaded file into the `Answer Evaluation` folder within the project directory.
4. **Run the application:**
    
    - You can run the system using either the Python script or the Jupyter Notebook:
        - To run the Python script:
        ```bash
        python main.py
        ```
            
        - To run the Jupyter Notebook:            
            ```
            jupyter notebook
            ```
            

Choose the `.py` or `.ipynb` file based on your preference; both provide the same functionalities for generating questions and evaluating answers.
    

## Future Work

- Enhance the system's adaptability with more complex question types.
- Improve evaluation accuracy with advanced NLP models like BERT or GPT.
- Integrate with learning management systems for seamless use in educational environments.

## Acknowledgments

Special thanks to Dr. Pradeep Mane and Prof. Minal Nerkar for their guidance and support.

## References

1. Rohan Bhirangi, Smita Vinit Bhoir, “Automated Question Paper Generation System,” _International Journal of Emerging Research in Management & Technology_, ISSN: 2278-9359 (Volume-5, Issue-4).
    
2. Sahar Abd El, Ali Zolait, “Automated Test Paper Generation Using Utility-Based Agent and Shuffling Algorithm,” _International Journal of Web-Based Learning and Teaching Technologies_ (Volume 14, Issue 1), January-March 2019.
    
3. Shakeel Ahmad, Sana Showkat, “Randomized Question Paper Generation,” _International Journal of Creative Research Thoughts (IJCRT)_, Volume 6, Issue 2, April 2018, ISSN: 2320-2882.
    
4. Ashokj Immanuel V, Tulasi Bomatpalli, “Framework for Automatic Examination Paper Generation System,” _International Journal of Computer Science and Technology (IJCST)_, Volume 6, Issue 1, January-March 2018, ISSN: 0976-8491 (Online), ISSN: 2229-4333 (Print).
    
5. Roshni Dwivedi, Kriti Kashyap, Pranita Dhoke, Swati Korhale, Prof. G.S. Navale, “Adaptive Question Paper Generation System,” _International Research Journal of Engineering and Technology (IRJET)_, Volume: 04 Issue: 05, May 2017.
    
6. Parth Shah, Uzair Faquih, Rutuja Devkar, Yogesh Shahare, “An Intelligent Question Paper Generator Using Randomization Algorithm,” _International Journal of Engineering Research & Technology (IJERT)_, Paper ID: IJERTV11IS040041, ISSN: 2278-0181 (Online), Volume 11, Issue 04, April 2022.
    
7. V. Paul and J. D. Pawar, “Use of Syntactic Similarity Based Similarity Matrix for Evaluating Descriptive Answer,” _2014 IEEE Sixth International Conference on Technology for Education_, Clappana, 2014, pp. 253-256.
    
8. V. U. Thompson, C. Panchev, and M. Oakes, “Performance Evaluation of Similarity Measures on Similar and Dissimilar Text Retrieval,” _2015 7th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management (IC3K)_, Lisbon, 2015, pp. 577-584.
    
9. V. Nandini, P. Uma Maheswari, “Automatic Assessment of Descriptive Answers in Online Examination System Using Semantic Relational Features,” _The Journal of Supercomputing_, 2018.
    
10. S. K. Chowdhury and R. J. R. Sree, “Dimensionality Reduction in Automated Evaluation of Descriptive Answers Through Zero Variance, Near Zero Variance, and Non-Frequent Words Techniques - A Comparison,” _2015 IEEE 9th International Conference on Intelligent Systems and Control_, Coimbatore, 2015, pp.1-6.
    
11. Pooja Kudi and Amitkumar Manekar, “Online Examination with Short Text Matching,” _IEEE Global Conference on Wireless Computing and Networking_, 2014.
    
12. Shweta M. Patil and Prof. Ms. Sonal Patil, “Evaluating the Student Descriptive Answer Using Natural Language Processing,” _International Journal of Engineering Research and Technology_, Volume 3, Issue 3.
    
13. M. Govindarajan and R. M. Chandrasekaran, “Classifier Based Text Mining for Neural Network,” _International Journal of Computer, Electrical, Automation, Control and Information Engineering_, Vol 1, Issue 3.
    
14. P. Selvi and A. K. Banerjee, “Automatic Short – Answer Grading System (ASAGS),” _InterJRI Computer Science and Networking (2010)_, Vol. 2, Issue 1, pp.18-23.