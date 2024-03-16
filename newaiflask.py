from sklearn.metrics.pairwise import cosine_similarity
from langchain.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
import pickle
import numpy as np
import os
from flask import Flask , request , jsonify
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

os.environ["GOOGLE_API_KEY"] = "AIzaSyB_wxoAPoz8C_Lf6wCE4ZXwjrc3PNMxiws"
api_key = "AIzaSyB_wxoAPoz8C_Lf6wCE4ZXwjrc3PNMxiws"
llm = GooglePalm(api_key = api_key,temprature = 0.9)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
df = pd.read_csv('data.csv')

with open("embeddings_crypto.pkl","rb") as f:
    vector_index = pickle.load(f)
    
def get_similar_chunks(query):
    encoded_query = embeddings.embed_query(query)
    similarities = []  
    for i in range(0,len(vector_index)):
        similarity = cosine_similarity(vector_index[i],encoded_query)
        similarities.append(similarity)

def get_similar_chunks(query):
    encoded_query = embeddings.embed_query(query)
    encoded_query = np.array(encoded_query)
    similarities = []  

    for i in range(len(vector_index)):
        vectors_index = np.array(vector_index[i]).reshape(1, -1)
        similarity = cosine_similarity(vectors_index, encoded_query.reshape(1, -1))
        
        similarities.append((i, similarity))  

    similarities.sort(key=lambda x: x[1], reverse=True)

    top_indices = [idx for idx, _ in similarities[:2]]
    return top_indices


conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
def get_ans(query):
    top_index = get_similar_chunks(query)
    context = df.ARTICLE[top_index[0]]+df.ARTICLE[top_index[1]]
    prompt = PromptTemplate(input_variables=["question","context"],template = "Now you are going to act as a chatbot for defi app try to provide descriptive answers and povide the answers mostly based on the context if you cant find the answer for the query within the context then if you know the answer give the answer or else say sorry at present the news is not available {query} {context}")
    input_prompt = prompt.format(query = query,context = context)
    output = conversation.run(input_prompt)
    d = {
        "output" : output,
        "titles" : [df.Title[top_index[0]],df.Title[top_index[1]]] ,
        "sources" : [df.URL[top_index[0]],df.URL[top_index[1]]]
    }
    return d
# print(get_ans("which might be a better crypto investment as of now"))

app = Flask(__name__)

@app.route("/cryptoinvest",methods=["GET"])
def return_response():
    inputtext = str(request.args["query"])
    answer = get_ans(inputtext)
    return jsonify(answer)

app.run(port = 4001)