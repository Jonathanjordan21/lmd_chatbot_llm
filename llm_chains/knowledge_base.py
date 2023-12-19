from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
import re
from components import transform
from langchain.vectorstores import FAISS


def check_threshold(l, cur, emb_model):
    query = l['question']
    # threshold = l['threshold']
    naming = l['naming']

    print(naming)

    try :
        cur.execute(f"SELECT * FROM {naming}_embeddings;")
        vecs = [x[0] for x in cur.fetchall()][0]
        vecs = FAISS.deserialize_from_bytes(
            embeddings=emb_model, serialized=vecs
        )
    except :
        raise Exception("Please cache your data in /cache_data. Make sure your knowledge data is a available in your postgre database")
    
    # doc = db

    d = [doc for doc,score in vecs.similarity_search_with_relevance_scores(query) if score >= 0]
    # d = [doc for doc,score in vecs.similarity_search_with_score(query, k=5, distance_threshold=0.1)]
    if len(d) < 1:
        raise Exception("Not found!")
    print(d)
    return "\n\n".join([x.page_content for x in d])

def format_docs(d):
    print(d)
    return "\n\n".join([x.page_content for x in d])

def load_model_chain(
    model, emb_model, id2en, en2id, 
    conn
):

    # template = """You are a customer service who is very careful on giving information to customers.
    # You prefer to reply with "I don't know" rather than giving unobvious answer.
    # Answer every customer question briefly in one or two sentences based only on the following context :

    # {context}

    # Question: {question}
    # """

    template = """Compose the below user input to a question. 
    User Input : {question}

    Answer the composed question briefly in one sentence based on the below context :
    
    {context} 
    """

    prompt = ChatPromptTemplate.from_template(template)

    cur = conn.cursor()

        
    full_chain = {
        "context" : lambda x: id2en(check_threshold(x, cur, emb_model)), "question" : itemgetter('question') | id2en
    } | prompt | model | StrOutputParser() | en2id
        
    return full_chain