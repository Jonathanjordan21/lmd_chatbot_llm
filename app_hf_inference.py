# import streamlit as st 
# from langchain_community.llms import HuggingFaceTextGenInference
import os
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import StrOutputParser

from llm_chains.combined_hf import CustomLLM, custom_chain_with_history

from typing import Optional

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil

# memory = ConversationBufferMemory(return_messages=True)

from flask import Flask, render_template, redirect, url_for, request, jsonify
from dotenv import load_dotenv
from components import eval, transform

import psycopg2
from psycopg2 import sql

from components.database import *

load_dotenv()

API_TOKEN = os.environ['HF_INFER_API']



app = Flask(__name__) 


conn = get_db_connection(os.environ['POSTGRE_URL'], password=os.environ.get('POSTGRE_PASSWORD', None)) 

# emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', encode_kwargs={'normalize_embeddings': True})

emb_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=API_TOKEN, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", encode_kwargs={'normalize_embeddings': True}
)


chain = custom_chain_with_history(CustomLLM(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_type='text-generation', api_token=API_TOKEN, stop=["\n<|","<|"]), emb_model, conn.cursor())



@app.route('/cache_data_file', methods=['POST'])
def update_knowledge_file():
    tenant_name = request.form["tenant_name"]
    module_flag = request.form["module_flag"]
    socmed_type = request.form["socmed_type"]
    # redis_url = request.form['redis_url']
    file_save = request.files.getlist("file")
    # db_name = request.form["table_name"] # Name of PostgreSQL table where knowledge base is stored in ["question", "answer"] format
    # print(file_save)
    # print([x for x in file_save])
    # 1/0

    if len(file_save) < 1:
        raise Exception("Error! File Does not Exists!")
    
    if not all([transform.allowed_file(x.filename, ['pdf', 'csv', 'txt']) for x in file_save]):
        raise Exception("Error! Only 'pdf', 'csv', and 'txt' format are allowed!")

    naming = f"{module_flag}_{tenant_name}_{socmed_type}" 

    # with NamedTemporaryFile() as temp:
    #     file_save.save(temp)
    #     temp.seek(0)
    #     loader = UnstructuredFileLoader(temp.name, mode='elements').load()

    os.makedirs("temp", exist_ok=True)
    
    for i,x in enumerate(file_save):
        x.save(os.path.join("temp",naming+f"{x.filename}_{i}"))

    loader = DirectoryLoader('temp', use_multithreading=True).load()

    shutil.rmtree('temp', ignore_errors=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)

    docs = splitter.split_documents(loader)

    print(docs)

    db = FAISS.from_documents(docs, emb_model)

    cur = conn.cursor()

    cur.execute(f'DROP TABLE IF EXISTS {naming}_embeddings;')
    cur.execute(f'CREATE table {naming}_embeddings(data bytea);')
    cur.execute(f'INSERT INTO {naming}_embeddings (data) values ({psycopg2.Binary(db.serialize_to_bytes())})')
    conn.commit()
    
    # set_llm_cache(SQLAlchemyCache(engine))
    print("Success Embedded data!")

    return {"status": 200, "data" : {"response" : f"Data successfully cached to Postgre in {naming}_embeddings table!"}}


@app.route('/cache_wi_knowledge', methods=['POST'])
def update_wi_knowledge():
    print("Initializing...")
    if request.method == 'POST':
        tenant_name = request.form["tenant_name"]
        module_flag = request.form["module_flag"]
        socmed_type = request.form["socmed_type"] # Name of PostgreSQL table where knowledge base is stored in ["question", "answer"] format

        naming = f"{module_flag}_{tenant_name}_{socmed_type}"

        v = open("wi_knowledge_embeddings", "rb")

        db = FAISS.deserialize_from_bytes(
            embeddings=emb_model, serialized=v
        )

        cur = conn.cursor()

        cur.execute(f'DROP TABLE IF EXISTS {naming}_embeddings;')
        cur.execute(f'CREATE table {naming}_embeddings(data bytea);')
        cur.execute(f'INSERT INTO {naming}_embeddings (data) values ({psycopg2.Binary(db.serialize_to_bytes())})')
        conn.commit()

        v.close()

        print(f"Success Embedded data to {naming}_embeddings!")
        
        return {"status": 200, "data" : {"response" : f"Data successfully cached to Postgre in {naming}_embeddings table!"}}



@app.route('/cache_data', methods=['POST']) # Endpoint to train the data
def update_knowledge(): 
    print("Initializing...")
    if request.method == 'POST':
        tenant_name = request.form["tenant_name"]
        module_flag = request.form["module_flag"]
        socmed_type = request.form["socmed_type"]
        db_name = request.form["table_name"] # Name of PostgreSQL table where knowledge base is stored in ["question", "answer"] format

        naming = f"{module_flag}_{tenant_name}_{socmed_type}" # redis cache naming convention format

        df = retrieve_all_data(conn, db_name) # Retrieve all data from database

        print(df)

        cur = conn.cursor()
        
        db = FAISS.from_texts(df['answer'].tolist(), emb_model)
        
        cur.execute(f'DROP TABLE IF EXISTS {naming}_embeddings;')
        cur.execute(f'CREATE table {naming}_embeddings(data bytea);')
        cur.execute(f'INSERT INTO {naming}_embeddings (data) values ({psycopg2.Binary(db.serialize_to_bytes())})')
        conn.commit()
        
        # set_llm_cache(SQLAlchemyCache(engine))
        print("Success Embedded data!")

        return {"status": 200, "data" : {"response" : f"Data successfully cached to Postgre in {naming}_embeddings table!"}}




@app.route('/chatbot_combined', methods=['POST'])
def chatbot_choose():
    global conn

    query = request.form["query"] # User input 
    tenant_name = request.form["tenant_name"]
    module_flag = request.form["module_flag"]
    socmed_type = request.form["socmed_type"]
    # data_source = request.form['data_source'] # Data source (knowledge / database)

    user_id = request.form['user_id']
    schema = request.form['schema']

    
    naming = f"{module_flag}_{tenant_name}_{socmed_type}"
    # naming = f"{module_flag}_{tenant_name}"
    
    # cur = conn.cursor()

    # memory = ConversationBufferMemory(
    #     return_messages=True, output_key="answer", input_key="question"
    # )
    # RedisChatMessageHistory(naming+"_"+user_id).clear()
    memory = RedisChatMessageHistory(naming+"_"+user_id, os.getenv('REDIS_URL', "redis://localhost:6379/0"))

    # memory = ConversationBufferMemory(return_messages=True)

    out = chain.invoke({"question":query, "memory":memory, "naming":naming}).split("\n<|")[0]

    
    # memory.save_context({"question":prompt}, {"output":prompt})
    # memory.chat_memory.messages = memory.chat_memory.messages[-15:]
    # # Add assistant response to chat history
    # messages.append({"role": "assistant", "content": response})
    
    

    # memory.chat_memory.messages = redis_message

    memory.add_user_message(query)
    memory.add_ai_message(out)
    
    # out = llm_chain.invoke({"question":query,'naming':naming, 'schema':schema}, config={"configurable":{"session_id":naming+"_"+user_id}}).content
    
    if len(memory.messages) > 15:
        redis_message = RedisChatMessageHistory(naming+"_"+user_id, os.getenv('REDIS_URL', "redis://localhost:6379/0")).messages[-15:]
        # print(redis_message)
        RedisChatMessageHistory(naming+"_"+user_id, os.getenv('REDIS_URL', "redis://localhost:6379/0")).clear()
        print(len(RedisChatMessageHistory(naming+"_"+user_id, os.getenv('REDIS_URL', "redis://localhost:6379/0")).messages))
        
        for x in redis_message:
            RedisChatMessageHistory(naming+"_"+user_id, os.getenv('REDIS_URL', "redis://localhost:6379/0")).add_message(x)
    print(len(RedisChatMessageHistory(naming+"_"+user_id, os.getenv('REDIS_URL', "redis://localhost:6379/0")).messages))
    # memory.save_context({"question":query},{"answer":out})
    # print(memory.chat_memory)
    
    return { "status" : 200, "data" : {"response":out} }


# React to user input

    # Add user message to chat history
    # messages.append({"role": "User", "content": prompt})
    
    # response = chain.invoke(prompt).split("\n<|")[0]

    
    # memory.save_context({"question":prompt}, {"output":prompt})
    # memory.chat_memory.messages = memory.chat_memory.messages[-15:]
    # # Add assistant response to chat history
    # messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    app.run(debug=True)