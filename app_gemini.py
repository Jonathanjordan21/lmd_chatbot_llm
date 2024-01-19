from utils import *

import os
import psycopg2
from psycopg2.errors import UndefinedColumn
import pickle
# import redis
import pandas as pd
import numpy as np
import urllib.parse as up
import re
# from langchain.utilities import SQLDatabase

from components.database import *
from components import eval, transform

from flask import Flask, render_template, redirect, url_for, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.vectorstores.redis import Redis
from langchain.cache import RedisSemanticCache, SQLAlchemyCache
from langchain.globals import set_llm_cache
from sqlalchemy import create_engine
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain_core.messages import AIMessage, HumanMessage
# from langchain.llms import CTransformers
import llm_chains.combined
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.memory import RedisChatMessageHistory
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os, shutil
from dotenv import load_dotenv


import psycopg2
from psycopg2 import sql



load_dotenv()


# from transformers import pipeline, AutoModelForSeq2Seq, AutoTokenizer



app = Flask(__name__) # Initialize Flask App



genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.1, convert_system_message_to_human=True)
print(llm)
emb_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",encode_kwargs={'normalize_embeddings': True})
# emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', encode_kwargs={'normalize_embeddings': True})



## set_llm_cache(RedisSemanticCache(redis_url="redis://localhost:6379", embedding=emb_model)) # Redis LLM Semantic Cache

# engine = create_engine("postgresql://postgres:postgres@localhost:5433/lmd_db")  
# set_llm_cache(SQLAlchemyCache(engine)) # Postgre LLM Cache

# r = get_redis_connection(redis_url) # Initialize redis connection

conn = get_db_connection(os.environ['POSTGRE_URL'], password=os.environ.get('POSTGRE_PASSWORD', None)) # Change password if it contains `@`


# memory = ConversationBufferMemory(
#         return_messages=True, output_key="answer", input_key="question"
# )

# memory_chain = llm_chains.knowledge_base.load_memory_chain(llm)

# # Initialize LLM chain for Database and Knowledge-base
# db_chain = llm_chains.database.load_model_chain_large(llm, llm, conn)#.with_fallbacks([llm_chains.knowledge_base.load_model_chain(llm_model, emb_model, conn)])
# knowledge_chain = llm_chains.knowledge_base.load_model_chain_large(
#     llm, emb_model,conn, memory_chain
# )#.with_fallbacks([llm_chains.database.load_model_chain(llm_model, sql_llm_model, conn)])

llm_chain = llm_chains.combined.load_model_chain_large(llm,emb_model,conn) # Knowledge base + Database. Both run concurrently
# llm_chain =  llm_chains.combined.load_model_chain_large_knowledge(llm,emb_model,conn) # Knowledge base Only


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



@app.route('/cache_data', methods=['POST']) # Endpoint to train the data
def update_knowledge(): 
    print("Initializing...")
    if request.method == 'POST':
        tenant_name = request.form["tenant_name"]
        module_flag = request.form["module_flag"]
        socmed_type = request.form["socmed_type"]
        redis_url = request.form['redis_url']
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
    
    cur = conn.cursor()

    # memory = ConversationBufferMemory(
    #     return_messages=True, output_key="answer", input_key="question"
    # )
    # RedisChatMessageHistory(naming+"_"+user_id, os.getenv('REDIS_URL', "redis://localhost:6379/0")).clear()
    

    # memory.chat_memory.messages = redis_message
    
    out = llm_chain.invoke({"question":query,'naming':naming, 'schema':schema}, config={"configurable":{"session_id":naming+"_"+user_id}}).content
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




@app.route('/chatbot_knowledge', methods=['POST'])
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
    
    cur = conn.cursor()

    # memory = ConversationBufferMemory(
    #     return_messages=True, output_key="answer", input_key="question"
    # )
    # RedisChatMessageHistory(naming+"_"+user_id, os.getenv('REDIS_URL', "redis://localhost:6379/0")).clear()
    

    # memory.chat_memory.messages = redis_message
    
    out = llm_chain.invoke({"question":query,'naming':naming, 'schema':schema}, config={"configurable":{"session_id":naming+"_"+user_id}}).content
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






@app.route('/delete', methods=['POST'])
def delete_cache():
    # tenant_name = request.form["tenant_name"]
    # module_flag = request.form["module_flag"]
    # socmed_type = request.form["socmed_type"]

    # naming = f"{module_flag}_{tenant_name}_{socmed_type}"

    # r.delete(naming)

    cur = conn.cursor()

    # Establish a connection to the PostgreSQL database
    cursor = conn.cursor()

    # Dynamic SQL to drop tables
    drop_tables_query = """
        DO $$ 
        DECLARE
            table_name_var text;
        BEGIN
            FOR table_name_var IN (SELECT table_name FROM information_schema.tables WHERE table_name LIKE %s AND table_schema = 'public') 
            LOOP
                EXECUTE 'DROP TABLE IF EXISTS ' || table_name_var || ' CASCADE';
            END LOOP;
        END $$;
    """

    # Define the pattern for table names
    table_name_pattern = '%_cache%'

    # Execute the dynamic SQL query
    cursor.execute(drop_tables_query, (table_name_pattern,))

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    # connection.close()
    # set_llm_cache(SQLAlchemyCache(engine))


    return {
        "status" : 200, "data" : {
        "response":"redis cache has been successfully deleted"
    }}


@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(404)
@app.errorhandler(400)
@app.errorhandler(500)
@app.errorhandler(403)
def not_found_error(error):
    error_data = {'status': error.code, "detail":{"error": error.name, "detail" : error.description}}
    return jsonify(error_data), error.code


# @app.after_request
# def after_request(response):
#     # Check the status code and modify the response JSON accordingly
#     err_stats = [400,404,500,403]
#     if response.status_code in err_stats:

#         response = dict({"status": response.status_code, "detail": {"detail" : response.data}})
#     return response


if __name__ == '__main__':
    app.run(debug=True)