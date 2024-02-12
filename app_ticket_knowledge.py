# import streamlit as st 
# from langchain_community.llms import HuggingFaceTextGenInference
import os
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import StrOutputParser

from typing import Optional

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
from sqlalchemy import create_engine
from urllib.parse import quote
import asyncio, pickle

import pandas as pd

from llm_chains.combined_ticket import CustomLLM, create_vectorstore, custom_database_chain, custom_combined_chain, custom_chain_with_history

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


conn = get_db_connection(os.environ['POSTGRE_URL'], password=os.getenv('POSTGRE_PASSWORD', None)) 

postgre_pass =  os.getenv('POSTGRE_PASSWORD', None)
if postgre_pass:
    uri1, uri2 = os.environ['POSTGRE_URL'].split(postgre_pass, 1)
    engine = create_engine(uri1 + quote(postgre_pass) + uri2)
else:
    engine = create_engine(os.environ['POSTGRE_URL'])

# emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', encode_kwargs={'normalize_embeddings': True})

emb_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=API_TOKEN, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", encode_kwargs={'normalize_embeddings': True}
)


memory_chain = custom_chain_with_history(CustomLLM(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_type='text-generation', api_token=API_TOKEN, stop=["\n<|","<|"]), emb_model, conn.cursor())

database_chain = custom_database_chain(
    CustomLLM(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_type='text-generation', 
    api_token=API_TOKEN, stop=["\n<|","<|"]), conn,
    ticket_submission_only=False # Ubah menjadi ticket_submission_only=True untuk Ebesha Ticket Submission
)

chain = custom_combined_chain(
    CustomLLM(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_type='text-generation', api_token=API_TOKEN, stop=["\n<|","<|"], max_new_tokens=4),
    database_chain, memory_chain, conn=conn, 
    ticket_submission_only=False  # Ubah menjadi ticket_submission_only=True untuk Ebesha Ticket Submission
)



@app.route('/cache_data_file', methods=['POST'])
def update_knowledge_file():
    tenant_name = request.form["tenant_name"]
    module_flag = request.form["module_flag"]
    socmed_type = request.form["socmed_type"]

    table_name = request.form.get('table_name')

    delimiter = request.form.get("delimiter", ",")
    schema = request.form.get("schema", "public")
    # redis_url = request.form['redis_url']
    file_save = request.files.getlist("file")
    # db_name = request.form["table_name"] # Name of PostgreSQL table where knowledge base is stored in ["question", "answer"] format
    # print(file_save)
    # print([x for x in file_save])
    # 1/0

    if len(file_save) < 1:
        raise Exception("Error! File Does not Exists!")
    
    if not all([transform.allowed_file(x.filename, ['pdf', 'csv', 'txt', 'docx', 'xlsx']) for x in file_save]):
        raise Exception("Error! Only 'pdf', 'csv', 'txt', 'docx', and 'xlsx' format are allowed!")

    naming = f"{module_flag}_{tenant_name}_{socmed_type}" 

    # with NamedTemporaryFile() as temp:
    #     file_save.save(temp)
    #     temp.seek(0)
    #     loader = UnstructuredFileLoader(temp.name, mode='elements').load()

    os.makedirs("temp_"+naming, exist_ok=True)

    dfs = []

    for i,x in enumerate(file_save):
        a,b = x.filename.split(".", 1)
        if b in ['pdf', 'txt', 'docx',]:
            x.save(os.path.join(f"temp_{naming}",naming+f"{a}_{i}.{b}"))
        elif b == 'csv':
            dfs.append(pd.read_csv(x, delimiter=delimiter))
        else :
            dfs.append(pd.read_excel(x))
    
    if len(dfs) > 0:
        df = pd.concat(dfs)
        
        df.to_sql(f"{schema}.{table_name}", engine, if_exists='replace')

        # unique_val = {x:[] for x in df.columns}
        # for x in df.columns:
        #     unique_val[x] += df[x].unique().tolist()

        # pickle_string = pickle.dumps(unique_val)

        # cur.execute(f'DROP TABLE IF EXISTS {schema}.{naming}_unique_val_embeddings;')
        # cur.execute(f'CREATE table {schema}.{naming}_unique_val_embeddings(data bytea);')
        # cur.execute(f'INSERT INTO {schema}.{naming}_unique_val_embeddings (data) values ({psycopg2.Binary(pickle_string)})')
        # conn.commit()

    loader = DirectoryLoader('temp_'+naming, use_multithreading=True)

    db = asyncio.run(create_vectorstore(docs=loader.load(), emb_model=emb_model))

    shutil.rmtree('temp_'+naming, ignore_errors=True)

    cur = conn.cursor()

    cur.execute(f'DROP TABLE IF EXISTS {schema}.{naming}_embeddings;')
    cur.execute(f'CREATE table {schema}.{naming}_embeddings(data bytea);')
    cur.execute(f'INSERT INTO {schema}.{naming}_embeddings (data) values ({psycopg2.Binary(db.serialize_to_bytes())})')
    conn.commit()
    
    
    # set_llm_cache(SQLAlchemyCache(engine))
    print("Success Embedded data!")

    return {"status": 200, "data" : {"response" : f"Data successfully cached to Postgre in {naming}_embeddings table!"}}






@app.route('/chatbot_combined', methods=['POST'])
async def chatbot_choose():
    global conn

    query = request.form["query"] # User input 
    tenant_name = request.form["tenant_name"]
    module_flag = request.form["module_flag"]
    socmed_type = request.form["socmed_type"]
    # data_source = request.form['data_source'] # Data source (knowledge / database)

    try :
        table_name = request.form['table_name'] #table_name hanya dibutuhkan jika ticket_submission_only=True
    except:
        table_name = ""

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

    # cur = conn.cursor()


    out = (await chain.ainvoke({"question":query, "user_id":user_id, "naming":naming, "table_name":table_name, "schema": schema})).split("\n<|")[0]

    
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