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
# from langchain.llms import LlamaCpp
# from langchain.llms import CTransformers
import llm_chains.database, llm_chains.knowledge_base

# from transformers import pipeline, AutoModelForSeq2Seq, AutoTokenizer

import psycopg2
from psycopg2 import sql

app = Flask(__name__) # Initialize Flask App


# Initialize Translation Model
id2en = HuggingFacePipeline.from_model_id(
    model_id="Helsinki-NLP/opus-mt-id-en",
    task="text2text-generation",
    # pipeline_kwargs={"temperature":0.},
)

en2id = HuggingFacePipeline.from_model_id(
    model_id="Helsinki-NLP/opus-mt-en-id",
    task="text2text-generation",
    # pipeline_kwargs={"temperature":0.},
)

# Initialize Finetuned LLM Model for SQL generation 
sql_llm_model = HuggingFacePipeline.from_model_id(
    # model_id="jonathanjordan21/flan-alpaca-base-finetuned-lora-wikisql",
    model_id ="jonathanjordan21/mt5-base-finetuned-lora-sql",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 60,"temperature":0.},
)

# sql_llm_model = LlamaCpp(
#     model_path=os.path.join("\\",*"\\Users\\jonat\\.cache\\huggingface\\hub\\models--TheBloke--stablelm-zephyr-3b-GGUF\\blobs\\74b2613b6e89d904a2ea38d56d233e4d2ca2fe663844b2e7aa90e769d359061b".split("\\")),
#     # temperature=0.75,
#     max_tokens=60,
#     top_p=1,
# )

# sql_llm_model = CTransformers(
#     model="C:\\Users\\jonat\\.cache\\huggingface\\hub\\models--TheBloke--TinyLlama-1.1B-Chat-v0.3-GGUF\\snapshots\\b32046744d93031a26c8e925de2c8932c305f7b9\\tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
#     config = {'max_new_tokens': 256, 'repetition_penalty': 1.1}
# )


# Initialize LLM Model for Retrieval Augmented Generation (RAG) of Knowledge-base
llm_model = HuggingFacePipeline.from_model_id(
    model_id="declare-lab/flan-alpaca-base",
    # model_id="jonathanjordan21/mt5-base-finetuned-lora-LaMini-instruction",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 256}#, "temperature":0.},
)

# llm_model = sql_llm_model


# Initialize Embedding Model
emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', encode_kwargs={'normalize_embeddings': True})

## set_llm_cache(RedisSemanticCache(redis_url="redis://localhost:6379", embedding=emb_model)) # Redis LLM Semantic Cache

engine = create_engine("postgresql://postgres:postgres@localhost:5433/lmd_db")  
set_llm_cache(SQLAlchemyCache(engine)) # Postgre LLM Cache

# r = get_redis_connection(redis_url) # Initialize redis connection

conn = get_db_connection("postgresql://postgres:postgres@localhost:5433/lmd_db", password=None) # Change password if it contains `@`


# Initialize LLM chain for Database and Knowledge-base
db_chain = llm_chains.database.load_model_chain(llm_model, sql_llm_model, conn)#.with_fallbacks([llm_chains.knowledge_base.load_model_chain(llm_model, emb_model, conn)])
knowledge_chain = llm_chains.knowledge_base.load_model_chain(
    llm_model, emb_model, id2en, en2id, 
    conn
)#.with_fallbacks([llm_chains.database.load_model_chain(llm_model, sql_llm_model, conn)])




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
        
        set_llm_cache(SQLAlchemyCache(engine))
        print("Success Embedded data!")

        return {"status": 200, "data" : {"response" : "Data successfully cached to Postgre!"}}





@app.route('/chatbot', methods=['POST'])
def chatbot():
    global conn
    # global llm
    # global sql_model

    if request.method == 'POST':
        query = request.form["query"] # User input 
        tenant_name = request.form["tenant_name"]
        module_flag = request.form["module_flag"]
        socmed_type = request.form["socmed_type"]
        try :
            openai_api_key = request.form["openai_api_key"]
            llm = sql_llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        except :
            openai_api_key = None
            llm = llm_model
            sql_llm = sql_llm_model
        
        # if openai_api_key != None:
        #     llm = sql_model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

        cur = conn.cursor()

        naming = f"{module_flag}_{tenant_name}_{socmed_type}" # redis cache naming convention 

        
        

        # vecs = FAISS.load_local(naming, emb_model)

        # vecs = Redis.from_existing_index(
        #     emb_model,
        #     redis_url=redis_url,
        #     index_name=naming,
        #     schema=f"{naming}_schema.yaml"
        # )

        try :
            cur.execute(f"SELECT * FROM {naming}_embeddings;")
            vecs = [x[0] for x in cur.fetchall()][0]
            vecs = FAISS.deserialize_from_bytes(
                embeddings=emb_model, serialized=vecs
            )

        except Exception as e:
            print(e)
            return {"status": 404, "detail":{"detail" : "Please cache your data in /cache_data. Make sure your knowledge data is a available in your postgre database"}}

        cur.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")

        tables = [a[0] for a in cur.fetchall() if 'knowled' not in a[0] and 'pg_' not in a[0] and '_embedd' not in a[0] and '_cache' not in a[0]]
        print(tables)
        cls_model = load_cls_chain(llm, tables)

        query = id2en(query)

        topic = cls_model.invoke({'question':query}).split(" ")[-1].lower()

        llm_all = load_model_chain(vecs, llm, sql_llm, topic, conn)

        res = llm_all.invoke({"question":query,"threshold":0.1,"tables":" ".join(tables)})
        
        try :
            print(res['out'][0])
            print(type(res['out'][0]))
            res, agg = res['out']
            print(agg)
            if agg == None:
                results = transform.decimal_to_float(res)
                table_name=topic.split(" ")[-1].lower()
                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = 'public';")
                
                col_name = cur.fetchall()
                
                results = [{col_n[0] : v for col_n,v in zip(col_name,res)} for res in results]
            else :
                results = llm(f"""Generate final response based on the following question and the answer\n\nQUESTION:\n{query}\n\nANSWER:\n{agg}:{' '.join([str(x[0]) for x in res])}""")
                results = en2id(results)
                print(results)
        except Exception as e:
            print(e)
            results = en2id(res)

        return {"status" : 200, "data" : {
            "response":results, 
            # "tag_score" : float(tag['scores'][0]),
            # "q_score" : float(q_score)
        }}






def transform_output(res, cur,question, model):
    res, agg,table_name = res
    cur = conn.cursor()
    if agg == None:
        results = transform.decimal_to_float(res)
        table_name = table_name.replace('"', "")
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = 'public';")
        
        col_name = cur.fetchall()
        
        print(col_name)
        results = [{col_n[0] : v for col_n,v in zip(col_name,res)} for res in results]

        return results

        # res = [" | ".join([f"{col_n[0]} : {v}" for col_n,v in zip(col_name,res)]) for res in results]
        # print(res)
    
    # results = llm(f"""Generate final response based on the following question and the answer\n\nQUESTION:\n{query}\n\nANSWER:\n{agg}:{' '.join([str(x[0]) for x in res])}""")
    # print(results)
    # s = "\n"
    else :
        answer = '\n'.join([str(x[0]) for x in res]) 
        prompt = PromptTemplate.from_template("Generate final response based on the below question and the answer\n\nQUESTION:\n{query}\n\n"+f"ANSWER:\n{answer}")

        return (prompt | model | en2id).invoke({"query":question, "answer":answer})


@app.route('/chatbot_choose', methods=['POST'])
def chatbot_choose():
    global conn

    query = request.form["query"] # User input 
    tenant_name = request.form["tenant_name"]
    module_flag = request.form["module_flag"]
    socmed_type = request.form["socmed_type"]
    data_source = request.form['data_source'] # Data source (knowledge / database)
    try :
        openai_api_key = request.form["openai_api_key"]
        llm = sql_llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    except :
        openai_api_key = None
        llm = llm_model
        sql_llm = sql_llm_model

    
    naming = f"{module_flag}_{tenant_name}_{socmed_type}"
    
    cur = conn.cursor()

    if data_source == 'database':
        try :
            out = db_chain.invoke({"question":query, "naming":naming})
            out = transform_output(out['out'], cur,query, llm)
        except Exception as e:
            print(e)
            out = knowledge_chain.invoke({"question":query,'naming':naming})
    else :
        try :
            out = knowledge_chain.invoke({"question":query,'naming':naming})
        except Exception as e :
            print(e)
            out = db_chain.invoke({"question":query, "naming":naming})
            out = transform_output(out['out'], cur,query, llm)
    
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
    set_llm_cache(SQLAlchemyCache(engine))


    return {
        "status" : 200, "data" : {
        "response":"redis cache has been successfully deleted"
    }}


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


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


