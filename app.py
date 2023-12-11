from utils import *

import os
import psycopg2
from psycopg2.errors import UndefinedColumn
import pickle
import redis
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

app = Flask(__name__) # Initialize Flask App

# app.template_folder = os.path.join('..', 'templates') # Reference templates folder in parent directory

# emb_tokenizer, emb_model = model.load_emb_model() # initialize embedding 
# tokenizer, alpaca_model = model.load_alpaca_model()
# id2en = model.load_id2en_model()
# llm = model.load_llm_model() # Initialize llm model



emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', encode_kwargs={'normalize_embeddings': True})

print("Models Loaded!")
# set_llm_cache(RedisSemanticCache(redis_url="redis://localhost:6379", embedding=emb_model))
engine = create_engine("postgresql://postgres:postgres@localhost:5433/lmd_db")
set_llm_cache(SQLAlchemyCache(engine))

llm = HuggingFacePipeline.from_model_id(
    model_id="declare-lab/flan-alpaca-base",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 50},
)

sql_llm = HuggingFacePipeline.from_model_id(
    model_id="jonathanjordan21/flan-alpaca-base-finetuned-lora-wikisql",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 30},
)


def set_env_var():

    #Elephant SQL postgre server
    os.environ['conn_url'] = "postgresql://wdpvftju:uoOA333yqjhTp2lzFWpelAhhyAMv6qaA@berry.db.elephantsql.com/wdpvftju"
    # Redis Upstash connection
    os.environ['redis_url'] = 'redis://default:fe62b6f869b9426f802d08cc5540810b@us1-helped-baboon-39634.upstash.io:39634'


os.environ['conn_url'] = "postgresql://wdpvftju:uoOA333yqjhTp2lzFWpelAhhyAMv6qaA@berry.db.elephantsql.com/wdpvftju"
# Redis Upstash connection
os.environ['redis_url'] = 'redis://default:fe62b6f869b9426f802d08cc5540810b@us1-helped-baboon-39634.upstash.io:39634'

conn_url = os.environ['conn_url']
# redis_url = os.environ['redis_url']

# r = get_redis_connection(redis_url) # Initialize redis connection

# conn = get_db_connection(conn_url)
conn = get_db_connection("postgresql://postgres:postgres@localhost:5433/lmd_db")



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

        conn = get_db_connection("postgresql://postgres:postgres@localhost:5433/lmd_db") # get db connection

        df = retrieve_all_data(conn, db_name) # Retrieve all data from database

        print(df)

        # tenant_df = r.get(naming) # retrieve data from redis cache


        # db = Redis.from_texts(
        #     df['answer'].tolist(),
        #     emb_model,
        #     redis_url=redis_url,
        #     index_name=naming,
        # )
        # db.write_schema(f"{naming}_schema.yaml")
        cur = conn.cursor()
        
        db = FAISS.from_texts(df['answer'].tolist(), emb_model)
        
        cur.execute(f'DROP TABLE IF EXISTS {naming}_embeddings;')
        cur.execute(f'CREATE table {naming}_embeddings(data bytea);')
        cur.execute(f'INSERT INTO {naming}_embeddings (data) values ({psycopg2.Binary(db.serialize_to_bytes())})')
        conn.commit()
        # db.save_local(naming)

        # check whether dataframe is available or not in redis cache
        # if tenant_df == None:
        #     print("tenant none")
            
        #     # Embedd data
        #     # data = transform.extract(df, desc="question", message="answer")
        #     # corpus_embeddings = transform.embed_corpus(emb_model, data)
        #     corpus_embeddings = emb_model.embed_documents(df['answer'].tolist())
            
        #     df['emb'] = corpus_embeddings.tolist() # Store corpus embeddings to dataframe
        #     df_binary = pickle.dumps(df) # create pickle binary dataframe
        # else :
        #     print("tenant available")

        #     new_df = eval.update_data(pickle.loads(tenant_df), df, emb_model) # compare the redis cache data with postgre data
        #     df_binary = pickle.dumps(new_df) # create pickle binary dataframe

        # r.set(naming, df_binary) # set redis pickle dataframe to redis cache 
        print("Success Embedded data!")
        
        # conn.close()
        
        return {"status": 200, "data" : {"response" : "Data successfully cached to Postgre!"}}





@app.route('/chatbot', methods=['POST'])
def chatbot():
    global r
    global conn

    if request.method == 'POST':
        query = request.form["query"] # User input 
        tenant_name = request.form["tenant_name"]
        module_flag = request.form["module_flag"]
        socmed_type = request.form["socmed_type"]
        try :
            openai_api_key = request.form["openai_api_key"]
        except :
            openai_api_key = None
        # try :
        #     redis_url = request.form['redis_url']
        # except :
        #     redis_url = "redis://localhost:6379"

        # th = float(request.form["th"]) # query threshold

        # try :
        #     redis_url = request.form["redis_url"]
        # except:
        #     redis_url = os.environ['redis_url']
        # try:
        #     conn_url = request.form["conn_url"]
        # except:
        #     conn_url = os.environ['conn_url']
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

        # if vecs == None: # Check whether redis cache is available or not
        except Exception as e:
            print(e)
            return {"status": 404, "detail":{"detail" : "Please cache your data in /cache_data. Make sure your knowledge data is a available in your postgre database"}}

        cur.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")

        tables = [a[0] for a in cur.fetchall() if 'knowled' not in a[0] and 'pg_' not in a[0] and '_embedd' not in a[0] and '_cache' not in a[0]]
        print(tables)
        cls_model = load_cls_chain(llm, tables)
        llm_all = load_model_chain(vecs, llm, sql_llm, cls_model, conn, openai_api_key)

        print(cls_model.invoke({'question':query}))

        res = llm_all.invoke({"question":query,"threshold":0.1,"tables":" ".join(tables)})
        
        print(res)
        # print(res)
        print(type(res))
        
        # results = transform.decimal_to_float(cur.fetchall())
        try :
            print(res['out'][0])
            print(type(res['out'][0]))
            results = transform.decimal_to_float(res['out'][0])
        except Exception as e:
            print(e)
            results = res

        return {"status" : 200, "data" : {
            "response":results, 
            # "tag_score" : float(tag['scores'][0]),
            # "q_score" : float(q_score)
        }}
    

# @app.route('/delete', methods=['POST'])
# def delete_redis():
#     tenant_name = request.form["tenant_name"]
#     module_flag = request.form["module_flag"]
#     socmed_type = request.form["socmed_type"]

#     naming = f"{module_flag}_{tenant_name}_{socmed_type}"

#     r.delete(naming)
#     return {
#         "status" : 200, "data" : {
#         "response":"redis cache has been successfully deleted"
#     }}


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


