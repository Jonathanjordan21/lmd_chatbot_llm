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

import psycopg2
from psycopg2 import sql

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




# conn_url = os.environ['conn_url']
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

    if request.method == 'POST':
        query = request.form["query"] # User input 
        tenant_name = request.form["tenant_name"]
        module_flag = request.form["module_flag"]
        socmed_type = request.form["socmed_type"]
        try :
            openai_api_key = request.form["openai_api_key"]
            llm = sql_model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        except :
            openai_api_key = None

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
        topic = cls_model.invoke({'question':query}).split(" ")[-1].lower()

        

        llm_all = load_model_chain(vecs, llm, sql_llm, topic, conn)


        res = llm_all.invoke({"question":query,"threshold":0.1,"tables":" ".join(tables)})
        

        try :
            print(res['out'][0])
            print(type(res['out'][0]))
            results = transform.decimal_to_float(res['out'][0])
            table_name=topic.split(" ")[-1].lower()
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = 'public';")
            
            col_name = cur.fetchall()
            
            results = [{col_n[0] : v for col_n,v in zip(col_name,res)} for res in results]
        except Exception as e:
            print(e)
            results = res

        return {"status" : 200, "data" : {
            "response":results, 
            # "tag_score" : float(tag['scores'][0]),
            # "q_score" : float(q_score)
        }}
    

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


