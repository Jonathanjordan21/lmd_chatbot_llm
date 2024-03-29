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

# en2id = HuggingFacePipeline.from_model_id(
#     model_id="Helsinki-NLP/opus-mt-en-id",
#     task="text2text-generation",
#     # pipeline_kwargs={"temperature":0.},
# )

# Initialize Finetuned LLM Model for SQL generation 
sql_llm_model = HuggingFacePipeline.from_model_id(
    # model_id="jonathanjordan21/flan-alpaca-base-finetuned-lora-wikisql",
    model_id ="cssupport/t5-small-awesome-text-to-sql",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 512}
    # pipeline_kwargs={"max_new_tokens": 256,"temperature":0.},
)


conn = get_db_connection("postgresql://postgres:postgres@localhost:5433/lmd_db", password=None) # Change password if it contains `@`

table_str = retrieve_fast_sql(conn, schema='public')




@app.route('/chatbot_sql', methods=['POST'])
def chatbot_sql():
    global conn

    query = request.form["query"] # User input
    # data_source = request.form['data_source'] # Data source (knowledge / database)
    en_q = id2en(query)
    sql_query = sql_llm_model(table_str+en_q).split("WHERE")
    sql_query[1] = sql_query[1].replace('"',"'")

    print(sql_query)

    cur = conn.cursor()
    cur.execute(sql_query)
    out = cur.fetchall()
    
    return { "status" : 200, "data" : {"response":out} }

# @app.route('/chatbot_emb', methods=['POST'])
# def chatbot_emb():




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


if __name__ == '__main__':
    app.run(debug=True)