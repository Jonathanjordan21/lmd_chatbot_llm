import redis
import psycopg2
import redis
import urllib.parse as up
import pandas as pd

def get_db_connection(conn_url):
    # conn = psycopg2.connect(
    #     user="postgres",  # Replace with your PostgreSQL username
    #     host="localhost",
    #     database="lmd_db",  # Replace with your database name
    #     port="5432",
    # )

    url = up.urlparse(conn_url)
    conn = psycopg2.connect(
        database=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port
    )
    

    print("Connection to database succesfull!")
    return conn


def retrieve_all_data(conn, db_name):
    cur = conn.cursor()

    try :
        # Retrieve column names of the corresponding knowledge base table from postgre database 
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{db_name}';")
        col_names = [str(x[0]) for x in cur.fetchall()]

        # Retrieve all data from the corresponding knowledge base table
        cur.execute(f"SELECT * FROM {db_name};")
        results = cur.fetchall()
        
        # Create dataframe from SQL table dictionary
        df = pd.DataFrame([{col_name : v for col_name,v in zip(col_names,res)} for res in results])
    except Exception as err:
        print("ERROR -", err)
        cur.close()
        return pd.DataFrame()
    cur.close()
    return df


def get_redis_connection(redis_url, ssl=True):
    url = up.urlparse(redis_url)
    r = redis.Redis(
        password=url.password,
        host=url.hostname,
        port=url.port, ssl=ssl
    )

    print("Connection to Redis successful!")
    return r