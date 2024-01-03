import psycopg2
import urllib.parse as up
import pandas as pd

def get_db_connection(conn_url, password=None):
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
        password=password if password is not None else url.password,
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


def get_table_column(cur, table_name, schema):
    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{schema}';")
    col, d  = zip(*cur.fetchall())
    d = [x if x[:2] == 'te' or x[:2] =='da' else 'real' for x in d]
    
    col_names = ", ".join([f'"{a}" {b}' for a,b in zip(col,d)])

    return f""" CREATE TABLE {table_name} ( {col_names} )"""

def retrieve_fast_sql(conn, schema):
    cur = conn.cursor()

    cur.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}';")

    tables = [a[0] for a in cur.fetchall() if 'knowled' not in a[0] and 'pg_' not in a[0] and '_embedd' not in a[0] and '_cache' not in a[0]]

    table_str = ""
    for table_name in tables:
        table_str += get_table_column(cur, table_name, schema)

    return "tables:\n"+table_str[1:]+"\nquery for: "


# def get_redis_connection(redis_url, ssl=True):
#     url = up.urlparse(redis_url)
#     r = redis.Redis(
#         password=url.password,
#         host=url.hostname,
#         port=url.port, ssl=ssl
#     )

#     print("Connection to Redis successful!")
#     return r