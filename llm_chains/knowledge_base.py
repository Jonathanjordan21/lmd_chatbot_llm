from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory


from operator import itemgetter
import re
from components import transform
from langchain.vectorstores import FAISS
import psycopg2
from psycopg2 import sql


def transform_chat_history(hist):
    return "\n".join(["You: "+x.content if i%2==1 else "Customer: "+x.content for i,x in enumerate(hist)])


def generate_chat_history(memory):
    # print(memory.load_memory_variables({}))
    print(memory.chat_memory)
    memory.chat_memory.messages = memory.chat_memory.messages[-15:]
    a = ["Customer: "+x.content if type(x) is HumanMessage else "You: "+x.content for x in memory.chat_memory.messages]
    print(a)
    return "\n".join(a)


def check_threshold(l, cur, emb_model):
    try :
        query = l['question']
    except:
        query = l['standalone_question']

    print(query)
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

    # d = [doc for doc,score in vecs.similarity_search_with_relevance_scores(query) if score >= 0]

    d = vecs.max_marginal_relevance_search(query, k=4)
    # d = [doc for doc,score in vecs.similarity_search_with_score(query, k=5, distance_threshold=0.1)]
    # if len(d) < 1:
    #     raise Exception("Not found!")
    print(d)
    return "\n\n".join([x.page_content for x in d])

def format_docs(d):
    print(d)
    return "\n\n".join([x.page_content for x in d])



def load_model_chain(
    model, emb_model, id2en=None, en2id=None,
    conn=None
):

    #You prefer to reply with "I don't know" rather than giving unobvious answer.
    
    template = """You are a customer service who is very careful on giving information to customers.
    
    Answer every customer question briefly in one or two sentences based only on the following context :

    {context}

    Question: {question}
    """

    chain = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    )

    memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )



    cur = conn.cursor()


    chains = loaded_memory | {"context":lambda x: check_threshold(x["question"], vectorstore), "question": lambda x: x["question"], "chat_message":lambda x:transform_chat_history(x["chat_history"])} | ANSWER_PROMPT | llm

    
        
    full_chain = {
        "context" : lambda x: check_threshold(x, cur, emb_model), "question" : itemgetter('question')
    } | prompt | model | StrOutputParser()


        
    return full_chain




def load_model_chain_large(llm, emb_model, conn, memory_chain):


    # template = """Instruct: You are a helpful and informative AI customer service assistant. Always remember to thank the customer when they say thank you and greet them when they greet you.

    # You have access to the following context of knowledge base and internal resources to find the most relevant information for the customer's needs:

    # {context}



    # A customer has made the following statement: {question}

    # Output:"""


    template = """You are a helpful and informative AI customer service assistant. Always remember to thank the customer when they say thank you and greet them when they greet you.

    You have access to the following context of knowledge base and internal resources to find the most relevant information for the customer's needs:

    {context}
    


    Always prioritize the following result of database query to find the most relevant information for the customer's needs:

    Database Tables Schema:
    {table_schema}

    {table}"""

    ANSWER_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", template), MessagesPlaceholder(variable_name='history'), ("human", "{question}")
        ]
    )


    # cur = conn.cursor()
    # ctx = RunnablePassthrough.assign(context=lambda x: check_threshold(x, conn.cursor(), emb_model),)
    # db = 
        
    # full_chain = {
    #     "context" : lambda x: check_threshold(x, cur, emb_model), "question" : itemgetter('question')
    # } | prompt | model | StrOutputParser()
        

    # full_chain =  memory_chain | {
    #     "context":lambda x: check_threshold(x, conn.cursor(), emb_model),
    #     "question": lambda x: x["standalone_question"]
    #     } | ANSWER_PROMPT | llm

    def get_table_column(cur, table_name, schema):
        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{schema}';")
        col, d  = zip(*cur.fetchall())
        d = [x if x[:2] == 'te' or x[:2] =='da' else 'real' for x in d]
        
        col_names = ", ".join([f'"{a}" {b}' for a,b in zip(col,d)])

        return f"""{table_name} ( {col_names} )\n"""

    def get_table(question, schema, cur):

        cur.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}';")

        tables = [a[0] for a in cur.fetchall() if 'knowled' not in a[0] and 'pg_' not in a[0] and '_embedd' not in a[0] and '_cache' not in a[0]]

        table_str = ""
        for table_name in tables:
            table_str += get_table_column(cur, table_name, schema)


        # PromptTemplate.from_template("""Create SQL Syntax given the below question and tables
        # Question: {question}
        # Tables: 
        # {table}""")

        return table_str

    db_prompt = PromptTemplate.from_template("""Create SQL Syntax without 'GROUP BY' function given the below question and tables schema
        Question: {question}
        Tables Schema: 
        {table_schema}""")




    db_chain = RunnablePassthrough.assign(text=db_prompt | llm |StrOutputParser()) | RunnableLambda(lambda x:parse_or_fix(conn, llm,x['text'], x['question'], x['table_schema']))

    full_chain = RunnablePassthrough.assign(table_schema=lambda x: get_table(x['question'],x['schema'],conn.cursor())) | {
        "context" : lambda x: check_threshold(x, conn.cursor(), emb_model),
        "question" : lambda x: x['question'],
        "table": db_chain,
        "history":itemgetter('history'), 
        "table_schema":itemgetter('table_schema')
        } | ANSWER_PROMPT | llm

    return RunnableWithMessageHistory(
        full_chain,
        lambda session_id : RedisChatMessageHistory(session_id),
        input_messages_key="question",
        history_messages_key='history',
    )




def load_memory_chain(llm):
    
    _template = """Instruct: Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Do not rephrase the follow up question if it is a greeting such as "Hello", "Good Morning", etc or thanking such as "Thank you", "Thanks", etc

    Chat History:
    {chat_history}

    Follow Up Input: {question}

    Output:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    chain = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),

        naming=lambda x:x['naming']
    )

    
    loaded_memory = RunnablePassthrough.assign(
        chat_history=lambda x:RunnableLambda(x['memory'].load_memory_variables) | itemgetter("history"),
    )

    return loaded_memory | chain






def parse_or_fix(conn, llm, text, question, table):#, question:str, table:str):
    cur = conn.cursor()
    # pattern = re.compile(r'\bFROM\s+(\w+)')
    # print(PromptTemplate.from_template("{question} | {table}"))
    print(question)
    text = "SELECT" + text.split(";")[0].replace('`','').split("SELECT")[-1]
    
    fixing_chain = (
        PromptTemplate.from_template(
            "Question: {question}\n\n"
            "SQL tables:\n{table}\n\n"
            "Based on the above table schema and question, fix the following SQL query:\n\n```text\n{input}\n```\nError: {error}"
            " Don't narrate, just respond with the fixed data."
            " Change the table name according to the above SQL table schema."
            " If and only if there is no aggregation function such as ['AVG', 'COUNT', 'SUM', 'MIN', 'MAX'], change the SQL query to select all columns"
            # " Don't change the after WHERE statement"
        )
        | llm
        | StrOutputParser()
    )

    e_ = None
    for _ in range(4):
        try:
            # print(text)
            text = text.replace("```","").replace("sql","").split('GROUP BY')[0]
            print(text)
            

            # pattern = r'"(.*?)"'
            # try :
            #     topic = re.search(pattern, text).group(0)
            #     print(topic)
            #     text = text.replace(topic, "*", 1)
            #     # topic = re.search(pattern, text).group(0)
            # except :
            #     topic = re.search(r'FROM(.*?)WHERE', text, re.IGNORECASE).group(0).strip().split(" ")[1]

            select_start = text.find('SELECT')
            from_start = text.find('FROM')

            # # Replace everything between SELECT and FROM with '*'
            text = text[:select_start + len('SELECT')] + ' * ' + text[from_start:]

            # text = re.sub(r'(SELECT\s+).*?(FROM)', r'\1* \2', text, flags=re.IGNORECASE)


            print("Fixed Text : ", text)
            cur.execute(text)

            out = cur.fetchall()
            # print(out)
            # if len(out) == 0:
            #     raise Exception("The result is empty! Try other way, for example try to change the string value")
            # out = "\n".join([" | ".join([str(a) for a in x]) for x in out])
            # print(out)
            # print(type(out))

            result = f"""SQL query syntax: {text}\n\nDatabase query result: {out}"""
            
            return result
            # return out,text#,agg,topic#,question
        except Exception as e:
            print(e)
            conn.commit()
            text = fixing_chain.invoke({"input": text, "error": e, "question":question, "table":table})
            e_ = e
    return ""



# def transform_output(cur):
    
    
#     table_name = table_name.replace('"', "")
#     cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = 'public';")
    
#     col_name = cur.fetchall()

#     return 
        
#         # print(col_name)
        # results = [{col_n[0] : v for col_n,v in zip(col_name,res)} for res in results]


def fetch_all_rows_and_format(cursor, schema):

    # Get a list of all tables in the database
    cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema}';")
    tables = [a[0] for a in cursor.fetchall() if 'knowled' not in a[0] and 'pg_' not in a[0] and '_embedd' not in a[0] and '_cache' not in a[0]]

    # Initialize the result string
    result_string = ""

    # Iterate through each table
    for table_name in tables:

        print(table_name)

        # Get the column names for the table
        cursor.execute(sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s;"), [table_name])
        columns = cursor.fetchall()
        column_names = [column[0] for column in columns]

        # Fetch all rows from the table
        cursor.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name)))
        rows = cursor.fetchall()

        # Format the result string for the table
        result_string += f"\n{table_name} : \n"
        result_string += " | ".join(column_names) + "\n"
        for row in rows:
            result_string += " | ".join(map(str, row)) + "\n"
    print(result_string)
    return result_string
