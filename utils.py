from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
import re
from components import transform

def load_cls_chain(model, tables):
  cls_chain = (
    PromptTemplate.from_template(
        f"""Given the user question below, classify it as either being about data {', '.join(tables)}""" + """ or unknown.
                                        
    Do not respond with more than one word.

    <question>
    {question}
    </question>

    Classification:"""
        )
        | model
        | StrOutputParser()
    )
  
  return cls_chain

def check_threshold(l, vecs):
    query = l['question']
    threshold = l['threshold']
    
    # doc = db

    d = [doc for doc,score in vecs.similarity_search_with_relevance_scores(query) if score >= 0]
    # d = [doc for doc,score in vecs.similarity_search_with_score(query, k=5, distance_threshold=0.1)]
    if len(d) < 1:
        raise Exception("Not found!")
    print(d)
    return "\n\n".join([x.page_content for x in d])

def format_docs(d):
    print(d)
    return "\n\n".join([x.page_content for x in d])

def load_model_chain(vecs, model, sql_model, table_name, conn):
    cur = conn.cursor()

    template = """You are a customer service who is very careful on giving information to customers.
    You prefer to reply with "I don't know" rather than giving unobvious answer.
    Answer every customer question briefly in one sentence based only on and the following context :

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    sql_template = "Question: {question}\nTable: {table_schema}\nSQL:"
    sql_prompt = PromptTemplate.from_template(sql_template)


    branch = RunnableBranch(
        (
        lambda x: any([tab in table_name.lower() for tab in x['tables'].lower().split(' ')]),  
        {
            "table_schema":lambda x: find_table_schema(table_name, cur),
            "question" : lambda x: x['question']
        }   | sql_prompt 
            | sql_model 
            | StrOutputParser()
            | {
                "out" : lambda x: parse_or_fix(conn, model, x, itemgetter('question'), itemgetter('table_schema'), table_name),# "table_name":lambda x: [tab for tab in x['tables'].lower().split(' ') if tab in x['topic'].lower()][0]
              }
            # | {"answer":lambda x_:"\n".join([" | ".join([str(a) for a in x]) for x in x_['out'][0]]), "orig_question":lambda x:x['out'][1]}
            # | ChatPromptTemplate.from_messages(
            #     [
            #         ("human", "{orig_question}"),
            #         ("ai", "{answer}"),
            #         ("system", "Generate a final response given the answer")
            #     ]
            # )
            # | model
            # | StrOutputParser()
        ) ,
        (
        {"context" : lambda x: check_threshold(x, vecs), "question" : lambda x: x['question']}
        # {"context": vecs.as_retriever() | format_docs, "question": lambda x: x['question']}
        | prompt
        | model
        | StrOutputParser()
        ),
    )

    full_chain = {
        # "context" : lambda x:x['context'],
        # "topic" : table_name,#load_cls_chain(model),
        "tables" : lambda x:x['tables'],
        "question": lambda x: x['question'], 
        'threshold' : lambda x: x['threshold']#, "context": lambda x : x['context']
        } | branch
    return full_chain

def find_table_schema(table_name, cur):
    # name = [tab for tab in x['tables'].lower().split(' ') if tab in x['topic'].lower()][0]
    # name = x['topic']
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name.lower()}' AND table_schema = 'public';")
    col_names = ", ".join([f'"{a[0]}" text' for a in cur.fetchall()])
    
    return f"{table_name.lower()} ( {col_names} )"

# def query_sql(x, cur):
    



def parse_or_fix(conn, llm, text, question, table, table_name):#, question:str, table:str):
    cur = conn.cursor()
    # pattern = re.compile(r'\bFROM\s+(\w+)')
    # print(PromptTemplate.from_template("{question} | {table}"))
    
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Question: {question}\n\n"
            "SQL table schema: {table}\n\n"
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
        # text = pattern.sub(f'FROM {str(table).split(" (")[0]}', text)
        try:
            print(table_name)
            print("Fixed Text : ", text)
            # cur.execute(s)
            # col_name = [x[0] for x in cur.fetchall()]
            # new_table_name = table
            # print(new_table_name)
            # new_table_name = 'server'
            text = transform.replace_aggregation_functions(text, table_name.lower())[-1]
            # print(text)
            cur.execute(text)

            out = cur.fetchall()
            print(out)
            if len(out) == 0:
                raise Exception("The result is empty! Try other way, for example try to change the string value")
            # out = "\n".join([" | ".join([str(a) for a in x]) for x in out])
            # print(out)
            # print(type(out))
            return out,question
        except Exception as e:
            conn.commit()
            text = fixing_chain.invoke({"input": text, "error": e, "question":question, "table":table})
            e_ = e
    raise e_