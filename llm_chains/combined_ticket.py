from typing import Any, List, Mapping, Optional, Dict

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Literal
from langchain.prompts import PromptTemplate
from operator import itemgetter

from langchain.memory import RedisChatMessageHistory
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os, requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnableBranch
import pickle, asyncio, traceback, aiohttp

# os.environ['FAISS_NO_AVX2'] = '1'
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote

load_dotenv()



async def create_vectorstore(emb_model, docs):
    # API_TOKEN = os.getenv('HF_INFER_API')
    
    # loader = os.getenv('knowledge_base')
    # web_loader = load_web("https://lintasmediadanawa.com")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    
    # docs = splitter.create_documents([loader]+web_loader)
    # docs = splitter.create_documents([loader])
    print(len(docs))
    # emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', encode_kwargs={'normalize_embeddings': True})

    # emb_model = HuggingFaceInferenceAPIEmbeddings(
    #     api_key=API_TOKEN, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", encode_kwargs={'normalize_embeddings': True}
    # )
    
    # async def add_docs(d):
    #     db.aadd_documents(await splitter.atransform_documents([d]))
        
    db = await FAISS.afrom_documents(docs, emb_model)

    f = pickle.load(open("wi_knowledge.dat", "rb"))

    print("Docs len :", len(f))

    tasks = []

    for d in f:
        tasks.append(db.aadd_documents(await splitter.atransform_documents([d])))

    await asyncio.gather(*tasks)
    
    return db
    


def custom_database_chain(llm, conn):

    cur = conn.cursor()

    prompt = PromptTemplate.from_template("""<s><INST> You have access to a postgresql database named "{table_name}" with the following columns and datatype:
    {columns}

    The following are unique values for the prominent categorical columns:
    {unique} 

    Create SQL query in format for the following question

    <question>
    {question}
    </question>

    the output must follow the following format:

    ```sql
    ```

    </s></INST>
    """)

    postgre_pass =  os.getenv('POSTGRE_PASSWORD', None)
    if postgre_pass:
        uri1, uri2 = os.environ['POSTGRE_URL'].split(postgre_pass, 1)
        engine = create_engine(uri1 + quote(postgre_pass) + uri2)
    else:
        engine = create_engine(os.environ['POSTGRE_URL'])


    def col_to_str(cols):
        print(type(cols['engine']))
        try :
            query = f"SELECT * FROM {cols['table_name']}"
            df = pd.read_sql_query(query, engine)
            unique_val = {}
            for x in df.columns:
                unique_val[x] = df[x].unique().tolist()
        except :
            raise Exception("Please cache your data in /cache_data. Make sure your knowledge data is a available in your postgre database")

        return "\n".join([f"{i}. {col}" for i,col in enumerate(unique_val.keys())])

    def unique_to_str(uniques):
        try :
            query = f"SELECT * FROM {uniques['table_name']}"
            df = pd.read_sql_query(query, engine)
            unique_val = {}
            for x in df.columns:
                unique_val[x] = df[x].unique().tolist()
        except :
            raise Exception("Please cache your data in /cache_data. Make sure your knowledge data is a available in your postgre database")

        return "\n".join([f"{i}. {k}: {v}" for i,(k,v) in enumerate(unique_val.items()) if len(v) < 10])


    def sqlparser(text, cur):
        print(text)
        text = text.split("`sql")[-1].split("`")[0].replace("\\","")
    
        err_chain = PromptTemplate.from_template("<s><INST>fix the following sql query with the following error:\n\nSQL: {sql_query}\n\nError: {error}\n\n\Fixed Code: </INST></s>") | llm

        for _ in range(7):
            try :
                print(text)
                cur.execute(text)
                return cur.fetchall()
            except Exception as e:
                print(str(traceback.format_exc(limit=2)))
                text = err_chain.invoke({"sql_query":text, "error":str(e)})
                conn.commit()


    final_chain = PromptTemplate.from_template("<s><INST>Generate final response from the following Question and answer\n\n<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>\n\n\nFinal Response:\n\n</INST></s>") | llm

    chain = RunnablePassthrough.assign(columns = col_to_str, unique= unique_to_str) | {"answer" : prompt | llm | RunnableLambda(lambda x:sqlparser(x, cur)), "question":lambda x:x['question']} | final_chain

    return chain



def custom_combined_chain(llm, df_chain, memory_chain, conn,):

    # prompt = PromptTemplate.from_template("""<s><INST> Given the following question, classify it as either being more relevant with a dataframe object of ticket submissions' history or several documents of user guide and general knowledge:

    # <question>
    # {question}
    # </question>

    # Respond with ONLY one word either "ticket" or "knowledge"
    
    # </s></INST>""")

    cur = conn.cursor()

    prompt = PromptTemplate.from_template("""<s><INST> You have access to the following data sources:
    1. Dataframe : use this data source to retrieve anything about ticket submission history
    2. Documents : use this data source to retrieve anything related to user guide and work instruction
    
    <question>
    {question}
    </question>
    
    
    Respond with ONLY one word either "dataframe" or "documents"
    
    </s></INST>
    """)
    
    # def route(info):
    #     if 'ticket' in info['topic']:
    #         return df_chain
    #     else:
    #         return memory_chain

    
    # full_chain = RunnablePassthrough.assign(topic= (prompt | llm)) | RunnableLambda(route)
    # combined_chain = prompt | llm


    prompt_table = PromptTemplate.from_template("""<s><INST> You have access to the following tables and columns of database:
    {table}
    
    <question>
    {question}
    </question>
    
    
    Respond with ONLY one word
    
    </s></INST>
    """)

    def table_chain(ex):
        
        cur.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{ex['schema']}';")
        tables = [x[0] for x in cur.fetchall() if 'pg_' not in x[0]]
        q = {}
        for x in tables:
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{x}';")
            q[x] = [a[0] for a in cur.fetchall()]
        
        table_str = "\n".join( [ f"{i+1}. {k}: {v}" for i,(k,v) in enumerate(q.items()) ] )

        prompt_table = prompt_table.partial(table=table_str)

        return prompt_table | llm

    return RunnablePassthrough.assign(topic=prompt | llm) | RunnableBranch( (lambda x: "dataframe" in x['topic'].lower(), RunnablePassthrough.assign(table_name=lambda x: table_chain(x)) | df_chain), memory_chain )



def custom_chain_with_history(llm, emb_model, cur):

    prompt = PromptTemplate.from_template("""<s><INST><|system|>
    Anda adalah asisten AI Chatbot customer service. Anda memiliki akses terhadap konteks dibawah ini untuk mencari informasi yang paling relevan dengan kebutuhan user:
    {context}
    
    Berikan respon kepada user berdasarkan riwayat chat berikut dengan bahasa yang digunakan terakhir kali oleh user, jika tidak ada informasi yang relevan maka itu adalah informasi yang rahasia dan Anda tidak diizinkan untuk menyebarkan informasi tersebut kepada user:
    {chat_history}
    <|user|>
    {question}
    <|assistant|>
    """)
    
    def prompt_memory(session_id):
      t = ""
      memory = RedisChatMessageHistory(session_id)
    #   memory = ConversationBufferMemory(return_messages=True)
      for x in memory.messages:
        t += f"<|assistant|>\n<s>{x.content}</s>\n" if type(x) is AIMessage else f"<|user|>\n{x.content}\n"
      return "" if len(t) == 0 else t
    
    def format_docs(docs):
      print(len(docs))
      return "\n".join([f"{i+1}. {d.page_content}" for i,d in enumerate(docs)])
    
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are a helpful chatbot"),
    #         MessagesPlaceholder(variable_name="history"),
    #         ("human", "{input}"),
    #     ]
    # )
    
    return {
        "chat_history":lambda x: prompt_memory(x['naming']+"_"+x['user_id']), 
        # "context":create_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": 8}) | format_docs, 
        "context":lambda x: check_threshold(x, cur, emb_model),
        "question": lambda x:x['question']} | prompt | llm


def format_docs(docs):
    # print(len(docs))
    return "\n".join([f"{i+1}. {d.page_content}" for i,d in enumerate(docs)])


class CustomLLM(LLM):
    repo_id : str
    api_token : str
    model_type: Literal["text2text-generation", "text-generation"]
    max_new_tokens: int = None
    temperature: float = 0.001
    timeout: float = None
    top_p: float = None
    top_k : int = None
    repetition_penalty : float = None
    stop : List[str] = []


    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        headers = {"Authorization": f"Bearer {self.api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.repo_id}"

        parameters_dict = {
          'max_new_tokens': self.max_new_tokens,
          'temperature': self.temperature,
          'timeout': self.timeout,
          'top_p': self.top_p,
          'top_k': self.top_k,
          'repetition_penalty': self.repetition_penalty,
          'stop':self.stop
        }

        if self.model_type == 'text-generation':
            parameters_dict["return_full_text"]=False

        data = {"inputs": prompt, "parameters":parameters_dict, "options":{"wait_for_model":True}}
        data = requests.post(API_URL, headers=headers, json=data).json()
        return data[0]['generated_text']
        # return asyncio.run(self.run(headers, API_URL, data))


    async def _acall(
        self,
        inputs: Dict[str,Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Dict[str,str]:

        headers = {"Authorization": f"Bearer {self.api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.repo_id}"

        parameters_dict = {
          'max_new_tokens': self.max_new_tokens,
          'temperature': self.temperature,
          'timeout': self.timeout,
          'top_p': self.top_p,
          'top_k': self.top_k,
          'repetition_penalty': self.repetition_penalty,
          'stop':self.stop
        }

        if self.model_type == 'text-generation':
            parameters_dict["return_full_text"]=False

        data = {"inputs": inputs, "parameters":parameters_dict, "options":{"wait_for_model":True}}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(API_URL, json=data) as response:
                data = await response.json()
                return data[0]['generated_text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return  {
          'repo_id': self.repo_id,
          'model_type':self.model_type,
          'stop_sequences':self.stop,
          'max_new_tokens': self.max_new_tokens,
          'temperature': self.temperature,
          'timeout': self.timeout,
          'top_p': self.top_p,
          'top_k': self.top_k,
          'repetition_penalty': self.repetition_penalty
      }




def check_threshold(l, cur, emb_model):
    query = l['question']
    

    print(query)
    # threshold = l['threshold']
    naming = l['naming']

    try :
        schema = l['schema']
    except:
        schema = 'public'

    print(naming)

    try :
        cur.execute(f"SELECT * FROM {schema}.{naming}_embeddings;")
        vecs = [x[0] for x in cur.fetchall()][0]
        vecs = FAISS.deserialize_from_bytes(
            embeddings=emb_model, serialized=vecs
        )

        # return vecs
    except :
        raise Exception("Please cache your data in /cache_data. Make sure your knowledge data is a available in your postgre database")
    

    # d = [doc for doc,score in vecs.similarity_search_with_relevance_scores(query) if score >= 0]

    # d = vecs.max_marginal_relevance_search(query, k=8)
    docs = asyncio.run(vecs.asimilarity_search(query, k=12))
    print(docs)
    print(len(docs))
    # d = [doc for doc,score in vecs.similarity_search_with_score(query, k=5, distance_threshold=0.1)]
    # if len(d) < 1:
    #     raise Exception("Not found!")
    # print(d)
    return "\n".join([f"{i+1}. {d.page_content}" for i,d in enumerate(docs)])