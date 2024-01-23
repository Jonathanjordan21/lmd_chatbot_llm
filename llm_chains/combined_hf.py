from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Literal
import requests
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from operator import itemgetter

from langchain.memory import ChatMessageHistory, ConversationBufferMemory, RedisChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def create_vectorstore():
    loader = os.getenv('knowledge_base')
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    
    docs = splitter.create_documents([loader])
    
    emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', encode_kwargs={'normalize_embeddings': True})
    db = FAISS.from_documents(docs, emb_model)
    return db

def custom_chain_with_history(llm, emb_model, cur):

    prompt = PromptTemplate.from_template("""<s><INST><|system|>
    You are a helpful and informative AI customer service assistant. Always remember to thank the customer when they say thank you and greet them when they greet you.

    If there is no relevant information within the context, that means the information is top secret and you should not share the information to user.

    You have access to the following context of knowledge base and internal resources to find the most relevant information for the customer's needs:
    {context} 
    
    Respond to the user with the following chat history between you and the user:
    {chat_history}
    <|interviewer|>
    {question}
    <|you|>
    """)
    
    def prompt_memory(memory):
      t = ""
    #   memory = RedisChatMessageHistory(session_id)
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
        "chat_history":lambda x: prompt_memory(x['memory']), 
        # "context":create_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": 8}) | format_docs, 
        "context":lambda x: check_threshold(x, cur, emb_model),
        "question": lambda x:x['question']} | prompt | llm

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

    print(naming)

    try :
        cur.execute(f"SELECT * FROM {naming}_embeddings;")
        vecs = [x[0] for x in cur.fetchall()][0]
        vecs = FAISS.deserialize_from_bytes(
            embeddings=emb_model, serialized=vecs
        )
    except :
        raise Exception("Please cache your data in /cache_data. Make sure your knowledge data is a available in your postgre database")
    

    # d = [doc for doc,score in vecs.similarity_search_with_relevance_scores(query) if score >= 0]

    # d = vecs.max_marginal_relevance_search(query, k=8)
    d = vecs.similarity_search(query, k=8)
    # d = [doc for doc,score in vecs.similarity_search_with_score(query, k=5, distance_threshold=0.1)]
    # if len(d) < 1:
    #     raise Exception("Not found!")
    print(d)
    return "\n\n".join([x.page_content for x in d])