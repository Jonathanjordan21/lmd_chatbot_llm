# LMD Large Language Model (LLM) Chatbot
## How to use 
### Run The App
Deploy docker Postgre and Redis images
```bash
docker-compose up
```
<br>

Install python packages and run Flask App
```bash
pip install -r requirements.txt
```

```bash
flask run
```

<br>

### Cache Data
Before generating the Chatbot response, make POST api call of `form-data` type to cache the knowledge base embeddings to /cache_data endpoint. <br>
The required parameters are :
1. tenant_name : tenant name (e.g lmd)
2. socmed_type : socmed type (e.g whatsapp)
3. module_flag : module flag (e.g ebesha)
4. table_name : name of the knowledge base table, in the postgres/init-scripts/init.sql there are two knowledge base tables : lmd_knowledge and indomaret_knowledge 
<br>

## Generate Chatbot Response 
### /chatbot endpoint
Uses zero shot classification method to classify whether the user input belongs to the database or knowledge-base domain. <br><br>

Make POST API call to /chatbot endpoint <br>
The required parameters are :
1. query : user input in ENGLISH (e.g What is LMD?)
2. tenant_name : tenant name (e.g lmd)
3. socmed_type : socmed type (e.g whatsapp)
4. module_flag : module flag (e.g ebesha)
<br>

### /chatbot_choose endpoint
User choose which data source to retrieve, the knowledge-base or the database. If the choosen data source doesn't find a match, then the other data source will catch the error and be used to search for the query. If it also didn't find a match, then an exception will be returned. <br><br>

Make POST API call to /chatbot_choose endpoint <br>
The required parameters are :
1. query : user input in ENGLISH (e.g What is LMD?)
2. tenant_name : tenant name (e.g lmd)
3. socmed_type : socmed type (e.g whatsapp)
4. module_flag : module flag (e.g ebesha)
5. data_source : data source to retrieve (database or knowledge)
<br>

## Expected output
1. Chatbot generate response based on the knowledge-base via RAG 
2. Chatbot retrieve data from Database and generate response of list of selected items
3. Chatbot return error response because no simillar data exists in the knowledge-base
4. Chatbot return error response because the generated SQL syntax is error due to either unavailable data in the Database or bad SQL syntax
5. Chatbot generate response "I don't know" because the Chatbot cannot understand the knowledge-base



