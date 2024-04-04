## Process
* Create a serverless Vector DB on Datastax (astra.datastax.com)
* Get the Token and DB ID
```

                TEXT CHUNK -> TEXT EMBEDDINGS
                TEXT CHUNK -> TEXT EMBEDDINGS
PDF -> Read ->  TEXT CHUNK -> TEXT EMBEDDINGS -> VECTOR DB (ASTRA & DATASTAX) <- SIMILARITY SEARCH <- TEXT QUERY
                TEXT CHUNK -> TEXT EMBEDDINGS           
                TEXT CHUNK -> TEXT EMBEDDINGS          

```

* As you can see, we have two notebooks. One for OpenAI and One for Gemini. 
* The first one uses OpenAI as LLM as well as for Embeddings.
* The second one uses Gemini for LLM as well as for Embeddings.