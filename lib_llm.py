from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
import os



cache_dir = "./cache"
def getFlanLarge():
  
   model_id = 'google/flan-t5-large'
   print(f">> Prep. Get {model_id} ready to go")
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir)
  
   pipe = pipeline(
       "text2text-generation",
       model=model,
       tokenizer=tokenizer,
       max_length=100
   )
   llm = HuggingFacePipeline(pipeline=pipe)
   return llm

local_llm = getFlanLarge()

def make_the_llm():
   template_informed = """
   I am a helpful AI that answers questions.
   When I don't know the answer I say I don't know.
   I know context: {context}
   when asked: {question}
   my response using only information in the context is: """
   prompt_informed = PromptTemplate(
       template=template_informed,
       input_variables=["context", "chat_history","question"])
   return LLMChain(prompt=prompt_informed, llm=local_llm)


def make_the_llm_memory():
   template_informed = """
   I am a helpful AI that answers questions.
   When I don't know the answer I say I don't know.
   context: {context}

   {chat_history}
   Human: {question}
   AI: """
   prompt_informed = PromptTemplate(
       template=template_informed,
       input_variables=["context", "chat_history","question"])
   #return LLMChain(prompt=prompt_informed, llm=local_llm)

   message_history = RedisChatMessageHistory(url='redis://localhost:6379', ttl=600, session_id='client_id')
   memory = ConversationBufferMemory(return_messages=True, memory_key = 'chat_history', input_key = 'question', chat_memory = message_history)
   conversation = load_qa_chain(
      llm = local_llm,
      prompt = prompt_informed,
      chain_type = 'stuff',
      verbose = True,
      memory = memory
   )
   return conversation

topic = "Star Wars"
index_name = "book_wookieepedia_mpnet"

# Create the HuggingFace Transformer like before
model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)

url = f"http://localhost:9200"
db = ElasticVectorSearch(embedding=hf, elasticsearch_url=url, index_name=index_name)

llm_chain_informed= make_the_llm_memory()

def ask_a_question(question):
   ## get the relevant chunk from Elasticsearch for a question
   similar_docs = db.similarity_search(question)
   print(f'The most relevant passage: \n\t{similar_docs[0].page_content}')
   informed_context= similar_docs[0].page_content
   informed_response = llm_chain_informed.run(
       context=informed_context,
       question=question)
   return informed_response

def ask_a_question_memory(question):
   ## get the relevant chunk from Elasticsearch for a question
   similar_docs = db.similarity_search(question, k = 1)
   print(f'The most relevant passage: \n\t{similar_docs[0].page_content}')
   
   #informed_context= list(similar_docs[0])
   informed_response = llm_chain_informed(
       {'input_documents': similar_docs, 'question': question},
       return_only_outputs = True)
   return informed_response

# The conversational loop
print(f'I am a trivia chat bot, ask me any question about {topic}')
#while True:
#   command = input("User Question >> ")
#  response= ask_a_question(command)
#  print(f"\tAnswer  : {response}")

while True:
  command = input("User Question >> ")
  response= ask_a_question_memory(command)
  print(f"\tAnswer  : {response['output_text']}")