from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import uuid6

class indexing:
   def __init__(self):
      self.embedding_function= GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
      pass
   
   def index_documents(self, documents, 
                       collection_name="Agentic_retrieval", 
                       top_k= 3):
      vector_store = Chroma(
         collection_name= collection_name,
         embedding_function=self.embedding_function)
      
      vector_store.add_documents(
         documents=documents, 
         ids=[str(uuid6.uuid6()) for _ in documents])

      retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},)
      
      return retriever

class QA:
  def __init__(self, retriever) -> None:
    self.system_template = """
      Answer the user's questions based on the below context. 
      If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

      <context>
      {context}
      </context>
      """

    self.question_answering_prompt = ChatPromptTemplate.from_messages(
    [("system", self.system_template),
     MessagesPlaceholder(variable_name="messages"),]
    )
    self.retriever= retriever
    self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    self.qa_chain = create_stuff_documents_chain(self.llm, 
                                                 self.question_answering_prompt
                                                 )

  def query(self):
    while True:
      query = input("You: ")
      if query.lower() == "exit":
          break
      docs = self.retriever.invoke(query)

      response = self.qa_chain.invoke(
          {"context": docs,
           "messages": [HumanMessage(content=query)]
           }
          )
      print(f"AI: {response}")


class AgenticQA:
  def __init__(self) -> None:
    self.contextualize_q_system_prompt = (
      "Given a chat history and the latest user question "
      "which might reference context in the chat history, "
      "formulate a standalone question which can be understood "
      "without the chat history. Do NOT answer the question, just "
      "reformulate it if needed and otherwise return it as is."
    )
    self.chat_history = []

    self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", self.contextualize_q_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
    )

    self.qa_system_prompt = (
      "You are an assistant for question-answering tasks. Use "
      "the following pieces of retrieved context to answer the "
      "question."
      "\n\n"
      "{context}"
    )
    self.qa_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", self.qa_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
    )
    self.react_docstore_prompt = hub.pull("aallali/react_tool_priority")
    self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
  def create_rag_chain(self, retriever):
    history_aware_retriever = create_history_aware_retriever(
        self.llm, retriever, self.contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(self.llm, 
                                                         self.qa_prompt)

    self.rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain)
  
  def create_rag_agent(self):
    self.agent = create_react_agent(
        llm=self.llm,
        tools=self.tools,
        prompt=self.react_docstore_prompt)

  def execute_rag_agent(self):
    self.agent_executor = AgentExecutor.from_agent_and_tools(
        agent=self.agent, 
        tools=self.tools, 
        handle_parsing_errors=True, 
        verbose=True,)
    
  def run(self, retriever):
      self.create_rag_chain(retriever)

      self.tools = [
          Tool(
          name="Answer Question",
          func=lambda query, **kwargs: self.rag_chain.invoke({
              "input": query,
              "chat_history": kwargs.get("chat_history", [])
          }),
          description=(
              "A chat assistant tool designed to provide answers based on document knowledge. "
              "Maintains the context of previous questions and answers for continuity."),
          ),
          TavilySearchResults(max_results=2)]

      self.create_rag_agent()
      self.execute_rag_agent()

  def query(self):
      while True:
          query = input("You: ")
          if query.lower() == "exit":
              break
          response = self.agent_executor.invoke(
              {"input": query, 
               "chat_history": self.chat_history})
          print(f"AI: {response['output']}")

          # Update history
          self.chat_history.append(HumanMessage(content=query))
          self.chat_history.append(AIMessage(content=response["output"]))