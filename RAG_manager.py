import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_models.bedrock import ChatPromptAdapter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
llm=ChatOpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
    model=os.getenv("MINIMAX_MODEL"),
    extra_body={
        "thinking":{"type":"disabled"}
    },
    temperature=0
)

class RAG:
        def __init__(self):
            self.llm=llm
            self.embed_model=HuggingFaceEmbeddings(
                model="C:/Users/zxsoul/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
            )
            self.vectorstroe=Chroma(
                collection_name="ISagent1",
                embedding_function=self.embed_model,
                persist_directory="./chromadb"
            )
            self.retriever=self.vectorstroe.as_retriever()
            self.splitter=RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

        def load_file(self,docs_path:str=None):
            print(docs_path)
            if docs_path and os.path.exists(docs_path):
                #对于相同路径下的一致文档采取了防重复导入功能
                print(1)
                try:
                    loader=UnstructuredWordDocumentLoader(docs_path)
                    docs=loader.load()
                    print(f"word文档载入成功")
                    file_path=docs[0].metadata.get("source")
                    exsting_docs=self.vectorstroe.similarity_search("",k=1,filter={"source":file_path})

                    if exsting_docs:
                        print(f"文档已经存在于知识库中，不必导入")
                    else:
                        chunks = self.splitter.split_documents(docs)
                        print(f"文档已成功切割")
                        return chunks

                except Exception as e:
                    print(f"word文档载入失败：{e}")
            else:
                print(f"文件路径不存在")
                return None

        def add_file(self,chunks):
            if not chunks:
                print("当前没有写入的新文档分块，跳过写入操作")
                return
            try:
                self.retriever.vectorstore.add_documents(chunks)
                print(f"成功将{len(chunks)}个文档写入数据库")
            except Exception as e:
                print(f"文档写入数据库失败：{e}")

        def query(self,question:str):
            relevant_docs=self.retriever.invoke(question)
            if not relevant_docs:
                return "抱歉，在知识库中没能找到相关内容"

            content="\n\n".join([
                f"文档：{i+1}\n{doc.page_content}"
                for i,doc in enumerate(relevant_docs[:5])
            ])
            # print(content)
            prompt=ChatPromptTemplate.from_template(
                """
                    你是一个回答机器人，请根据一下文档信息回答用户问题，禁止自主编造，如果文档内容不足以回答问题，请返回 "我无法回答您的问题"
                    原问题：{question}
                    文档：{content}
                """
            )
            chain=prompt|self.llm|StrOutputParser()
            response=chain.invoke(
                {
                    "question":question,
                    "content":content
                }
            )
            return response


    # query="作品要求是什么"

rag=RAG()


    # response=rag.query(query)
    # print(f"正式回答是:\n{response}")
# file_path="C:/Users/zxsoul/Downloads/经管学院促俄项目报名通知.docx"
# if file_path:
#         chunks=rag.load_file(file_path)
#         rag.add_file(chunks)
