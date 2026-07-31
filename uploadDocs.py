from RAG_manager import rag


file_path=input("请输入文档地址：\n").strip().strip('"')
if file_path:
        chunks=rag.load_file(file_path)
        rag.add_file(chunks)