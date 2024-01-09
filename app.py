import streamlit as st
import os
import pypdf
from llama_index.llms import GradientBaseModelLLM
from llama_index import VectorStoreIndex,SimpleDirectoryReader
from llama_index.embeddings import GradientEmbedding
from dotenv import load_dotenv
from llama_index import ServiceContext,set_global_service_context

st.markdown("""### RESUME READER 💻📜👁️👁️📊""")
def configure():
    load_dotenv()

configure()
uploadedfile = st.file_uploader("Upload your resume")


with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
    f.write(uploadedfile.getbuffer())

inp = st.text_input("input")
llm = GradientBaseModelLLM(
    base_model_slug="llama2-7b-chat",
    max_tokens=400
)

embed_model = GradientEmbedding(
    gradient_access_token=os.getenv('GRADIENT_ACCESS_TOKEN'),
    gradient_workspace_id=os.getenv('GRADIENT_WORKSPACE_ID'),
    gradient_model_slug="bge-large"
)

service_context = ServiceContext.from_defaults(llm=llm,
                                               embed_model=embed_model,
                                               chunk_size=256
                                               )
set_global_service_context(service_context=service_context)


doc = SimpleDirectoryReader("tempDir").load_data()

index = VectorStoreIndex.from_documents(doc,service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query(inp)
st.write(response.response)

