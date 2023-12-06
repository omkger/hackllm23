from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# For Loading Augmentation Data for context generation
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings



model_path = "llama-2-7b-chat.Q6_K.gguf"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# load the model
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.75,
    max_tokens=400,
    n_ctx = 2048,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# create a prompt
template = """
You are a helpful, respectful and honest Telecommuniatons Regulation assistant. Must use the following pieces of context to answer the question only.
If you don't know the answer from the provided context, don't make up an answer. Also, do not replace any word with your own word.
Use three sentences maximum and keep the answer as concise as possible.

{context}
Question: {question}
Helpful Answer:
"""

# Load documents:
loader = PyPDFLoader("What is Giga.pdf")
splitted_data = loader.load_and_split()

model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
cache_folder = "huggingfaceembeddings_cache"
embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder = cache_folder
)

# create embedding from the documents
vectorstore = Chroma.from_documents(documents=splitted_data, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.6,'fetch_k': 50})

question_text = "What is Giga?"
context_text = retriever.invoke(question_text)

prompt = PromptTemplate.from_template(template)

# Format the prompt
filled_prompt = prompt.format(context = context_text, question = question_text)

print(filled_prompt)

response = llm.invoke(filled_prompt)

### The chain way of doing things:
#llm_chain = LLMChain(prompt=prompt, llm=llm)
#response = llm_chain.run({"context": context, "question": question})
###

print(response)
