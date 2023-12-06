from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



model_path = "llama-2-7b-chat.Q6_K.gguf"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# load the model
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.75,
    max_tokens=400,
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

context_text = "You also like to crack jokes."
#question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

question_text = "How hard is it to get internet in Afghanistan?"

prompt = PromptTemplate.from_template(template)

# Format the prompt
filled_prompt = prompt.format(context = context_text, question = question_text)

response = llm.invoke(filled_prompt)

### The chain way of doing things:
#llm_chain = LLMChain(prompt=prompt, llm=llm)
#response = llm_chain.run({"context": context, "question": question})
###

print(response)