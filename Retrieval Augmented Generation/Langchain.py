# Import necessary modules from the Langchain library
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document from which information will be retrieved
# In this example, we are loading a text file named "state_of_the_union.txt"
loader = TextLoader("../../state_of_the_union.txt")
documents = loader.load()

# Split the loaded document into smaller chunks
# This is done to make the document more manageable and improve retrieval performance
# In this example, each chunk is 1000 characters long with no overlap between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Generate embeddings for the document chunks
# Embeddings are numerical representations of the text that are used for semantic search
embeddings = OpenAIEmbeddings()

# Create a vector store for semantic search using the embeddings. You can choose any vector databse.
# Chroma is a vector store that efficiently handles large sets of embeddings
docsearch = Chroma.from_documents(texts, embeddings)

# Initialize the RetrievalQA model
# This model uses a large language model (in this case, OpenAI's model) for question-answering
# It retrieves relevant document chunks using the vector store and generates a response
# The "stuff" chain type stuffs all of the retrieved document into the LLM context
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# Define the input query
# In this example, we are asking what the president said about Ketanji Brown Jackson
query = "What did the president say about Ketanji Brown Jackson"

# Run the RetrievalQA model with the input query
# This will retrieve relevant document chunks and generate a response
output = qa.run(query)

# Print the generated response
print(output)
