
#Installing The Required Libraries
 %pip install "git+https://github.com/ibm-granite-community/utils.git" \
    transformers \
    langchain_community \
    'langchain_huggingface[full]' \
    langchain_milvus \
    docling \
    replicate
#Selecting System Components

#Choose your Embeddings Model
#Use Ibm-granite



from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

#Use the Granite model

from langchain_community.llms import Replicate
from ibm_granite_community.notebook_utils import get_env_var

model_path = "ibm-granite/granite-3.3-8b-instruct"
model = Replicate(
    model=model_path,
    replicate_api_token=get_env_var("REPLICATE_API_TOKEN"),
    model_kwargs={
        "max_tokens": 1000, # Set the maximum number of tokens to generate as output.
        "min_tokens": 100, # Set the minimum number of tokens to generate as output.
    },
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Now that  the model is downloaded, let's try asking it a question

from ibm_granite_community.langchain import TokenizerChatPromptTemplate

query = "Who won in the Pantoja vs Asakura fight at UFC 310?"

Create a Granite prompt for question-answering
prompt_template = TokenizerChatPromptTemplate.from_template(template="{input}", tokenizer=tokenizer)

chain = prompt_template | model

output = chain.invoke({"input": query})

print(output)

# Now, I know that UFC 310 happened in 2024, and this does not seem to be the right Pantoja. The model doesn't seem to know the answer but at least understands that this matchup did not occur. Let's see if it has some specific UFC rules info."""

query1 = "How much weight allowance is allowed in non championship fights in the UFC?"

output = chain.invoke({"input": query1})

print(output)

# Based on the official UFC rules, this is also incorrect. Let's try getting some documents that contains this information for the model.

### Choose your Vector Database

Specify the database to use for storing and retrieving embedding vectors.

To connect to a vector database other than Milvus, replace this code cell with one from [this Vector Store recipe](https://github.com/ibm-granite-community/granite-kitchen/blob/main/recipes/Components/Langchain_Vector_Stores.ipynb).
"""

from langchain_milvus import Milvus
import tempfile

db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
print(f"The vector database will be saved to {db_file}")

vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    enable_dynamic_field=True,
    index_params={"index_type": "AUTOINDEX"},
)

"""## Step 3: Building the Vector Database

In this example, from a set of source documents, we use [Docling](https://docling-project.github.io/docling/) to convert the documents into text and then split the text into chunks, derive embedding vectors using the embedding model, and load it into the vector database. Creating this vector database will allow us to easily search across our documents, enabling us to use RAG.

### Use Docling to download the documents, convert to text, and split into chunks

Here we have found a website that gives us information on UFC 310, as well as a PDF of the official UFC rules. Below, we will see that Docling can both convert and chunk the two documents.
"""

# Docling imports
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document

# Here are our documents, feel free to add more documents in formats that Docling supports
sources = [
    "https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-310-pantoja-vs-asakura",
    "https://media.ufc.tv/discover-ufc/Unified_Rules_MMA.pdf",
]

converter = DocumentConverter()

# Convert and chunk out documents
doc_id = 0
texts: list[Document] = [
    Document(page_content=chunk.text, metadata={"doc_id": (doc_id:=doc_id+1), "source": source})
    for source in sources
    for chunk in HybridChunker(tokenizer=embeddings_tokenizer).chunk(converter.convert(source=source).document)
    if any(filter(lambda c: c.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH], iter(chunk.meta.doc_items)))
]

print(f"{len(texts)} document chunks created")

# Print all created documents
for document in texts:
    print(f"Document ID: {document.metadata['doc_id']}")
    print(f"Source: {document.metadata['source']}")
    print(f"Content:\n{document.page_content}")
    print("=" * 80)  # Separator for clarity

"""### Populate the vector database

NOTE: Population of the vector database may take over a minute depending on your embedding model and service.
"""

ids = vector_db.add_documents(texts)
print(f"{len(ids)} documents added to the vector database")

"""## Step 4: RAG with Granite

Now that we have succesfully converted our documents and vectorized them, we can set up out RAG pipeline.

### Retrieve relevant chunks

Here we will test the as_retriever method to search through our newly created vector database for chunks that are relevant to our original query
"""

retriever = vector_db.as_retriever()

docs = retriever.invoke(query)
print(docs)

"""Looks like it pulled some chunks that would have the information we are looking for. Let's go ahead and contruct our RAG pipeline.

### Create the prompt for Granite

Next, we construct the prompt pipeline. This creates the prompt which holds the retrieved chunks from out previous search and feeds this to the model as context for answering our question.
"""

from ibm_granite_community.langchain import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Assemble the retrieval-augmented generation chain
combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
)
rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(),
    combine_docs_chain=combine_docs_chain,
)

"""### Generate a retrieval-augmented response to a question

The pipeline uses the query to locate documents from the vector database and use them as context for the query.
"""

output = rag_chain.invoke({"input": query})

print(output['answer'])

"""Awesome! It looks like the model figured out our first question. Let's see if it figure out the rule we were looking for."""

output = rag_chain.invoke({"input": query1})

print(output['answer'])

"""Awesome! We can now see that we have created a pipeline that can successfully leverage knowledge from multiple document types for generation.

## Next Steps

- Explore advanced RAG workflows for other industries
- Experiment with other document types and larger datasets.
- Optimize prompt engineering for better Granite responses.

