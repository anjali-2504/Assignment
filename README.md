

## Instructions
```python
!pip install langchain --upgrade
!pip install langchain-experimental
!pip install langchain-openai
!pip install beautifulsoup4
!pip install faiss-cpu 
```

```console
!pip install requirements.txt
```
* Replace the ['OPENAI_API_KEY'] with your own key.

* Make sure to run each cell one by one, starting from top to down.
## Data PreProcessing:


I first create a parser which takes in the url to lecture notes, filter out the contents in the scrolling section,
removes the anchor links and then gets the link to each of the chapters. Also, I have made use of all the chapters
present in the tutorial.

```python
if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all links to individual chapter pages
        chapter_links = soup.find_all('a', href=True)
        # Extract URLs of individual chapter pages excluding links with #
        chapter_urls = [link['href'] for link in chapter_links if '#' not in link['href']]
        return chapter_urls     
```

Once, it gets the links to each of the chapters, it extracts the text out of it and uses regex to remove the latex commands within the text. Around 8 regex filters were used in total.

```python
def preprocess_text(text):
    # Remove LaTeX commands enclosed within \(\) and $$
    text = re.sub(r'\\\((.*?)\\\)', '', text)
    text = re.sub(r'\$\$(.*?)\$\$', '', text)
    ## .... and other regex
    return text
```

Similarly, for fecthing the results of the table of model architectures,

```python
article = soup.find("article")
# Find the table containing milestone papers within the <article> tag
table = article.find("table")

# Extract table rows
rows = table.find_all("tr")
```

## Data Storage and Vectorisation

Data Structures to be used:
I have used the vector stores class present in the **langchain library** of OpenAI to create a vector index from a list of documents using FAISS, which is a library for efficient similarity search and clustering of dense vectors.

Overall, this line of code creates a vector index (vector) using FAISS, where each document in the documents list is represented by a vector obtained by embedding the text using the embeddings model. This index can then be used to efficiently search for similar documents based on their vector representations.

My earlier thoughts from using the text data as it is to using a vectorised storage came through my research on the complicacies faced
during search.

        Suppose we have a set of documents represented as vectors in a high-dimensional space, where each dimension   corresponds to a word or a feature in the document. Let's say we have two documents, A and B, represented as vectors A and B, respectively.

        The cosine similarity between two vectors A and B is given by the formula:

        cosine_similarity(A,B) = (A.B)/(||A||.||B||)

        where,
        A.B represents the dot product of vectors A and B.
        ||A|| and ||B|| represent the Euclidean norms of vectors A and B, respectively.
        Now, let's consider the process of finding the most similar document to a given query document Q from
        a set of documents using cosine similarity.


        Without vectorization:

        Compute the cosine similarity between Q and each document in the set.
        Select the document with the highest cosine similarity as the most similar document.


        With vectorization:

        Compute the cosine similarity between Q and the entire set of documents in a **single operation**.
        Select the document with the highest cosine similarity as the most similar document.
        The key efficiency gain comes from performing the cosine similarity computation in a single operation for all documents in the set. This is made possible by representing the documents as vectors and leveraging efficient linear algebra operations.

        In mathematical terms, computing the dot product between two vectors and their Euclidean norms is computationally efficient and can be performed in parallel for multiple vectors. This allows us to process large sets of documents quickly and efficiently, making vectorized representations ideal for similarity search tasks.


As of now, we are concatenating the cells within the same row to form a concatenated string to maintain the simplicity of the program. However, when the size of the table increases, we may consider storing it in some MySQL database.

Since, the langchain library is mainly used for storing textual data in vectorised formats, we try to convert every form of data to strong format. And then store them as vectors by first converting them to Documents.

**For Bonus Part 1:**

I create the document using the `text` as the page content and the `link` in the metadata section.
```python
docs = []
for link, text in chapter_text_data.items():
    doc = Document(page_content=text, metadata={"link": link})
    docs.append(doc)
```
Later on, when the response is generated, we can get the citations by:
```python
## GETTING CITATIONS
for res in response["context"]:
    print(res.metadata.get('link'))
```

I have combined the response and the citations together in the function 
`get_response` which takes in the `query` as input and generates the citations and the prompt.


**For Bonus Part 2:**
I have created a history aware retreiver which takes in the chat history of a particular session and then answers 
the consecutive follow-ups in the conversation.

```python
retriever = vector.as_retriever()
# First we need a prompt that we can pass into an LLM to generate this search query
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
```


Now, everytime, a chat session is initiated, the variable 
```python
chat_history
``` 
keeps on getting updated.


```python

while(flag):
  query = input()
  if query=='-1':
    break
  response = retrieval_chain.invoke({  
    "chat_history": chat_history,
    "input": query,
  })
  chat_history.append(query)
  print("Question asked: \n", query)
  conversation_history.append("User : "+query)
  conversation_history.append("AI : "+response["answer"])
  print(response["answer"])

```

This code provides the user to enter their query in the run time and get responses.
Here, one important thing to note is that, the agent would stop generating responses, once the user enters -1
as the input indiciating that he/she has no more queries to ask.

    
    Question asked: 
    What are LLM's used for?
    Large Language Models (LLMs) are used for various natural language processing tasks such as text generation, language translation, sentiment analysis, question answering, summarization, and more. They are capable of understanding and generating human-like text based on the patterns and data they have been trained on.

    Question asked: 
    Can LLMs summarise data? 
    Yes, Large Language Models (LLMs) are capable of summarizing data. They can be used for text summarization tasks where they condense longer pieces of text into shorter summaries while retaining the key information and meaning. This ability makes them useful for tasks like creating abstracts of articles, condensing reports, or generating brief overviews of documents.


    Question asked: 
    Which LLM should I use for my work? 
    It depends on the specific natural language processing task you need to perform. Different LLMs have been trained and specialized for various tasks. For example, GPT-3 is known for its text generation capabilities, while BERT is often used for tasks like question answering and sentiment analysis. Evaluate the strengths and weaknesses of each model based on your requirements to determine the best fit for your work.

    -1



**For the Bonus Part 3:**

I make use of the pipeline architecture of the transformers which takes in the entire conversation between the human and the AI and generates a summary based on the constraints provided.
```python

# Load the summarization pipeline
summarizer = pipeline("summarization")


# Concatenate conversation history into a single text
conversation_text = "\n".join(conversation_history)

# Generate summary of conversation
summary = summarizer(conversation_text, max_length=100, min_length=30, num_beams=4, early_stopping=True)

# Print the summary
print("Summary of conversation:")
print(summary[0]['summary_text'])
```

Summary Response from the transformer: 

    Large Language Models (LLMs) are used for various natural language processing tasks such as text generation, language translation, sentiment analysis, question answering, summarization, and more . Can be used for text summarization tasks where they condense longer pieces of text into shorter summaries while retaining the key information and meaning . Different LLMs have been trained and specialized for various tasks.

## Results of the Agent to Human Query

> "What are some milestone model architectures and papers in the last few years?"

_Some milestone model architectures and papers in the last few years include:_

_1. BERT (Bidirectional Encoder Representations from Transformers) - Devlin et al., 2018_

_2. GPT-2 (Generative Pre-trained Transformer 2) - Radford et al., 2019_

_3. T5 (Text-to-Text Transfer Transformer) - Raffel et al., 2019_

_4. GPT-3 (Generative Pre-trained Transformer 3) - Brown et al., 2020_

_5. RoBERTa (A Robustly Optimized BERT Approach) - Liu et al., 2019_

_These models have significantly advanced the field of natural language processing and have been instrumental in various NLP tasks and applications._


> “What are the layers in a transformer block?”

_In a transformer block, there are typically two main layers:_

_1. Self-Attention Layer: This layer allows each token in the sequence to interact with every other token in the sequence, capturing dependencies and relationships between different parts of the input sequence._

_2. Feed-Forward Layer: This layer processes each token independently through a series of fully connected neural networks, providing non-linear transformations to the input embeddings._


> “Tell me about datasets used to train LLMs and how they’re cleaned” (summarizing query)

_Large language models (LLMs) are trained on datasets that contain vast amounts of text data from various sources like the web, news, Wikipedia, and fiction. These datasets are cleaned to remove noise, irrelevant information, and potentially harmful content. For example, the WebText dataset used to train GPT-2 was created by scraping outbound links with upvotes, filtering out Wikipedia content, and ensuring high-quality diverse text. OpenAI's OpenWebText dataset was created by extracting URLs from Reddit submissions, filtering out non-English text, and removing duplicates. The Colossal Clean Crawled Corpus (C4) used to train the T5 model involved removing "bad words," code, and non-English text. Cleaning the datasets helps improve the quality and reliability of the training data for LLMs._

> What are some challenges encountered during cleaning of datasets, and how are these challenges addressed?

_During the cleaning of datasets used to train Large Language Models (LLMs), several challenges can be encountered. Some of these challenges include:_

_1. **Data Quality**: Ensuring that the data is of high quality and relevant to the task at hand. Low-quality or irrelevant data can negatively impact the performance of the model._

_2. **Data Bias**: Datasets may contain biases that can lead to discriminatory or unfair outcomes when the model is deployed. Addressing bias in the data is crucial to ensure fair and equitable results._

_3. **Data Privacy**: Protecting sensitive information present in the data is essential to maintain privacy and confidentiality. Anonymizing or removing personally identifiable information is often necessary._

_4. **Data Duplication**: Duplicates in the dataset can skew the model's training and evaluation. Removing duplicates or handling them appropriately is important for accurate training._

_5. **Data Size**: Large datasets can be challenging to clean efficiently. Managing and cleaning massive amounts of data can be time-consuming and resource-intensive._

_These challenges are addressed through various techniques and processes, including:_

_1. **Data Preprocessing**: This involves cleaning, normalizing, and transforming the data to make it suitable for training the model. Steps like removing irrelevant information, handling missing values, and standardizing formats are part of this process._

_2. **Data Augmentation**: Adding variations to the dataset by generating new samples from existing data can help improve model performance and reduce overfitting._

_3. **Data Sampling**: Balancing the dataset by sampling from different classes or categories can address imbalances and biases in the data._

_4. **Data Validation**: Checking the cleaned data for errors, inconsistencies, and biases through validation techniques to ensure the quality and integrity of the dataset._

_5. **Ethical Considerations**: Considering ethical implications, such as fairness, transparency, and accountability, when cleaning and preparing the data to ensure that the model's outcomes are unbiased and reliable._

_By addressing these challenges and implementing appropriate strategies, researchers and data scientists can create high-quality datasets that lead to well-performing Large Language Models._


> Thanks for answering.

_You're welcome! If you have any more questions or need further information, feel free to ask. I'm here to help!_





##
 $$
 Note:
 $$

In order to try with more queries, you can simply call 
```python
get_response(query)
```

However, in order to generate the summary for the entire response, you would need to start from the
 **interactive conversation** and then run the **summariser** code.


I have not created an agent class using object oriented programming given the time constraint of solving
the problem in 3-4 hours. However, this could be easily wrapped up in a class and provided as an agent to 
the user.


## Future Scopes:

To incorporate more efficient storage for storing large sized tables, images, large links (by shortening the url and hashing)

To use the LLM API for summarising the conversation

To expand the number of servers (horizontal scaling) when the number of lectures and data increase. 

Replace with distributed environments and Elasticsearch or Apache Solr search solutions.

Use of Caching Mechanisms and Load Balancing to cater to increasing requests.
