WikiQA
======

Answer simple questions using open language models and Wikipedia.

Operation
---------

The high-level operation of the application is as follows:

1. Accept a natural language query
2. Pull proper nouns from the query
3. For each proper noun, access its most relevant Wikipedia page
4. Using sentence embeddings, select the sentences from Wikipedia most related to the query.
5. Pass the query and the related knowledge to a language model fine-tuned for dialog question answering.
6. Hopefully get an accurate response back from the model based on the knowledge.

Credits
-------

Stop Word list from Terrier