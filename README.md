# NLP-Legal-Document-Tagging
A program that automatically generates tags for legal text documents.

Tags have been a relevant part of information retrieval system. They help the system to get relevant information in most effective way. These tags have been generated manually by human annotators and makes the system more expensive for the clients. The only way to make these technologies affordable is to find a way to automatically generate tags that resembles to human annotations.

This project has used the liberties of the ‘spacy’ model to add patterns and pipes to the existing model and train it based on the added pattern described below.

The pattern added was to associate each tag with each keyword extracted from the main training docs. 
This way, there were tags associated with 80 training data files and their respective keywords.
Once the pattern was formed, the model was deployed. 

The accuracy can be improved by adding additional features and changing thresholds. This is the basic version.
