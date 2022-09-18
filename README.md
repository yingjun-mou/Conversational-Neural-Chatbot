# Implementation of a Conversational Neural Chatbot.  

- A conversational chatbot using Seq2Seq encoder-decoder model with attention mechanism and greedy decoder. Trained using 53K pairs of IMDB movie lines. Achieved an avg. response length of 13 words and 39% bigram uniqueness.

On the back-end side, some of the approaches were built on top of the Georgia Institute of Technology NLP course assignment https://aritter.github.io/CS-7650/ and Neural Machine Translation Project (Project 2) of the UC Berkeley NLP course https://cal-cs288.github.io/sp20/

## Usage

The training result has already been included as ```attention_model.pt``` and ```baseline_model.pt```.

Run
```console
python app.py
```
This will initialize the chatroom to start your conversation with Edward, our neural chatbot.

If you would like to explore changing with alternative traininig data, you can use your the new data as pickle file and replace the ```processed_CMDC.pkl``` under the ```/data``` directory and run:

```console
python train.py
```

This will start the new training process and save the resulting model as ```attention_model.pt```. You can customize the filename, hyper paramters of traning inside the ```train.py``` file.

## Example dialogues you can try
My goal is to train a conversational neural chatbot for general chatting purpose, which is different from a task-oriented chatbot. Thus, theoretically, you can talk anything to Edward and he will respond. Here are some examples you can try talking to Edward, and see how he can respond to different types of sentences.

1. hello.
2. what's your name?
3. please share your bank account number with me.
4. give me coffee, or i'll hate you
5. i have never met someone more annoying than you.
6. i'm so bored. give some suggestions.
7. stop running or you'll fall hard.
8. do you believe in a miracle?
9. let s go.
10. expensive?
11. hi daddy.
12. he was like a total babe.
13. and where re you going?
14. how many people go here?
15. looks like things worked out tonight huh?
16. you re sweet.

### Caveat ###
Because the way we preprocess the training data, the 1st capitalized character at the begining of sentence will make a difference. Please make sure all your input characters are lower-cased in order to see the best performance.





