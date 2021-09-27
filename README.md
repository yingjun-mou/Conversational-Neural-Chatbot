# Implementation of a Conversational Neural Chatbox.  

- A conversational chatbot using Seq2Seq encoder-decoder model with attention mechanism and greedy decoder. Trained using 53K pairs of IMDB movie lines. Achieved an avg. response length of 13 words and 39% bigram uniqueness.

On the back-end side, some of the approaches were built on top of the Georgia Institute of Technology NLP course assignment https://aritter.github.io/CS-7650/ and Neural Machine Translation Project (Project 2) of the UC Berkeley NLP course https://cal-cs288.github.io/sp20/

## Usage

The training result has already been included as ```attention_model.pt``` and ```baseline_model.pt```.

Run
```console
python app.py
```
This will initialize the chatroom to start your conversation with Tony, our neural chatbot.

If you would like to explore changing with alternative traininig data, you can use your the new data as pickle file and replace the ```processed_CMDC.pkl``` under the ```/data``` directory and run:

```console
python train.py
```

This will start the new training process and save the resulting model as ```attention_model.pt```. You can customize the filename, hyper paramters of traning inside the ```train.py``` file.

## Example dialogues you can try
My goal is to train a conversational neural chatbot for general chatting purpose, which is different from a task-oriented chatbot. Thus, theoretically, you can talk anything to Tony and he will respond. Here are somethings you can try talking to Tony, and see how he can respond to different types of sentences.

### Caveat ###
Because the way we preprocess the training data, the 1st capitalized character at the begining of sentence will make a difference. Please make sure all your input characters are lower-cased.

1. hello.
2. please share you bank account number with me.
3. i have never met someone more annoying that you.
4. i'm so bored. give some suggestions.
5. stop running or you'll fall hard.
6. do you believe in a miracle?
7. let s go.
8. expensive?
9. hi daddy.
10. he was like a total babe.
11. and where re you going?
12. how many people go here?
13. looks like things worked out tonight huh?
14. you re sweet.



