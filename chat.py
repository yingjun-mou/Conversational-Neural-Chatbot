import torch
from model import Seq2seqAttention
from train import bos_id, vocab, eos_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load pre-trained model from file
FILE = "attention_model.pt"
# FILE = "baseline_model.pt"
model = Seq2seqAttention(vocab).to(device)
model.load_state_dict(torch.load(FILE, map_location=device))

bot_name = "Tony"


def get_response(msg, max_length=100):
    model.eval()
    source = torch.tensor(vocab.get_ids_from_sentence(msg))
    source = source.unsqueeze(-1).to(device)

    # Forward input through encoder model
    encoder_outputs, encoder_mask, encoder_hidden = model.encode(source)

    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:model.decoder.num_layers]
    # Initialize decoder input with SOS_token
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * bos_id
    # Initialize tensors to append decoded words to
    all_tokens = torch.zeros([0], dtype=torch.long).to(device)
    all_scores = torch.zeros([0]).to(device)
    # Iteratively decode one word token at a time
    for _ in range(max_length):
        # Forward pass through decoder
        decoder_output, decoder_hidden, _ = model.decode(decoder_input, decoder_hidden, encoder_outputs, encoder_mask)
        # Obtain most likely word token and its softmax score
        decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
        # Record token and score
        all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
        all_scores = torch.cat((all_scores, decoder_scores), dim=0)

        # terminate greed search it we get a EOS token
        if decoder_input == eos_id:
            break

        # Prepare current token to be next decoder input (add a dimension)
        decoder_input = torch.unsqueeze(decoder_input, 0)

    # Return collections of word tokens and scores
    output_id = all_tokens.detach().cpu().numpy()
    output = vocab.decode_sentence_from_ids(output_id)

    # return all_tokens, all_scores
    return output
