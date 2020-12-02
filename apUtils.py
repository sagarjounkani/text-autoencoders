import random
import numpy as np
import torch
import spacy
from torchtext.data.metrics import bleu_score

# Contains nearby characters on the keyboard for substitution when generating typos
character_replacement = dict()

character_replacement['a'] = 'qwsz'
character_replacement['b'] = 'nhgv '
character_replacement['c'] = 'vfdx '
character_replacement['d'] = 'fresxc'
character_replacement['e'] = 'sdfr43ws'
character_replacement['f'] = 'gtrdcv'
character_replacement['g'] = 'hytfvb'
character_replacement['h'] = 'juytgbn'
character_replacement['i'] = 'ujklo98'
character_replacement['j'] = 'mkiuyhn'
character_replacement['k'] = 'jmloij'
character_replacement['l'] = 'kpok'
character_replacement['m'] = 'njk '
character_replacement['n'] = 'bhjm '
character_replacement['o'] = 'plki90p'
character_replacement['p'] = 'ol0o'
character_replacement['q'] = 'asw21'
character_replacement['r'] = 'tfde45'
character_replacement['s'] = 'dxzawe'
character_replacement['t'] = 'ygfr56'
character_replacement['u'] = 'ijhy78'
character_replacement['v'] = 'cfgb '
character_replacement['w'] = 'saq23e'
character_replacement['x'] = 'zsdc'
character_replacement['y'] = 'uhgt67'
character_replacement['z'] = 'xsa'
character_replacement['1'] = '2q'
character_replacement['2'] = '3wq1'
character_replacement['3'] = '4ew2'
character_replacement['4'] = '5re3'
character_replacement['5'] = '6tr4'
character_replacement['6'] = '7yt5'
character_replacement['7'] = '8uy6'
character_replacement['8'] = '9iu7'
character_replacement['9'] = '0oi8'
character_replacement['0'] = 'po9'


def generate_typo(s: str, sub_rate: float = 0.01, del_rate: float = 0.005, dupe_rate: float = 0.005,
                  transpose_rate: float = 0.01) -> str:
    """
    Generates a new string containing some plausible typos
    :param s: the input string
    :param sub_rate: character substitution rate (0 < x < 1)
    :param del_rate: character deletion rate (0 < x < 1)
    :param dupe_rate: character duplication rate (0 < x < 1)
    :param transpose_rate: character transposition rate (0 < x < 1)
    :return: the string with typos
    """
    if len(s) == 0:
        return s

    new_string = list()
    for i, char in enumerate(s.lower()):

        # Decide what to do
        do = np.random.uniform(size=(4,))
        do_swap = do[0] < sub_rate
        do_delete = do[1] < del_rate
        do_duplicate = do[2] < dupe_rate
        do_transpose = do[3] < transpose_rate

        if do_swap and char in character_replacement:
            # Exchange the character for a randomly selected replacement of nearby keys
            new_string.append(random.choice(character_replacement[char]))
        elif do_delete:
            # Don't include this character in the replacement string
            continue
        elif do_duplicate:
            # Add this character twice to the new string
            new_string.extend([char] * 2)
        elif do_transpose and len(new_string) > 0:
            # Swap this and the previous character
            new_string.append(new_string[-1])
            new_string[-2] = char
        else:
            # Keep the character
            new_string.append(char)

    # if an empty string is generated, give it another go
    if len(new_string) == 0:
        return generate_typo(s, sub_rate, del_rate, dupe_rate, transpose_rate)

    return ''.join(new_string)


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
    spacy_ger = spacy.load("de")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hiddens, cells = model.decoder(
                previous_word, outputs_encoder, hiddens, cells
            )
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


if __name__ == "__main__":
    pass
