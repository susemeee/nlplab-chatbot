
from preprocessing.const import LINE_BREAK, END_SEQ, START_SEQ

class Data:
    def __init__(self):
        pass

def load_data(data_path, data_size=10000):
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:min(data_size, len(lines) - 2)]):
            input_text, target_text = line.split('\t')
            # We use START_SEQ as the "start sequence" character
            # for the targets, and END_SEQ as "end sequence" character.
            target_text = START_SEQ + target_text.replace(LINE_BREAK, '\n') + END_SEQ
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

    data = Data()
    data.input_texts = input_texts
    data.target_texts = target_texts
    data.input_characters = sorted(list(input_characters))
    data.target_characters = sorted(list(target_characters))
    data.num_encoder_tokens = len(input_characters)
    data.num_decoder_tokens = len(target_characters)
    data.max_encoder_seq_length = max([len(txt) for txt in input_texts])
    data.max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', data.num_encoder_tokens)
    print('Number of unique output tokens:', data.num_decoder_tokens)
    print('Max sequence length for inputs:', data.max_encoder_seq_length)
    print('Max sequence length for outputs:', data.max_decoder_seq_length)

    data.input_token_index = dict(
        [(char, i) for i, char in enumerate(data.input_characters)])
    data.target_token_index = dict(
        [(char, i) for i, char in enumerate(data.target_characters)])

    return data
