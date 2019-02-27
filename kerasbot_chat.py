import os
import traceback
import numpy as np
from lib import load_data
from keras.models import Model, load_model
from keras.layers import Input

from preprocessing.const import START_SEQ, END_SEQ

latent_dim = 256
data = load_data(os.environ.get('KB_DATA_PATH', 'chat_logs_dec_r.txt'), int(os.environ.get('KB_DATA_SIZE', 10000)))

model = load_model('s2s.h5')

encoder_inputs = model.input[0] #input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1] #input_2
decoder_state_input_h = Input(shape=(latent_dim,),name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,),name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs=decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in data.input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in data.target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, data.num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, data.target_token_index[START_SEQ]] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == END_SEQ or len(decoded_sentence) > data.max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, data.num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def predict(input_text):
    # Take one sequence (part of the training set)
    # for trying out decoding.

    encoder_input_data = np.zeros((1, len(input_text), encoder_inputs.shape[-1].value), dtype='float32')
    for t, char in enumerate(input_text):
        encoder_input_data[0, t, data.input_token_index[char]] = 1.

    decoded_sentence = decode_sequence(encoder_input_data)
    print('-')
    print('Input sentence:', input_text)
    print('Decoded sentence:', decoded_sentence)


try:
    while True:
        inp = input('what: ')
        predict(inp)
except:
    traceback.print_exc()