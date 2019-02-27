
import os
import re
import pandas as pd

from const import LINE_BREAK
from pprint import pprint as pp

df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'chat_logs.csv'))

HOW_MANY_OURS = 4
our_user_ids = list(df.groupby('user_id').count().reset_index([0]).sort_values('chat_id', ascending=False)[:HOW_MANY_OURS]['user_id'])

def get_qa_pairs(df, consume_all=True):
    qa_pairs = []

    for chat_id in df['chat_id'].unique():
        threads = df.loc[df['chat_id'] == chat_id]
        answers = []
        questions = []

        # making qa pair
        for i, thread in threads.iterrows():
            # ours
            if thread['user_id'] in our_user_ids:
                if thread['message'].strip().endswith('?'):
                    questions.append(thread['message'])
                else:
                    answers.append(thread['message'])
            # theirs
            else:
                questions.append(thread['message'])

            if len(questions) > 0 or len(answers) > 0:
                while len(questions) > 0 and len(answers) > 0:
                    qa_pairs.append((questions.pop(), answers.pop(),))

        # consume remainders
        if consume_all is True:
            for block in [questions, answers]:
                while len(block) > 0:
                    try:
                        qa_pairs.append((block.pop(0), block.pop(0),))
                    except IndexError:
                        pass
    return qa_pairs

def write_to_tsv(file_name, pairs, replace_line_break=True):
    with open(file_name, 'w') as out_file:
        for q, a in pairs:
            out_file.write('{0}\t{1}\n'.format(q, a.replace('\n', LINE_BREAK) if replace_line_break else a))
            out_file.flush()

qa_pairs = get_qa_pairs(df)
write_to_tsv(os.path.join(os.getcwd(), 'data', 'chat_logs_pairs.tsv'), qa_pairs)
print('done.')
# import pdb; pdb.set_trace()