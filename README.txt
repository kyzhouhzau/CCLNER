Abstract:This tool was develop for NER by three model
#BiLSTM+CRF    https://arxiv.org/abs/1508.01991
#BiLSTM+lstm   https://arxiv.org/pdf/1706.05075.pdf
#BiLSTM+Attention this model is often used in machine translation and Semantic generation.
#indrnn+CRF   C:\Users\zhouk\PycharmProjects\TensorFlow\TEST_lstm_MODEL\0_0_1\BiLSTM_decode\README.txt
I use F score for each tags' evaluate.

This Model still need some Improve.


##Part1:data_preprocess.py was used to preprocess raw data.
            raw data type:


            The O
            dog B-animal
            is O
            fly B-action

            The O
            dog B-animal
            is O
            fly B-action

            The O
            dog B-animal
            is O
            fly B-action


##Part2:AEP_model.py was use to train defferent model.
##eval_model.py was used to eval our model.


Directionary :rnn_cell contain
My_LSTM.py ,this was a LSTM cell realized by some tensorflow ops.
My_LSTM.py ,this was a attention cell realized by others.
indrnn.py ,this realize inrnn from https://github.com/batzner/indrnn
Others ,not important



