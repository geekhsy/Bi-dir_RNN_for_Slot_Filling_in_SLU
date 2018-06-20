# coding=utf-8
import tensorflow as tf
from data import *
from modelRNN import Model_basic,Model_LSTM
from Metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
import time
import sys
import progressbar
from argparse import ArgumentParser

input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = 871
slot_size = 125
intent_size = 22
epoch_num = 40
fold=0
cell=''


def get_model():
    model = Model_LSTM(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size, n_layers) if cell=="LSTM" else Model_basic(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size, n_layers)
    model.build()
    return model

def train(is_debug=False):
    print ('Folder:'+str(fold))
    model = get_model()
    sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())

    train_data,test_data = get_data(fold)

    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)

    word2index, slot2index,  intent2index,  = get_info_from_training_data(train_data_ed)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    
    best_ep=0
    best_sl_acc=0
    best_f1_score=0

    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        tic = time.time()

        bar = progressbar.ProgressBar(maxval=(len(index_train)/batch_size) ,widgets=[("[Epoch {}] >>Training  ".format(epoch)),progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i, batch in enumerate(getBatch(batch_size, index_train)):
            # Perform a batch training
            _, loss, decoder_prediction, intent, mask, slot_W = model.step(sess, "train", batch)
            mean_loss += loss
            train_loss += loss
            train_loss /= (i + 1)
            bar.update(i+1)
        bar.finish()
        sys.stdout.flush()
        print('Training completed in {:.2f} (sec)'.format(time.time()-tic))
        # One epoch per training, test once
        pred_slots = []
        slot_accs = []
        for j, batch in enumerate(getBatch(batch_size, index_test)):
            decoder_prediction, intent = model.step(sess, "test", batch)
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            slot_pred_length = list(np.shape(decoder_prediction))[1]
            pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps-slot_pred_length)),
                                     mode="constant", constant_values=0)
            pred_slots.append(pred_padded)
            true_slot = np.array((list(zip(*batch))[2]))
            true_length = np.array((list(zip(*batch))[1]))
            true_slot = true_slot[:, :slot_pred_length]
            slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
            slot_accs.append(slot_acc)
        pred_slots_a = np.vstack(pred_slots)
        true_slots_a = np.array(list(zip(*index_test))[2])[:pred_slots_a.shape[0]]
        f1_score=f1_for_sequence_batch(true_slots_a, pred_slots_a)
        print("Slot accuracy for epoch {}: {:.3f}".format(epoch, np.average(slot_accs)*100))
        print("Slot F1 score for epoch {}: {:.3f}".format(epoch,f1_score*100 ))
        if (f1_score >best_f1_score):
            best_ep=epoch
            best_sl_acc=np.average(slot_accs)
            best_f1_score=f1_score
    print('\nBEST RESULT: epoch {}, valid accurasy {:.3f}, best test F1 score {:.3f}'.format(best_ep,best_sl_acc*100,best_f1_score*100))

    sess.close()
    with open('results.txt', 'a') as outfile:
        outfile.write('For Folder:'+str(fold)+' using '+str(cell)+' cell, BEST RESULT: epoch '+ str(best_ep)+ ', valid score {:.3f}, best test F1 score {:.3f}'.format( best_sl_acc*100,best_f1_score*100)+'\n')

if __name__ == '__main__':
    parser = ArgumentParser(description='Provide RNN Cell. Use either BasicRNN, or LSTM as arguments.')
	# parser.add_argument('N', type=int,help="Problem size", metavar="N")
    parser.add_argument("Model", default='BasicRNN',type=str,
						help="BasicRNN or LSTM algorithm")
    args = parser.parse_args()
    cell=args.Model
    train()
