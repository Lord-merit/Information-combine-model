
import sys
import os
import tensorflow as tf
import json
import time
from SeqUnitUni import *
from SeqUnitBi import *
from SeqUnitStacked import *
from DataLoader import DataLoader
import numpy as np
import util as utility
from PythonROUGE import PythonROUGE
from ROUGE2 import FilesRouge
#from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import *
from collections import OrderedDict

tf.app.flags.DEFINE_string('model_type', 'bi', 'Unidirectional or Bidirectional Model: uni or bi or stacked')
tf.app.flags.DEFINE_integer("hidden_units", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("embedding_size", 400, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("max_epochs",1, "Number of training epoch.")
tf.app.flags.DEFINE_integer("num_encoder_symbols", 20004,'vocabulary size')
tf.app.flags.DEFINE_integer("num_decoder_symbols",20004,'target vocabulary size')
tf.app.flags.DEFINE_integer("num_encoder_field_symbols", 419,'vocabulary size')
tf.app.flags.DEFINE_integer("num_encoder_position_symbols", 100,'position vocabulary size')

tf.app.flags.DEFINE_integer("display_freq", 15,'report valid results after some steps')
tf.app.flags.DEFINE_integer('save_freq', 1150, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 1150, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')
tf.app.flags.DEFINE_integer('depth',3, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_string('cell_type', 'gru', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')
tf.app.flags.DEFINE_boolean('use_residual', True, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('use_beamsearch_decode', False, 'Use beam search in decode phase')
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')
tf.app.flags.DEFINE_integer('beam_width', 5, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('decode_batch_size', 1, 'Batch size used for testing')
tf.app.flags.DEFINE_integer('max_decode_step', 500, 'Maximum time step limit to test')
tf.app.flags.DEFINE_boolean('write_n_best', True, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_integer('max_seq_length', 100, 'Maximum sequence length')
tf.app.flags.DEFINE_string('model_name', 'model', 'File name used for model checkpoints')

tf.app.flags.DEFINE_string("mode",'train','train or test')
tf.app.flags.DEFINE_string("load",'0','load directory')
#tf.app.flags.DEFINE_string("mode",'test','train or test')
#tf.app.flags.DEFINE_string("load",'1669409409339','load directory')

tf.app.flags.DEFINE_string("dir",'processed_data','data set directory')
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')
tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')

tf.app.flags.DEFINE_boolean("field_concat", True,'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position_concat", True, 'concat position information to word embedding')
tf.app.flags.DEFINE_integer('num_gpus', 2,'How many GPUs to use.')

FLAGS = tf.app.flags.FLAGS
last_best = 0.0

# test phase
if FLAGS.load != "0":
    save_dir = 'results/res/' + FLAGS.load + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + FLAGS.load + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    sum_beam_path = pred_dir + 'original_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'

# train phase
else:
    prefix = str(int(time.time() * 1000))
    save_dir = 'results/res/' + prefix + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + prefix + '/'
    os.mkdir(save_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    sum_beam_path = pred_dir + 'original_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'

log_file = save_dir + 'log.txt'
eval_log_file = pred_dir + 'evallog.txt'

######-------------- MODEL ------------######

def create_model(session, dataloader, FLAGS):
    config = OrderedDict(sorted(FLAGS.__flags.items()))

    if FLAGS.__flags['model_type'] == 'uni':
        model = SeqUnitUni(dataloader, config, 'train')
    elif FLAGS.__flags['model_type'] == 'bi':
        model = SeqUnitBi(dataloader, config, 'train')
    else:
        model = SeqUnitStacked(dataloader, config, 'train')

    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('Created new model parameters..')
        session.run(tf.global_variables_initializer())

    return model

######-------------- TRAIN ------------######

def train(sess, dataloader, model):
        # Create a log writer object
        log_writer = tf.summary.FileWriter(save_dir, graph=sess.graph)
        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()

        # Training loop
        print ('Training..')
        write_log("#######################################################")
        for flag in FLAGS.__flags:
            write_log(flag + " = " + str(FLAGS.__flags[flag]))
        write_log("#######################################################")
        trainset = dataloader.train_set

        for epoch_idx in range(FLAGS.max_epochs):
            print('Epoch Num',epoch_idx+1)
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print ('Training is already complete.', \
                      'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs))
                break

            for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True):
                if x['enc_in'] is None or x['dec_in'] is None:
                    print ('No samples under max_seq_length ', FLAGS.max_seq_length)
                    continue
                #obtain a batch
                batch_source, batch_source_len, batch_field, batch_field_len, batch_pos, batch_pos_len, batch_rpos, batch_rpos_len, batch_target, batch_target_len = dataloader.prepare_train_batch(x['enc_in'], x['enc_fd'], x['enc_pos'], x['enc_rpos'], x['dec_in'], FLAGS.max_seq_length)
                 # Execute a single training step
                step_loss, summary = model.train(sess, batch_source,batch_source_len, batch_field, batch_pos,  batch_rpos, batch_target, batch_target_len)

                loss += float(step_loss) / FLAGS.display_freq
                words_seen += float(np.sum(batch_source_len + batch_target_len))
                sents_seen += float(batch_source.shape[0])  # batch_size

                progress_bar(model.global_step.eval() % FLAGS.display_freq, FLAGS.display_freq)

                if model.global_step.eval() % FLAGS.display_freq == 0:
                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / FLAGS.display_freq

                    words_per_sec = words_seen / time_elapsed
                    sents_per_sec = sents_seen / time_elapsed

                    print('Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(), \
                         'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time ', step_time, \
                        '{0:.2f} sents/s'.format(sents_per_sec), '{0:.2f} words/s'.format(words_per_sec))

                    write_log("%d : loss = %.3f, time = %.3f " % ( model.global_step.eval() // FLAGS.display_freq, loss, step_time))
                    loss = 0
                    words_seen = 0
                    sents_seen = 0
                    start_time = time.time()

                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # Execute a validation step
                if dataloader.valid_set and model.global_step.eval() % FLAGS.valid_freq == 0:
                    print ('Development step')
                    validset = dataloader.valid_set
                    valid_loss = 0.0
                    valid_sents_seen = 0

                    for x in dataloader.batch_iter(validset, FLAGS.batch_size, True):
                        # Get a batch from validation parallel data
                        val_source, val_source_len, val_field, val_field_len, val_pos, val_pos_len, val_rpos, val_rpos_len, val_target, val_target_len = dataloader.prepare_train_batch(
                            x['enc_in'], x['enc_fd'], x['enc_pos'], x['enc_rpos'], x['dec_in'], FLAGS.max_seq_length)

                        # Compute validation loss: average per word cross entropy loss
                        step_loss, summary = model.eval(sess, val_source, val_source_len, val_field, val_pos,
                                                     val_rpos, val_target, val_target_len)

                        batch_size = val_source.shape[0]

                        valid_loss += step_loss * batch_size
                        valid_sents_seen += batch_size
                        print ('  {} samples seen'.format(valid_sents_seen))

                    valid_loss = valid_loss / valid_sents_seen
                    print ('Valid perplexity: {0:.2f}'.format(math.exp(valid_loss)))

                # Save the model checkpoint
                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print ('Saving the model..')
                    checkpoint_path = os.path.join(save_dir, FLAGS.model_name)
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    json.dump(FLAGS.__flags,
                            open('%s.json' % (checkpoint_path), 'w'),
                            indent=2)

            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            print ('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))

        print ('Saving the last model..')
        checkpoint_path = os.path.join(save_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(FLAGS.__flags,
                  open('%s.json' % (checkpoint_path), 'w'),
                  indent=2)

        print('Training Terminated')

def write_log(s):
    print (s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')

def init_evallog(s):
    print (s)
    with open(eval_log_file, 'w') as f:
         f.write(s + '\n')

def write_evallog(s):
    print (s)
    with open(eval_log_file, 'a') as f:
         f.write(s + '\n')

def load_config():
    modelpath=save_dir+"model"
    print(modelpath)
    config = unicode_to_utf8(
        json.load(open('%s.json' % modelpath, 'r')))
    return config


def load_model(session, dataloader, FLAGS):
    # Load model config
    config_dict = load_config()
    print('config', config_dict)
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True,
                                          gpu_options=tf.GPUOptions(allow_growth=True,
                                          visible_device_list='0,1'))) as sess:
         # Reload existing checkpoint
         if FLAGS.__flags['model_type'] == 'uni':
             model = SeqUnitUni(dataloader, config_dict, 'test')
         elif FLAGS.__flags['model_type'] == 'bi':
             model = SeqUnitBi(dataloader, config_dict, 'test')
         else:
             model = SeqUnitStacked(dataloader, config_dict, 'test')

         model_path=save_dir+FLAGS.model_name
         print('model ', model_path)
         if tf.train.checkpoint_exists(model_path):
           print('Reloading model parameters..')
           model.restore(session, model_path)
         else:
           raise ValueError(
            'No such file:[{}]'.format(model_path))
    return model

def test(sess, dataloader, model):

    # Load source data to decode
    testset = dataloader.test_set

    # Load inverse dictionary used in decoding
    vocab = Vocab()

    try:
        print ('Decoding........')
        if FLAGS.write_n_best:
            fout = [utility.fopen(("%s%d" % (pred_beam_path, k)), 'w') \
                    for k in range(FLAGS.beam_width)]
            fout2 = [utility.fopen(("%s%d" % (sum_beam_path, k)), 'w') \
                    for k in range(FLAGS.beam_width)]
        else:
            fout = [utility.fopen(("%s%d" % (pred_beam_path, 0)), 'w')]
            fout2 = [utility.fopen(("%s%d" % (sum_beam_path, 0)), 'w')]

        idx=1
        for x in dataloader.batch_iter(testset, FLAGS.decode_batch_size, False):
            # obtain test data as a single batch
            test_source, test_source_len, test_field, test_field_len, test_pos, test_pos_len, test_rpos, test_rpos_len, test_target, test_target_len = dataloader.prepare_train_batch( #should be prepare_test_batch
                x['enc_in'], x['enc_fd'], x['enc_pos'], x['enc_rpos'], x['dec_in'], FLAGS.max_seq_length)
            # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
            # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
            predicted_ids = model.predict(sess, test_source, test_source_len, test_field,test_pos,test_rpos)

            # Write decoding results
            fileindex=0
            for k, f in reversed(list(enumerate(fout))):
                f2=fout2[fileindex]
                fileindex+=1
                seqindex=0
                for seq in predicted_ids:
                    f.write(str(vocab.seq2words(seq[:, k]))+ '\n')
                    f2.write(str(vocab.seq2words(test_target[seqindex])+ '\n'))
                    seqindex+=1
                    if not FLAGS.write_n_best:
                        break
            print('  {}th line decoded'.format(idx * FLAGS.decode_batch_size))
            idx+=1
    except IOError:
        pass
    finally:
            [f.close() for f in fout]
            [f.close() for f in fout2]
            Metric_scores(pred_beam_path, sum_beam_path)

def Metric_scores(pred_filename, ref_filename):
    init_evallog("\n#######################################################")
    write_evallog("\nBLEU and ROUGE SCORES")
    write_evallog("\nMODEL PARAMETERS")
    write_evallog("--------------------------------------------------------")
    for flag in FLAGS.__flags:
        write_evallog(flag + " = " + str(FLAGS.__flags[flag]))
    write_evallog("--------------------------------------------------------")

    for x in range(0, FLAGS.beam_width):
        filename1=pred_filename+str(x)
        filename2=ref_filename+str(x)
        with open(filename1) as file1:
            lines = []
            for line in file1:
                line=line.rstrip('\n')
                lines.append(list(line.split()))

        with open(filename2) as file2:
            lines2 = []
            for line in file2:
                line=line.rstrip('\n')
                lines2.append([line.split()])

        recall, precision, f_measure = PythonROUGE(pred_dir, [filename1],[[filename2]], ngram_order=4)
        all_scores = FilesRouge().get_scores(filename1, filename2, avg=True)
        #bleu = corpus_bleu(lines2,lines,smoothing_function=SmoothingFunction().method1)
        copy_result = "File %s & %s with original data: \n\nROUGE-PRECISION: %s \t ROUGE-RECALL: %s ROUGE-FMEASURE: %s \n" % \
                   (filename1, filename2,  str(precision), str(recall), str(f_measure))
        copy_result2 = "Second ROUGE: \n ROUGE-L: %s \n ROUGE-1: %s \n ROUGE-2: %s \n ROUGE-3: %s \n ROUGE-4: %s \n" % \
                      (str(all_scores['rouge-l']),str(all_scores['rouge-1']),str(all_scores['rouge-2']),str(all_scores['rouge-3']),str(all_scores['rouge-4']))
        write_evallog(copy_result)
        write_evallog(copy_result2)

        file1.close()
        file2.close()
    write_evallog("#######################################################")




######-------------- TRAIN AND TEST ------------######
def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True,
                                          gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list = '0,1'))) as sess:
        copy_file(save_file_dir)
        dataloader = DataLoader(FLAGS.dir, FLAGS.limits)

        if FLAGS.mode == 'train':
            # Create a new model
            model = create_model(sess, dataloader, FLAGS)
            # Train the model
            train(sess, dataloader, model)
        else:
            # Reload an existing model
            model = load_model(sess,dataloader,FLAGS)
            # Test the model
            test(sess, dataloader, model)


if __name__=='__main__':
    with tf.device('/gpu:1'):
        main()