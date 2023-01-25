import pickle
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
#from OutputUnit import OutputUnit

class SeqUnitUni(object):
    def __init__(self, dataloader, config, mode):

        '''
        batch_size, hidden_size, embedding_size, field_size, pos_size: size of batch; hidden layer; word/field/position embedding
        num_encoder_symbols, num_decoder_symbols, field_vocab, position_vocab: vocabulary size of encoder words; decoder words; field types; position
        field_concat, position_concat: bool values, whether concat field/position embedding to word embedding for encoder inputs or not
        dual_att: bool values, whether use field-gating / dual attention or not
        encoder_add_pos, decoder_add_pos: bool values, whether add position embedding to field-gating encoder / decoder with dual attention or not
        '''
        self.dataloader = dataloader
        self.batch_size = config['batch_size']
        self.hidden_units = config['hidden_units']
        self.embedding_size = config['embedding_size']
        self.field_size = config['field_size']
        self.pos_size = config['pos_size']
        # for key, value in config.items():
        #    print (key, "=", value._value)
        self.uni_size = config['embedding_size'] if not config['field_concat'] else config[
                                                                                                      'embedding_size'] + \
                                                                                                  config[
                                                                                                      'field_size']
        self.uni_size = self.uni_size if not config['position_concat'] else self.uni_size + 2 * config[
            'pos_size']
        self.num_encoder_symbols = config['num_encoder_symbols']
        self.num_decoder_symbols = config['num_decoder_symbols']
        self.num_encoder_field_symbols = config['num_encoder_field_symbols']
        self.num_encoder_position_symbols = config['num_encoder_position_symbols']
        self.grad_clip = 5.0

        # self.start_token = start_token
        # self.stop_token = stop_token
        # self.max_length = max_length
        # self.scope_name = scope_name
        # self.name = name
        self.field_concat = config['field_concat']
        self.position_concat = config['position_concat']
        self.use_beamsearch_decode = config['use_beamsearch_decode']
        self.mode = mode
        self.attn_input_feeding = config['attn_input_feeding']
        self.optimizer = config['optimizer']
        # self.dual_att = config['dual_att']
        self.depth = config['depth']
        self.cell_type = config['cell_type']
        self.attention_type = config['attention_type']
        self.use_dropout = config['use_dropout']
        self.dropout_rate = config['dropout_rate']
        self.use_fp16 = config['use_fp16']
        self.keep_prob = 1.0 - self.dropout_rate
        self.keep_prob_placeholder = tf.placeholder(tf.float32, shape=[], name='keep_pro')
        self.use_residual = config['use_residual']
        # self.encoder_add_pos = config['encoder_add_pos']
        # self.decoder_add_pos = config['decoder_add_pos']
        self.learning_rate = config['learning_rate']
        self.max_gradient_norm = config['max_gradient_norm']
        self.units = {}
        self.params = {}
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        # self.use_beamsearch_decode = False
        if self.mode == 'test':
            self.beam_width = config['beam_width']
            self.use_beamsearch_decode = True if self.beam_width > 1 else False
            self.max_decode_step = config['max_decode_step']

        # self.encoder_input = tf.placeholder(tf.int32, [None, None])
        # self.encoder_field = tf.placeholder(tf.int32, [None, None])
        # self.encoder_pos = tf.placeholder(tf.int32, [None, None])
        # self.encoder_rpos = tf.placeholder(tf.int32, [None, None])
        # self.encoder_len = tf.placeholder(tf.int32, [None])
        # self.decoder_input = tf.placeholder(tf.int32, [None, None])
        # self.decoder_len = tf.placeholder(tf.int32, [None])
        # self.decoder_output = tf.placeholder(tf.int32, [None, None]) --BU ASAGIDA NEREDE????
        # self.enc_mask = tf.sign(tf.to_float(self.encoder_pos))

        if self.use_fp16 == True:
            self.dtype = tf.float16
        else:
            self.dtype = tf.float32
        print("building model..")

        # Building encoder and decoder networks
        self.init_placeholders()
        with tf.device('/gpu:0'):
            self.build_encoder()
        with tf.device('/gpu:1'):
            self.build_decoder()
        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self):
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=(None, None), name='encoder_inputs')

        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='encoder_inputs_length')

        self.encoder_field = tf.placeholder(dtype=tf.int32,
                                             shape=(None, None), name='encoder_fields')
        self.encoder_pos = tf.placeholder(dtype=tf.int32,
                                             shape=(None, None), name='encoder_leftpos')
        self.encoder_rpos = tf.placeholder(dtype=tf.int32,
                                             shape=(None, None), name='encoder_rightpos')

        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        if self.mode == 'train':
            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name='decoder_inputs')
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

            decoder_start_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * self.dataloader.start_token
            decoder_end_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * self.dataloader.end_token

            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            self.decoder_inputs_train = tf.concat([decoder_start_token,
                                                  self.decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            # decoder_targets_train: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                                   decoder_end_token], axis=1)

    ##### -------------------- ENCODER ------------------------------ #######

    def build_encoder(self):
        print("building encoder..")
        with tf.variable_scope('encoder'):
            # Building encoder_cell
            self.encoder_cell = self.build_encoder_cell()

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.encoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.num_encoder_symbols, self.embedding_size],
                                                      initializer=initializer, dtype=self.dtype)
            self.encoder_field_embeddings = tf.get_variable(name='field_embedding',
                                                      shape=[self.num_encoder_field_symbols, self.field_size],
                                                      initializer=initializer, dtype=self.dtype)
            self.encoder_pos_embeddings = tf.get_variable(name='pos_embedding',
                                                       shape=[self.num_encoder_position_symbols, self.pos_size],
                                                       initializer=initializer, dtype=self.dtype)
            self.encoder_rpos_embeddings = tf.get_variable(name='rpos_embedding',
                                                       shape=[self.num_encoder_position_symbols, self.pos_size],
                                                       initializer=initializer, dtype=self.dtype)

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=self.encoder_inputs)
            self.encoder_field_embedded = tf.nn.embedding_lookup(
                params=self.encoder_field_embeddings, ids=self.encoder_field)
            self.encoder_pos_embedded = tf.nn.embedding_lookup(
                params=self.encoder_pos_embeddings, ids=self.encoder_pos)
            self.encoder_rpos_embedded = tf.nn.embedding_lookup(
                params=self.encoder_rpos_embeddings, ids=self.encoder_rpos)

            if self.field_concat:
                self.encoder_inputs_embedded = tf.concat([self.encoder_inputs_embedded, self.encoder_field_embedded], 2)
            if self.position_concat:
                self.encoder_inputs_embedded = tf.concat([self.encoder_inputs_embedded, self.encoder_pos_embedded, self.encoder_rpos_embedded], 2)

            if self.field_concat:
                self.params.update({'field_embedding': self.encoder_field_embeddings})
            if self.position_concat:
                self.params.update({'pos_embedding': self.encoder_pos_embeddings})
                self.params.update({'rpos_embedding': self.encoder_rpos_embeddings})
            self.params.update({'embedding': self.encoder_embeddings})

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            with tf.device('/gpu:0'):
                self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                    cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                    time_major=False)

    # Building encoder cell
    def build_encoder_cell (self):
        with tf.device('/gpu:0'):
            return MultiRNNCell([self.build_single_cell() for i in range(self.depth)])

    def build_single_cell(self):
        cell_type = LSTMCell
        if (self.cell_type.lower() == 'gru'):
            cell_type = GRUCell

        cell = cell_type(self.hidden_units)
        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder, )
        if self.use_residual:
            cell = ResidualWrapper(cell)

        return cell

    ##### -------------------- ENCODER ------------------------------ #######

    ##### -------------------- DECODER ------------------------------ #######
    def build_decoder(self):
        print("building decoder and attention..")
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()
            # print("encoder_output ", self.encoder_outputs)
            # print("encoder_state ", self.encoder_last_state)
            # print("encoder_inputs_length", self.encoder_inputs_length)
            # print("decoder_cell ", self.decoder_cell)
            # print("initial state", self.decoder_initial_state)

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.decoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.num_decoder_symbols, self.embedding_size],
                                                      initializer=initializer, dtype=self.dtype)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.num_decoder_symbols, name='output_projection')

            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings, ids=self.decoder_inputs_train)

                # Embedded inputs having gone through input projection layer
                self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                         sequence_length=self.decoder_inputs_length_train,
                                                         time_major=False,
                                                         name='training_helper')
                with tf.device('/gpu:1'):
                    training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                            helper=training_helper,
                                                            initial_state=self.decoder_initial_state,
                                                            output_layer=output_layer)

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                with tf.device('/gpu:1'):
                    (self.decoder_outputs_train, self.decoder_last_state_train,
                     self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                        decoder=training_decoder,
                        output_time_major=False,
                        impute_finished=True,
                        maximum_iterations=max_decoder_length))

                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                    name='decoder_pred_train')

                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                with tf.device('/gpu:1'):
                    self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                      targets=self.decoder_targets_train,
                                                      weights=masks,
                                                      average_across_timesteps=True,
                                                      average_across_batch=True, )
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)

                # Construct graphs for minimizing loss
                self.init_optimizer()

            elif self.mode == 'test':
                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.dataloader.start_token
                end_token = self.dataloader.end_token

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    with tf.device('/gpu:1'):
                        decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                        end_token=end_token,
                                                                        embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    print("building greedy decoder..")
                    with tf.device('/gpu:1'):
                        inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                 helper=decoding_helper,
                                                                 initial_state=self.decoder_initial_state,
                                                                 output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately find the most likely translation
                    print("building beamsearch decoder..")
                    with tf.device('/gpu:1'):
                        inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                                                  embedding=embed_and_input_proj,
                                                                                  start_tokens=start_tokens,
                                                                                  end_token=end_token,
                                                                                  initial_state=self.decoder_initial_state,
                                                                                  beam_width=self.beam_width,
                                                                                  output_layer=output_layer, )
                with tf.device('/gpu:0'):
                    (self.decoder_outputs_decode, self.decoder_last_state_decode,self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                        decoder=inference_decoder,
                        output_time_major=False,
                        maximum_iterations=self.max_decode_step))

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids
        # Building decoder cell and attention. Also returns decoder_initial_state

    def build_decoder_cell(self):

        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:
            print("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        with tf.device('/gpu:0'):
            self.attention_mechanism = attention_wrapper.BahdanauAttention(
                num_units=self.hidden_units, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length, )
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        if self.attention_type.lower() == 'luong':
            with tf.device('/gpu:0'):
                self.attention_mechanism = attention_wrapper.LuongAttention(
                    num_units=self.hidden_units, memory=encoder_outputs,
                    memory_sequence_length=encoder_inputs_length, )

        # Building decoder_cell
        self.decoder_cell_list = [
            self.build_single_cell() for i in range(self.depth)]
        decoder_initial_state = encoder_last_state

        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(self.hidden_units, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        with tf.device('/gpu:0'):
             self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
                 cell=self.decoder_cell_list[-1],
                 attention_mechanism=self.attention_mechanism,
                 attention_layer_size=self.hidden_units,
                 cell_input_fn=attn_decoder_input_fn,
                 initial_cell_state=encoder_last_state[-1],
                 alignment_history=False,
                 name='Attention_Wrapper')
        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode \
            else self.batch_size * self.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
            batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)

        with tf.device('/gpu:0'):
            #return self.decoder_cell_list, decoder_initial_state
            return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

    def init_optimizer(self):
        print("setting optimizer..")
        # Gradients and SGD update operation for training the model
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    ##### -------------------- DECODER ------------------------------ #######

    ##### -------------------- TRAINING ------------------------------ #######
    def train(self, sess, encoder_inputs, encoder_inputs_length, encoder_field, encoder_pos, encoder_rpos,
              decoder_inputs, decoder_inputs_length):
        """Run a train step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          encoder_field: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder field inputs
          encoder_pos: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder pos inputs
          encoder_rpos: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder right pos input
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length, encoder_field, encoder_pos, encoder_rpos,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        output_feed = [self.updates,  # Update Op that does optimization
                       self.loss,  # Loss for current batch
                       self.summary_op]  # Training summary

        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]  # loss, summary

    def check_feeds(self, encoder_inputs, encoder_inputs_length,encoder_field, encoder_pos, encoder_rpos,
                    decoder_inputs, decoder_inputs_length, test):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          encoder_field: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder field inputs
          encoder_pos: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder pos inputs
          encoder_rpos: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder right pos input
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                             "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not test:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                                 "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                                 "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length
        input_feed[self.encoder_field] = encoder_field
        input_feed[self.encoder_pos] = encoder_pos
        input_feed[self.encoder_rpos] = encoder_rpos
        if not test:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed


 ##### -------------------- TRAINING ------------------------------ #######
 ##### -------------------- VALIDATING & PREDICTING------------------------------ #######
    def eval(self, sess, encoder_inputs, encoder_inputs_length, encoder_field, encoder_pos, encoder_rpos,
              decoder_inputs, decoder_inputs_length):
        """Run a train step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          encoder_field: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder field inputs
          encoder_pos: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder pos inputs
          encoder_rpos: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder right pos input
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

"""
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length, encoder_field, encoder_pos, encoder_rpos,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.loss,  # Loss for current batch
                       self.summary_op]  # Training summary
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]  # loss

    def predict(self, sess, encoder_inputs, encoder_inputs_length, encoder_field, encoder_pos, encoder_rpos):

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length, encoder_field, encoder_pos, encoder_rpos,
                                      decoder_inputs=None, decoder_inputs_length=None,
                                      test=True)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.decoder_pred_decode]
        #print('output_feed', output_feed)
        outputs = sess.run(output_feed, input_feed)

        # GreedyDecoder: [batch_size, max_time_step]
        return outputs[0]  # BeamSearchDecoder: [batch_size, max_time_step, beam_width]

##### -------------------- VALIDATING & PREDICTING ------------------------------ #######
##### -------------------- SAVING & RESTORING ------------------------------ #######

    def save(self, sess, path, var_list=None, global_step=None):
        saver = tf.train.Saver(var_list)
        #save_path = saver.save(sess, save_path=path, global_step=global_step)
        save_path = saver.save(sess, save_path=path)

        print('model saved at %s' % save_path)

    def restore(self, sess, path, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)



