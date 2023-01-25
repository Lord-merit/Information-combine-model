
import tensorflow as tf
import time
import numpy as np



class DataLoader(object):
    # Extra vocabulary symbols
    _PAD = '_PAD'
    _GO = '_GO'
    _EOS = '_EOS'
    _UNK = '_UNK'


    extra_tokens = [_PAD, _GO, _EOS, _UNK]

    pad_token = extra_tokens.index(_PAD)
    start_token = extra_tokens.index(_GO)  # start_token = 1
    end_token = extra_tokens.index(_EOS)  # end_token = 2
    unk_token = extra_tokens.index(_UNK)  # unknown = 3

    def __init__(self, data_dir, limits):
        self.train_data_path = [data_dir + '/train/train.summary.id', data_dir + '/train/train.box.val.id',
                                data_dir + '/train/train.box.lab.id', data_dir + '/train/train.box.pos',
                                data_dir + '/train/train.box.rpos']
        self.test_data_path = [data_dir + '/test/test.summary.id', data_dir + '/test/test.box.val.id',
                               data_dir + '/test/test.box.lab.id', data_dir + '/test/test.box.pos',
                               data_dir + '/test/test.box.rpos']
        self.valid_data_path = [data_dir + '/valid/valid.summary.id', data_dir + '/valid/valid.box.val.id',
                              data_dir + '/valid/valid.box.lab.id', data_dir + '/valid/valid.box.pos',
                              data_dir + '/valid/valid.box.rpos']
        self.limits = limits
        self.man_text_len = 100
        start_time = time.time()

        print('Reading datasets ...')
        self.train_set = self.load_data(self.train_data_path)
        print("train is done")
        self.test_set = self.load_data(self.test_data_path)
        print("test is done")
        # self.small_test_set = self.load_data(self.small_test_data_path)
        self.valid_set = self.load_data(self.valid_data_path)
        print("validation is done")
        print ('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))

    def load_data(self, path):
        summary_path, text_path, field_path, pos_path, rpos_path = path
        summaries = open(summary_path, 'r').read().strip().split('\n')
        texts = open(text_path, 'r').read().strip().split('\n')
        fields = open(field_path, 'r').read().strip().split('\n')
        poses = open(pos_path, 'r').read().strip().split('\n')
        rposes = open(rpos_path, 'r').read().strip().split('\n')

        if self.limits > 0:
            summaries = summaries[:self.limits]
            texts = texts[:self.limits]
            fields = fields[:self.limits]
            poses = poses[:self.limits]
            rposes = rposes[:self.limits]

        summaries = [list(map(int, summary.strip().split(' '))) for summary in summaries]
        texts = [list(map(int, text.strip().split(' '))) for text in texts]
        fields = [list(map(int, field.strip().split(' '))) for field in fields]
        poses = [list(map(int, pos.strip().split(' '))) for pos in poses]
        rposes = [list(map(int, rpos.strip().split(' '))) for rpos in rposes]
        return summaries, texts, fields, poses, rposes

    def batch_iter(self, data, batch_size, shuffle):
        summaries, texts, fields, poses, rposes = data
        data_size = len(summaries)
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            summaries = np.array(summaries)[shuffle_indices]
            texts = np.array(texts)[shuffle_indices]
            fields = np.array(fields)[shuffle_indices]
            poses = np.array(poses)[shuffle_indices]
            rposes = np.array(rposes)[shuffle_indices]

        for batch_num in range(num_batches):
            print("batchnum ",batch_num)
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            max_summary_len = max([len(sample) for sample in summaries[start_index:end_index]])
            max_text_len = max([len(sample) for sample in texts[start_index:end_index]])
            batch_data = {'enc_in':[], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[], 'enc_len':[],
                          'dec_in':[], 'dec_len':[], 'dec_out':[]}

            for summary, text, field, pos, rpos in zip(summaries[start_index:end_index], texts[start_index:end_index],
                                            fields[start_index:end_index], poses[start_index:end_index],
                                            rposes[start_index:end_index]):
                summary_len = len(summary)
                text_len = len(text)
                pos_len = len(pos)
                rpos_len = len(rpos)
                assert text_len == len(field)
                assert pos_len == len(field)
                assert rpos_len == pos_len
                gold = summary + [2] + [0] * (max_summary_len - summary_len)
                summary = summary + [0] * (max_summary_len - summary_len)
                text = text + [0] * (max_text_len - text_len)
                field = field + [0] * (max_text_len - text_len)
                pos = pos + [0] * (max_text_len - text_len)
                rpos = rpos + [0] * (max_text_len - text_len)
                
                if max_text_len > self.man_text_len:
                    text = text[:self.man_text_len]
                    field = field[:self.man_text_len]
                    pos = pos[:self.man_text_len]
                    rpos = rpos[:self.man_text_len]
                    text_len = min(text_len, self.man_text_len)
                
                batch_data['enc_in'].append(text)
                batch_data['enc_len'].append(text_len)
                batch_data['enc_fd'].append(field)
                batch_data['enc_pos'].append(pos)
                batch_data['enc_rpos'].append(rpos)
                batch_data['dec_in'].append(summary)
                batch_data['dec_len'].append(summary_len)
                batch_data['dec_out'].append(gold)
  
            yield batch_data

    # batch preparation of a given sequence pair for training
    def prepare_train_batch(self, seqs_x, seqs_f, seqs_p, seqs_rp, seqs_y, maxlen=None):

        # seqs_x, seq_f, seq_p, seq_rp, seqs_y: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        lengths_f = [len(s) for s in seqs_f]
        lengths_p = [len(s) for s in seqs_p]
        lengths_rp = [len(s) for s in seqs_rp]
        lengths_y = [len(s) for s in seqs_y]

        if maxlen is not None:
            new_seqs_x = []
            new_seqs_f = []
            new_seqs_p = []
            new_seqs_rp = []
            new_seqs_y = []
            new_lengths_x = []
            new_lengths_f = []
            new_lengths_p = []
            new_lengths_rp = []
            new_lengths_y = []
            for l_x, s_x, l_f, s_f, l_p,s_p, l_rp, s_rp, l_y, s_y in zip(lengths_x, seqs_x, lengths_f, seqs_f, lengths_p, seqs_p,lengths_rp,seqs_rp, lengths_y, seqs_y):
                if l_x <= maxlen and l_f <= maxlen and l_p <= maxlen and l_rp <= maxlen and l_y <= maxlen:
                    new_seqs_x.append(s_x)
                    new_lengths_x.append(l_x)
                    new_seqs_f.append(s_f)
                    new_lengths_f.append(l_f)
                    new_seqs_p.append(s_p)
                    new_lengths_p.append(l_p)
                    new_seqs_rp.append(s_rp)
                    new_lengths_rp.append(l_rp)
                    new_seqs_y.append(s_y)
                    new_lengths_y.append(l_y)

            lengths_x = new_lengths_x
            seqs_x = new_seqs_x
            lengths_f = new_lengths_f
            seqs_f = new_seqs_f
            lengths_p = new_lengths_p
            seqs_p = new_seqs_p
            lengths_rp = new_lengths_rp
            seqs_rp = new_seqs_rp
            lengths_y = new_lengths_y
            seqs_y = new_seqs_y

            if len(lengths_x) < 1 or len(lengths_f) < 1 or len(lengths_p) < 1 or len(lengths_rp) < 1 or len(lengths_y) < 1:
                return None, None, None, None, None, None, None, None, None, None

        batch_size = len(seqs_x)

        x_lengths = np.array(lengths_x)
        f_lengths = np.array(lengths_f)
        p_lengths = np.array(lengths_p)
        rp_lengths = np.array(lengths_rp)
        y_lengths = np.array(lengths_y)

        maxlen_x = np.max(x_lengths)
        maxlen_f = np.max(f_lengths)
        maxlen_p = np.max(p_lengths)
        maxlen_rp = np.max(rp_lengths)
        maxlen_y = np.max(y_lengths)

        x = np.ones((batch_size, maxlen_x)).astype('int32') *  self.end_token
        f = np.ones((batch_size, maxlen_f)).astype('int32') * self.end_token
        p = np.ones((batch_size, maxlen_p)).astype('int32') * self.end_token
        rp = np.ones((batch_size, maxlen_rp)).astype('int32') * self.end_token
        y = np.ones((batch_size, maxlen_y)).astype('int32') * self.end_token

        for idx, [s_x, s_f, s_p, s_rp, s_y] in enumerate(zip(seqs_x, seqs_f, seqs_p, seqs_rp, seqs_y)):
            x[idx, :lengths_x[idx]] = s_x
            f[idx, :lengths_f[idx]] = s_f
            p[idx, :lengths_p[idx]] = s_p
            rp[idx, :lengths_rp[idx]] = s_rp
            y[idx, :lengths_y[idx]] = s_y
        #print("train box val id ", x) --OPEN
        #print("length", x_lengths) --OPEN
        #print("summary val id", y) --OPEN

        return x, x_lengths,f, f_lengths,p, p_lengths,rp, rp_lengths, y, y_lengths

    # batch preparation of a given sequence pair for testing
    def prepare_test_batch(self, seqs_x, seqs_f, seqs_p, seqs_rp, maxlen=None):

        # seqs_x, seq_f, seq_p, seq_rp: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        lengths_f = [len(s) for s in seqs_f]
        lengths_p = [len(s) for s in seqs_p]
        lengths_rp = [len(s) for s in seqs_rp]

        if maxlen is not None:
            new_seqs_x = []
            new_seqs_f = []
            new_seqs_p = []
            new_seqs_rp = []
            new_lengths_x = []
            new_lengths_f = []
            new_lengths_p = []
            new_lengths_rp = []
            for l_x, s_x, l_f, s_f, l_p,s_p, l_rp, s_rp in zip(lengths_x, seqs_x, lengths_f, seqs_f, lengths_p, seqs_p,lengths_rp,seqs_rp):
                if l_x <= maxlen and l_f <= maxlen and l_p <= maxlen and l_rp <= maxlen:
                    new_seqs_x.append(s_x)
                    new_lengths_x.append(l_x)
                    new_seqs_f.append(s_f)
                    new_lengths_f.append(l_f)
                    new_seqs_p.append(s_p)
                    new_lengths_p.append(l_p)
                    new_seqs_rp.append(s_rp)
                    new_lengths_rp.append(l_rp)

            lengths_x = new_lengths_x
            seqs_x = new_seqs_x
            lengths_f = new_lengths_f
            seqs_f = new_seqs_f
            lengths_p = new_lengths_p
            seqs_p = new_seqs_p
            lengths_rp = new_lengths_rp
            seqs_rp = new_seqs_rp

            if len(lengths_x) < 1 or len(lengths_f) < 1 or len(lengths_p) < 1 or len(lengths_rp) < 1:
                return None, None, None, None, None, None, None, None

        batch_size = len(seqs_x)

        x_lengths = np.array(lengths_x)
        f_lengths = np.array(lengths_f)
        p_lengths = np.array(lengths_p)
        rp_lengths = np.array(lengths_rp)

        maxlen_x = np.max(x_lengths)
        maxlen_f = np.max(f_lengths)
        maxlen_p = np.max(p_lengths)
        maxlen_rp = np.max(rp_lengths)

        x = np.ones((batch_size, maxlen_x)).astype('int32') *  self.end_token
        f = np.ones((batch_size, maxlen_f)).astype('int32') * self.end_token
        p = np.ones((batch_size, maxlen_p)).astype('int32') * self.end_token
        rp = np.ones((batch_size, maxlen_rp)).astype('int32') * self.end_token

        for idx, [s_x, s_f, s_p, s_rp] in enumerate(zip(seqs_x, seqs_f, seqs_p, seqs_rp)):
            x[idx, :lengths_x[idx]] = s_x
            f[idx, :lengths_f[idx]] = s_f
            p[idx, :lengths_p[idx]] = s_p
            rp[idx, :lengths_rp[idx]] = s_rp
        return x, x_lengths,f, f_lengths,p, p_lengths,rp, rp_lengths
