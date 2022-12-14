
'''Concrete implementations of abstract base classes.'''

import torch

from abcs import EncoderBase, DecoderBase, EncoderDecoderBase


# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):

    def init_submodules(self):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.rnn, self.embedding
        # 2. You will need these object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}

        self.embedding = torch.nn.Embedding(self.source_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)
        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(self.word_embedding_size,
                                     self.hidden_state_size,
                                     num_layers=self.num_hidden_layers,
                                     dropout=self.dropout,
                                     bidirectional=True)
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(self.word_embedding_size,
                                    self.hidden_state_size,
                                    num_layers=self.num_hidden_layers,
                                    dropout=self.dropout,
                                    bidirectional=True)
        elif self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(self.word_embedding_size,
                                    self.hidden_state_size,
                                    num_layers=self.num_hidden_layers,
                                    dropout=self.dropout,
                                    bidirectional=True)

    def forward_pass(self, F, F_lens, h_pad=0.):
        # Recall:
        #   F is size (S, M)
        #   F_lens is of size (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use these methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states

        x = self.get_all_rnn_inputs(F)
        h = self.get_all_hidden_states(x, F_lens, h_pad)
        return h

    def get_all_rnn_inputs(self, F):
        # Recall:
        #   F is size (S, M)
        #   x (output) is size (S, M, I)

        x = self.embedding(F)
        return x

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # Recall:
        #   x is of size (S, M, I)
        #   F_lens is of size (M,)
        #   h_pad is a float
        #   h (output) is of size (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence

        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)

        if self.cell_type == 'lstm':
            h_packed = self.rnn(x_packed)[0]
        else:
            _, M, _ = x.size()
            h_0 = torch.zeros(self.num_hidden_layers * 2, M, self.hidden_state_size, device=x.device)
            h_packed = self.rnn(x_packed, h_0)[0]

        h = torch.nn.utils.rnn.pad_packed_sequence(h_packed,
                                                   padding_value=h_pad)[0]
        return h


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}

        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(self.word_embedding_size,
                                          self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(self.word_embedding_size,
                                         self.hidden_state_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(self.word_embedding_size,
                                         self.hidden_state_size)

        self.ff = torch.nn.Linear(self.hidden_state_size,
                                  self.target_vocab_size)

    def forward_pass(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of size (M,)
        #   htilde_tm1 is of size (M, 2 * H)
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   logits_t (output) is of size (M, V)
        #   htilde_t (output) is of same size as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use these methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.
        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)
        if self.cell_type == 'lstm':
            logits_t = self.get_current_logits(htilde_t[0])
        else:
            logits_t = self.get_current_logits(htilde_t)
        return logits_t, htilde_t

    def get_first_hidden_state(self, h, F_lens):
        # Recall:
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   htilde_tm1 (output) is of size (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch functions: torch.cat

        h_forward = h[F_lens - 1, torch.arange(len(F_lens)), 0: self.hidden_state_size // 2]
        h_backward = h[0, :, self.hidden_state_size//2:self.hidden_state_size]
        htilde_tm1 = torch.cat([h_forward, h_backward], dim=1)
        return htilde_tm1

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of size (M,)
        #   htilde_tm1 is of size (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   xtilde_t (output) is of size (M, Itilde)

        xtilde_t = self.embedding(E_tm1)
        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # Recall:
        #   xtilde_t is of size (M, Itilde)
        #   htilde_tm1 is of size (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same size as htilde_tm1

        htilde_t = self.cell(xtilde_t, htilde_tm1)
        return htilde_t

    def get_current_logits(self, htilde_t):
        # Recall:
        #   htilde_t is of size (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of size (M, V)

        logits_t = self.ff(htilde_t)
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.

        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(self.word_embedding_size + self.hidden_state_size,
                                          self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(self.word_embedding_size + self.hidden_state_size,
                                         self.hidden_state_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(self.word_embedding_size + self.hidden_state_size,
                                         self.hidden_state_size)

        self.ff = torch.nn.Linear(self.hidden_state_size,
                                  self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        # Hint: For this time, the hidden states should be initialized to zeros.
        htilde_tm1 = torch.zeros(h.shape[1], self.hidden_state_size, device=h.device)
        return htilde_tm1

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Hint: Use attend() for c_t
        x_t = self.embedding(E_tm1)
        c_t = self.attend(htilde_tm1, h, F_lens)
        xtilde_t = torch.cat([x_t, c_t], dim=1)
        return xtilde_t

    def attend(self, htilde_t, h, F_lens):
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of size ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of size ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of size ``(M, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        if self.cell_type == 'lstm':
            htilde_t = htilde_t[0]
        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)
        c_t = torch.sum(alpha_t.unsqueeze(2) * h, dim=0)
        return c_t

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of size (S, M)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Recall:
        #   htilde_t is of size (M, 2 * H)
        #   h is of size (S, M, 2 * H)
        #   e_t (output) is of size (S, M)
        #
        # Hint:
        # Relevant pytorch functions: torch.nn.functional.cosine_similarity
        htilde_t = htilde_t.unsqueeze(0)
        e_t = torch.nn.functional.cosine_similarity(htilde_t, h, dim=-1)
        return e_t


class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not modify this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize these submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need these object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        # 6. You do *NOT* need self.heads at this point
        self.W = torch.nn.Linear(self.hidden_state_size,
                                 self.hidden_state_size,
                                 bias=False)
        self.Wtilde = torch.nn.Linear(self.hidden_state_size,
                                      self.hidden_state_size,
                                      bias=False)
        self.Q = torch.nn.Linear(self.hidden_state_size,
                                 self.hidden_state_size,
                                 bias=False)

    def attend(self, htilde_t, h, F_lens):
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch functions:
        #   tensor().repeat_interleave, tensor().view
        # 3. You *WILL* need self.heads at this point
        # 4. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        S = h.shape[0]
        if self.cell_type == 'lstm':
            M = htilde_t[0].shape[0]
            htilde_t_input = (self.Wtilde(htilde_t[0]).view(M * self.heads, int(self.hidden_state_size / self.heads)), htilde_t[1])
        else:
            M = htilde_t.shape[0]
            htilde_t_input = self.Wtilde(htilde_t).view(M * self.heads, int(self.hidden_state_size / self.heads))
        h_input = self.W(h).view(S, M * self.heads, int(self.hidden_state_size / self.heads))
        F_lens_input = F_lens.repeat_interleave(self.heads)
        c_t = (super().attend(htilde_t_input, h_input, F_lens_input)).view(M, self.hidden_state_size)
        return self.Q(c_t)


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need these object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it
        self.encoder = encoder_class(self.source_vocab_size,
                                     pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     num_hidden_layers=self.encoder_num_hidden_layers,
                                     hidden_state_size=self.encoder_hidden_size,
                                     dropout=self.encoder_dropout,
                                     cell_type=self.cell_type)
        self.decoder = decoder_class(self.target_vocab_size,
                                     pad_id=self.target_eos,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size * 2,
                                     cell_type=self.cell_type,
                                     heads=self.heads)

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # Recall:
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   E is of size (T, M)
        #   logits (output) is of size (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        htilde_tm1 = self.decoder.get_first_hidden_state(h, F_lens)
        if self.cell_type == 'lstm':
            htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))
        T = len(E)
        logits_lst = []
        for i in range(T-1):
            logits_t, htilde_tm1 = self.decoder(E[i], htilde_tm1, h, F_lens)
            logits_lst.append(logits_t)
        logits = torch.stack(logits_lst)
        return logits

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of size (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of size (M, K)
        #   b_tm1_1 is of size (t, M, K)
        #   b_t_0 (first output) is of size (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of size (t + 1, M, K)
        #   logpb_t (third output) is of size (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of size z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]
        extensions_t = (logpb_tm1.unsqueeze(-1) + logpy_t).squeeze(1)
        extensions_flat = torch.flatten(extensions_t, start_dim=1)
        logpb_t, indices = torch.topk(extensions_flat, k=self.beam_width, dim=1)
        paths = indices // self.target_vocab_size
        tokens = indices % self.target_vocab_size
        b_tm1_1_gathered = torch.gather(b_tm1_1, 2, paths.expand_as(b_tm1_1).long())
        b_t_1 = torch.cat([b_tm1_1_gathered, tokens.unsqueeze(0)], dim=0)
        if self.cell_type == 'lstm':
            b_t_0 = (torch.gather(htilde_t[0], 1, paths.unsqueeze(-1).expand_as(htilde_t[0]).long()),
                     torch.gather(htilde_t[1], 1, paths.unsqueeze(-1).expand_as(htilde_t[1]).long()))
        else:
            b_t_0 = torch.gather(htilde_t, 1, paths.unsqueeze(-1).expand_as(htilde_t).long())
        return b_t_0, b_t_1, logpb_t
