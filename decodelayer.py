# -*- coding: utf-8 -*-
from __future__ import absolute_import
from keras.layers.recurrent import LSTM
import numpy as np
from keras import backend as K
import theano
from keras.layers.core import Layer

class StateTransferLSTM(LSTM):

    def __init__(self, state_input=True, **kwargs):
        self.state_outputs = []
        self.state_input = state_input
        super(StateTransferLSTM, self).__init__(**kwargs)

    def  build(self):
        stateful = self.stateful
        self.stateful = stateful or self.state_input or len(self.state_outputs) > 0
        if hasattr(self, 'states'):
            del self.states
        super(StateTransferLSTM, self).build()
        self.stateful = stateful

    def broadcast_state(self, rnns):
        if type(rnns) not in [list, tuple]:
            rnns = [rnns]
        self.state_outputs += rnns
        for rnn in rnns:
            rnn.state_input = self

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitly the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        mask = self.get_output_mask(train)

        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        last_output, outputs, states = K.rnn(self.step, X, initial_states,
                                             go_backwards=self.go_backwards,
                                             masking=mask)
        n = len(states)
        if self.stateful and not self.state_input:
            self.updates = []
            self.updates = []
            for i in range(n):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(n):
                o.updates.append((o.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def set_input_shape(self, shape):

        self._input_shape = shape
        self.build()

class LSTMDecoder(StateTransferLSTM):
    '''
    A basic LSTM decoder. Similar to [1].
    The output of at each timestep is the input to the next timestep.
    The input to the first timestep is the context vector from the encoder.

    Basic equation:
        y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder.

    In addition, the hidden state of the encoder is usually used to initialize the hidden
    state of the decoder. Checkout models.py to see how its done.
    '''
    input_ndim = 2

    def __init__(self, output_length, hidden_dim=None, **kwargs):

        self.output_length = output_length
        self.hidden_dim = hidden_dim
        input_dim = None
        if 'input_dim' in kwargs:
            kwargs['output_dim'] = input_dim
        if 'input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['input_shape'][-1]
        elif 'batch_input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['batch_input_shape'][-1]
        super(LSTMDecoder, self).__init__(**kwargs)
        self.return_sequences = True #Decoder always returns a sequence.
        self.updates = []

    def set_previous(self, layer, connection_map={}):
        '''Connect a layer to its parent in the computational graph.
        '''
        self.previous = layer
        self.build()

    def build(self):
        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim

        self.input = K.placeholder(input_shape)
        if not self.hidden_dim:
            self.hidden_dim = dim
        hdim = self.hidden_dim
        self.output_dim = dim
        outdim = self.output_dim
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (hidden_dim)
            self.states = [None, None]

        self.W_i = self.init((dim, hdim))
        self.U_i = self.inner_init((hdim, hdim))
        self.b_i = K.zeros((hdim))

        self.W_f = self.init((dim, hdim))
        self.U_f = self.inner_init((hdim, hdim))
        self.b_f = self.forget_bias_init((hdim))

        self.W_c = self.init((dim, hdim))
        self.U_c = self.inner_init((hdim, hdim))
        self.b_c = K.zeros((hdim))

        self.W_o = self.init((dim, hdim))
        self.U_o = self.inner_init((hdim, hdim))
        self.b_o = K.zeros((hdim))

        self.W_x = self.init((hdim, outdim))
        self.b_x = K.zeros((dim))

        self.trainable_weights = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_x, self.b_x
        ]

    def reset_states(self):
        assert self.stateful or self.state_input or len(self.state_outputs) > 0, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.hidden_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.hidden_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.hidden_dim)),
                           K.zeros((input_shape[0], self.hidden_dim))]

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, hidden_dim)
        initial_state = K.zeros_like(X)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.hidden_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, hidden_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def _step(self,
              x_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x, b_i, b_f, b_c, b_o, b_x):

        xi_t = K.dot(x_tm1, w_i) + b_i
        xf_t = K.dot(x_tm1, w_f) + b_f
        xc_t = K.dot(x_tm1, w_c) + b_c
        xo_t = K.dot(x_tm1, w_o) + b_o

        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)

        x_t = K.dot(h_t, w_x) + b_x
        return x_t, h_t, c_t

    def get_output(self, train=False):
        x_t = self.get_input(train)
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x_t)
        [outputs, hidden_states, cell_states], updates = theano.scan(
            self._step,
            n_steps=self.output_length,
            outputs_info=[x_t] + initial_states,
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c,
                           self.W_i, self.W_f, self.W_c, self.W_o,
                           self.W_x, self.b_i, self.b_f, self.b_c,
                           self.b_o, self.b_x])

        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))

        return K.permute_dimensions(outputs, (1, 0, 2))

    @property
    def output_shape(self):
        shape = list(super(LSTMDecoder, self).output_shape)
        shape[1] = self.output_length
        return tuple(shape)

    def get_config(self):
        config = {'name': self.__class__.__name__,
        'hidden_dim': self.hidden_dim,
        'output_length': self.output_length}
        base_config = super(LSTMDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMDecoder1(StateTransferLSTM):
    '''
    A basic LSTM decoder. Similar to [1].
    The output of at each timestep is the input to the next timestep.
    The input to the first timestep is the context vector from the encoder.

    Basic equation:
        y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder.

    In addition, the hidden state of the encoder is usually used to initialize the hidden
    state of the decoder. Checkout models.py to see how its done.
    '''
    input_ndim = 2

    def __init__(self, output_length, hidden_dim=None, **kwargs):

        self.output_length = output_length
        self.hidden_dim = hidden_dim
        input_dim = None
        if 'input_dim' in kwargs:
            kwargs['output_dim'] = input_dim
        if 'input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['input_shape'][-1]
        elif 'batch_input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['batch_input_shape'][-1]
        super(LSTMDecoder1, self).__init__(**kwargs)
        self.return_sequences = True #Decoder always returns a sequence.
        self.updates = []

    def set_previous(self, layer, connection_map={}):
        '''Connect a layer to its parent in the computational graph.
        '''
        self.previous = layer
        self.build()

    def build(self):
        input_shape = self.input_shape
        dim = self.hidden_dim
        self.input_dim = dim
        self.output_dim = dim
        self.input = K.placeholder(input_shape)
        hdim = self.hidden_dim

        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (hidden_dim)
            self.states = [None, None]

        self.W_i = self.init((dim, hdim))
        self.U_i = self.inner_init((hdim, hdim))
        self.b_i = K.zeros((hdim))

        self.W_f = self.init((dim, hdim))
        self.U_f = self.inner_init((hdim, hdim))
        self.b_f = self.forget_bias_init((hdim))

        self.W_c = self.init((dim, hdim))
        self.U_c = self.inner_init((hdim, hdim))
        self.b_c = K.zeros((hdim))

        self.W_o = self.init((dim, hdim))
        self.U_o = self.inner_init((hdim, hdim))
        self.b_o = K.zeros((hdim))

        self.W_x = self.init((hdim, dim))
        self.b_x = K.zeros((dim))

        self.trainable_weights = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_x, self.b_x
        ]

    def reset_states(self):
        assert self.stateful or self.state_input or len(self.state_outputs) > 0, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.hidden_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.hidden_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.hidden_dim)),
                           K.zeros((input_shape[0], self.hidden_dim))]

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, hidden_dim)
        initial_state = K.zeros_like(X)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.hidden_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, hidden_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def _step(self,
              x_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x, b_i, b_f, b_c, b_o, b_x):

        xi_t = K.dot(x_tm1, w_i) + b_i
        xf_t = K.dot(x_tm1, w_f) + b_f
        xc_t = K.dot(x_tm1, w_c) + b_c
        xo_t = K.dot(x_tm1, w_o) + b_o

        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)

        x_t = K.dot(h_t, w_x) + b_x
        return x_t, h_t, c_t

    def get_output(self, train=False):
        x_t = self.get_input(train)
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x_t)
        [outputs, hidden_states, cell_states], updates = theano.scan(
            self._step,
            n_steps=self.output_length,
            outputs_info=[x_t] + initial_states,
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c,
                           self.W_i, self.W_f, self.W_c, self.W_o,
                           self.W_x, self.b_i, self.b_f, self.b_c,
                           self.b_o, self.b_x])

        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))

        return K.permute_dimensions(outputs, (1, 0, 2))

    @property
    def output_shape(self):
        shape = list(super(LSTMDecoder1, self).output_shape)
        shape[1] = self.output_length
        return tuple(shape)

    def get_config(self):
        config = {'name': self.__class__.__name__,
        'hidden_dim': self.hidden_dim,
        'output_length': self.output_length}
        base_config = super(LSTMDecoder1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMDecoder2(LSTMDecoder):
    '''
    This decoder is similar to the first one, except that at every timestep the decoder gets
    a peek at the context vector.
    Similar to [2].

    Basic equation:
        y(t) = LSTM(s(t-1), y(t-1), C)
        y(0) = LSTM(s0, C, C)
        Where s is the hidden state of the LSTM (h and c), and C is the context vector
        from the encoder.

    '''
    def build(self):
        super(LSTMDecoder2, self).build()
        dim = self.input_dim
        hdim = self.hidden_dim
        self.V_i = self.init((dim, hdim))
        self.V_f = self.init((dim, hdim))
        self.V_c = self.init((dim, hdim))
        self.V_o = self.init((dim, hdim))
        self.trainable_weights += [self.V_i, self.V_c, self.V_f, self.V_o]

    def sstep(self,
              x_tm1,
              h_tm1, c_tm1, v,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x, v_i, v_f, v_c, v_o, b_i, b_f, b_c, b_o, b_x):

        #Inputs = output from previous time step, vector from encoder
        xi_t = K.dot(x_tm1, w_i) + K.dot(v, v_i) + b_i
        xf_t = K.dot(x_tm1, w_f) + K.dot(v, v_f) + b_f
        xc_t = K.dot(x_tm1, w_c) + K.dot(v, v_c) + b_c
        xo_t = K.dot(x_tm1, w_o) + K.dot(v, v_o) + b_o

        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)

        x_t = K.dot(h_t, w_x) + b_x
        return x_t, h_t, c_t

    def get_output(self, train=False):
        v = self.get_input(train)
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(v)
        [outputs,hidden_states, cell_states], updates = theano.scan(
            self.sstep,
            n_steps = self.output_length,
            outputs_info=[v] + initial_states,
            non_sequences=[v, self.U_i, self.U_f, self.U_o, self.U_c,
                          self.W_i, self.W_f, self.W_c, self.W_o,
                          self.W_x, self.V_i, self.V_f, self.V_c,
                          self.V_o, self.b_i, self.b_f, self.b_c,
                          self.b_o, self.b_x])
        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))

        return K.permute_dimensions(outputs, (1, 0, 2))

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(LSTMDecoder2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMDecoder_tag(LSTMDecoder2):
    input_ndim = 3
    def build(self):
        super(LSTMDecoder_tag, self).build()
        dim = self.input_dim
        hdim = self.hidden_dim
        self.input_length = self.input_shape[-2]
        if not self.input_length:
            raise Exception ('AttentionDecoder requires input_length.')
    def ssstep(self,
               h,
              x_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x,  v_i, v_f, v_c, v_o, b_i, b_f, b_c, b_o, b_x):
        xi_t = K.dot(x_tm1, w_i)+ b_i+ K.dot(h, v_i)
        xf_t = K.dot(x_tm1, w_f)  + b_f+ K.dot(h, v_f)
        xc_t = K.dot(x_tm1, w_c) + b_c+ K.dot(h, v_c)
        xo_t = K.dot(x_tm1, w_o)  + b_o+ K.dot(h, v_o)
        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        x_t = self.activation(K.dot(h_t, w_x) + b_x)
        return x_t, h_t, c_t

    def get_output(self, train=False):
        H = self.get_input(train)
        Hh = K.permute_dimensions(H, (1, 0, 2))
        def rstep(o,index,Hh):
            return Hh[index],index-1
        [RHh,index],update = theano.scan(
        rstep,
        n_steps=Hh.shape[0],
        non_sequences=[Hh],
        outputs_info= [Hh[-1]]+[-1])
        #RHh=K.permute_dimensions(RHh, (1, 0, 2))
        X = K.permute_dimensions(H, (1, 0, 2))[-1]
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)
        [outputs,hidden_states, cell_states], updates = theano.scan(
            self.ssstep,
            sequences=RHh,
            n_steps = self.output_length,
            outputs_info=[X] + initial_states,
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c,
                          self.W_i, self.W_f, self.W_c, self.W_o,
                          self.W_x, self.V_i, self.V_f, self.V_c,
                          self.V_o, self.b_i, self.b_f, self.b_c,
                          self.b_o, self.b_x])
        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))

        return K.permute_dimensions(outputs, (1, 0, 2))


class ReverseLayer2(Layer):
    def __init__(self,layer):
        self.layer=layer
        super(ReverseLayer2, self).__init__()

    @property
    def output_shape(self,train=False):
        return self.layer.output_shape

    def get_output(self, train=False):
         b=self.layer.get_output(train)
         a=b.dimshuffle((1, 0, 2))
         def rstep(o,index,H):
            return H[index],index-1
         [results,index],update = theano.scan(
         rstep,
        n_steps=a.shape[0],
        non_sequences=[a],
        outputs_info= [a[-1]]+[-1])
         results2=results.dimshuffle((1, 0, 2))
         return results2

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(ReverseLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_input(self, train=False):
        res = []
        o = self.layer.get_input(train)
        if not type(o) == list:
            o = [o]
        for output in o:
            if output not in res:
                res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None


class MergeLayerShape(Layer):
    def __init__(self,layers,zeroindex=1,concat_axis=-1,batchsize=50):
        self.layers=layers
        self.index = zeroindex
        self.concat_axis = concat_axis
        self.shape1 = layers[0].output_shape
        self.shape1 = (batchsize,self.shape1[1],self.shape1[2])
        self.shape2 = layers[1].output_shape
        self.shape2 = (batchsize, self.shape2[1], self.shape2[2])
        super(MergeLayerShape, self).__init__()

    def input_shape(self):
        return [layer.input_shape for layer in self.layers]

    @property
    def output_shape(self):
        input_shapes = [layer.output_shape for layer in self.layers]
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            output_shape[self.concat_axis] += shape[self.concat_axis]
        return tuple(output_shape)

    def get_output(self, train=False):
        l1 = self.layers[0].get_output(train)
        l2 = self.layers[1].get_output(train)
        if self.index == 0:
            l1shape = K.zeros(self.shape1)
            return K.concatenate([l1shape,l2], axis=self.concat_axis)
        else:
            l2shape = K.zeros(self.shape2)
            return K.concatenate([l1, l2shape], axis=self.concat_axis)

    def get_input(self, train=False):
        res = []
        for i in range(len(self.layers)):
            o = self.layers[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = []
        for l in self.layers:
            weights += l.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].trainable_weights)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(MergeLayerShape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

