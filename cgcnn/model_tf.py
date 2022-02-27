from __future__ import print_function, division

# import torch
# import torch.nn as nn
from tensorflow import keras
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

class ConvLayer(tf.keras.Model):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        tf.keras.backend.set_floatx('float64')
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        # input size : 2*self.atom_fea_len+self.nbr_fea_len
        self.fc_full = tf.keras.layers.Dense(2*self.atom_fea_len)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.softplus1 = tf.keras.layers.Activation('softplus')
        self.softplus2 = tf.keras.layers.Activation('softplus')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.c = 0
        
    def call(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = tf.gather(atom_in_fea, tf.cast(nbr_fea_idx, tf.int64))
        total_nbr_fea = tf.concat([tf.broadcast_to(tf.expand_dims(atom_in_fea, axis=1), [N, M, self.atom_fea_len]), atom_nbr_fea, nbr_fea], axis=2)
        # keras
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = tf.reshape(self.bn1(tf.reshape(total_gated_fea, [-1, self.atom_fea_len*2])), [N, M, self.atom_fea_len*2])
    
        nbr_filter, nbr_core =tf.split(total_gated_fea, num_or_size_splits=2, axis=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = tf.reduce_sum(nbr_filter * nbr_core, axis=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out
        


class CrystalGraphConvNet(tf.keras.Model):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        
        #self.fc_full = tf.keras.layers.Dense(2*self.atom_fea_len)
        #self.sigmoid = tf.keras.layers.Activation('sigmoid')
        #self.softplus1 = tf.keras.layers.Activation('softplus')
        #self.softplus2 = tf.keras.layers.Activation('softplus')
        #self.bn1 = tf.keras.layers.BatchNormalization()
        #self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.classification = classification
        self.embedding = tf.keras.layers.Dense(atom_fea_len)
        self.convs = [ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)]
        self.conv_to_fc = tf.keras.layers.Dense(h_fea_len)
        self.conv_to_fc_softplus = tf.keras.layers.Activation('softplus')
        if n_h > 1:
            self.fcs = [tf.keras.layers.Dense(h_fea_len, h_fea_len) for _ in range(n_h-1)]
            self.softpluses = [tf.keras.layers.Activation('softplus') for _ in range(n_h-1)]
        if self.classification:
            self.fc_out = tf.keras.layers.Dense(2)
        else:
            self.fc_out = tf.keras.layers.Dense(1)
        if self.classification:
            self.logsoftmax = tf.nn.log_softmax
            self.dropout = tf.keras.layers.Dropout(rate=0.5)
            #self.logsoftmax = tf.nn.log_softmax(dim=1)
            #self.dropout = nn.Dropout()

    def call(self, inputs):
    #def call(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs[0], inputs[1], inputs[2], inputs[3]
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out, axis=1)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.shape[0]
        summed_fea = [tf.math.reduce_mean(tf.gather(atom_fea,idx_map), axis=0, keepdims=True)
                      for idx_map in crystal_atom_idx]
        return tf.concat(summed_fea, axis=0)