import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):

    def __init__(self, pretrained_embeddings, num_layers, bidirectional, outputs, hidden_size, dropout_value,
                 embeddings_size):
        """
        Creates LSTM model

        :param pretrained_embeddings: pretrained embeddings
        :param num_layers: number of layers
        :param bidirectional: if True - bidirectional LSTM is used, unidirectional otherwise
        :param outputs: number of outputs
        :param hidden_size: size of hidden state
        :param dropout_value: value of dropout for regularization
        :param embeddings_size: dimension of embeddings
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.embeddings_size = embeddings_size
        self.dropout_value = dropout_value
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)  # embedding layer

        # LSTM layer
        self.lstm = nn.LSTM(embeddings_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_value)
        self.fc = nn.Linear(self.hidden_size * (1 if not bidirectional else 2), outputs)

    def forward(self, input_ids, input_ids_lengths):
        """
        Performs a forward pass through the network

        :param input_ids: input tensor containing ids of tokens
        :param input_ids_lengths: lengths of inputs
        :return: result of propagation
        """
        text_emb = self.embedding(input_ids)
        text_emb = self.dropout(text_emb)

        packed_input = pack_padded_sequence(text_emb, input_ids_lengths, batch_first=True)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), input_ids_lengths - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        out = self.dropout(out_reduced)

        out = self.fc(out)
        out = torch.squeeze(out, 1)

        return out
