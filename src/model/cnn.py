import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, pretrained_embeddings, outputs, dropout_value, embeddings_size, num_filters,
                 filter_sizes):
        """
        Creates CNN model

        :param pretrained_embeddings: pretrained embeddings
        :param outputs: number of outputs
        :param dropout_value: value of dropout for regularization
        :param embeddings_size: dimension of embeddings
        :param num_filters: number of filters of each size
        :param filter_sizes: list of specific filter sizes
        """
        super(CNN, self).__init__()

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embeddings_size = embeddings_size
        self.dropout_value = dropout_value
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embeddings_size, out_channels=num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])

        self.fc = nn.Linear(num_filters * len(filter_sizes), outputs)
        self.dropout = nn.Dropout(p=dropout_value)

    def forward(self, input_ids, _input_ids_lengths=None):
        """
        Performs a forward pass through the network

        :param input_ids: input tensor containing ids of tokens
        :param _input_ids_lengths: lengths of inputs - not used
        :return: result of propagation
        """
        del _input_ids_lengths

        x_embed = self.embedding(input_ids)

        x_reshaped = x_embed.permute(0, 2, 1)

        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        out = self.fc(self.dropout(x_fc))

        return out
