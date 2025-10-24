import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.autograd import Function

class Adapter(nn.Module):
    def __init__(self, d_model, layers):
        super(Adapter, self).__init__()
        # self.init_trans = nn.Linear(d_model, layers[0])
        self.layers = nn.Sequential()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(nn.ReLU())
        # self.end_trans = nn.Linear(layers[-1], d_model)

    def forward(self, x):
        # x = self.init_trans(x)
        x = self.layers(x)
        # for layer in self.layers:
        #     x = layer(x)
        # x = self.end_trans(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layers = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.drop_out(src2)
        src = self.norm1(src)

        src2 = self.layers(src)
        src = src + src2
        src = self.norm2(src)

        return src

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Fusion(nn.Module):
    def __init__(self, d_model):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            # nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, eeg, eye):
        feature = torch.cat([eeg, eye], dim=-1)
        return self.fusion(feature)

class Avarage_Fusion(nn.Module):
    def __init__(self, d_model):
        super(Avarage_Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def forward(self, feature_list):
        feature_num = len(feature_list)
        sum = 0
        for feature in feature_list:
            sum += feature
        fused_feature = sum / feature_num
        fused_feature = self.fusion(fused_feature)
        return fused_feature

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class VQEmbedding(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, beta=1.0):
        super(VQEmbedding, self).__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = 0.99
        self.epsilon = 1e-5
        self.commitment_cost = 0.25
        init_bound = 1 / n_embeddings

        eeg_embedding = torch.Tensor(n_embeddings, embedding_dim)
        eeg_embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("eeg_embedding", eeg_embedding)
        self.register_buffer("eeg_ema_count", torch.zeros(n_embeddings))
        self.register_buffer("eeg_ema_embedding", self.eeg_embedding.clone())


    def eeg_vq(self, eeg_z):
        B, T, D = eeg_z.size()
        eeg_z_flat = eeg_z.contiguous().view(-1, D)
        distance = torch.sum(eeg_z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.eeg_embedding ** 2, dim=1) - 2 * \
            torch.matmul(eeg_z_flat, self.eeg_embedding.t())
        indices = torch.argmin(distance, dim=-1)
        encodings = F.one_hot(indices, self.n_embeddings).float()
        eeg_zq = torch.matmul(encodings, self.eeg_embedding).view_as(eeg_z)
        eeg_zq = eeg_z + (eeg_zq - eeg_z).detach()
        return eeg_zq


    def forward(self, eeg_semantic,):
        M, D = self.eeg_embedding.size()
        B, T, D = eeg_semantic.size()
        eeg_flat = eeg_semantic.contiguous().reshape(-1, D)


        eeg_distances = torch.addmm(torch.sum(self.eeg_embedding ** 2, dim=1) +
                                    torch.sum(eeg_flat ** 2, dim=1, keepdim=True),
                                    eeg_flat, self.eeg_embedding.t(),
                                    alpha=-2.0, beta=1.0)

        eeg_encoding_indices = torch.argmin(eeg_distances, dim=-1)
        eeg_encodings = F.one_hot(eeg_encoding_indices, M).float()
        eeg_quantized = torch.matmul(eeg_encodings, self.eeg_embedding).view_as(eeg_semantic)

        if self.training:
            eeg_ema_count = self.decay * self.eeg_ema_count + (1 - self.decay) * torch.sum(eeg_encodings, dim=0)
            n = torch.sum(eeg_ema_count)
            eeg_ema_count = (eeg_ema_count + self.epsilon) / (n + M * self.epsilon) * n
            self.eeg_ema_count = eeg_ema_count
            eeg2eeg_dw = torch.matmul(eeg_encodings.float().t(), eeg_flat)
            self.eeg_ema_embedding = self.eeg_ema_embedding * self.decay + (1 - self.decay) * eeg2eeg_dw
            self.eeg_embedding = self.eeg_ema_embedding / self.eeg_ema_count.unsqueeze(1)

        # eeg_e_latent_loss = F.mse_loss(eeg_semantic, eeg_quantized.detach())
        # vq_loss = self.commitment_cost * eeg_e_latent_loss
        eeg_quantized = eeg_semantic + (eeg_quantized - eeg_semantic).detach()

        return eeg_quantized

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, head_num=4, num_layers=3):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=head_num)
        self.encoder = Encoder(self.encoder_layer, num_layers=num_layers)
        self.affine_matrix = nn.Linear(input_dim, d_model)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)
        return feature

class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(Decoder, self).__init__()
        self.recon_linear = nn.Linear(embedding_dim, output_dim * 2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(output_dim * 2, output_dim)

    def forward(self, z_q):
        recon = self.recon_linear(z_q)
        recon = self.relu(recon)
        recon = self.linear(recon)
        return recon

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, head_num=4, num_layers=3):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = EncoderLayer(d_model=d_model, nhead=head_num)
        self.decoder = Encoder(self.decoder_layer, num_layers=num_layers)
        self.affine_matrix = nn.Linear(input_dim, d_model)

    def forward(self, feature):
        feature = self.affine_matrix(feature)
        feature = self.decoder(feature)
        return feature

class DomainClassifier(nn.Module):
    def __init__(self, embedding_dim, domain_num):
        super(DomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, domain_num)
        )
    def forward(self, feature):
        domain_pred = self.domain_classifier(feature)

        return domain_pred

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0

        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        B, seq_len, D = x.size()
        q = self.query(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_score, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, seq_len, D)

        return output

class Classifier(nn.Module):
    def __init__(self, embedding_dim, class_num):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, 512)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(512)
        self.linear2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, class_num)

    def forward(self, z_q):
        hidden1 = self.linear1(z_q)
        hidden1 = self.relu1(hidden1)
        hidden1 = self.norm1(hidden1)
        hidden2 = self.linear2(hidden1)
        hidden2 = self.relu2(hidden2)
        hidden2 = self.norm2(hidden2)
        pred = self.classifier(hidden2)

        return pred

def calculate_entropy(output):
    # probabilities = F.softmax(output, dim=0)
    probabilities = F.softmax(output, dim=1)
    log_probabilities = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    return entropy

def calculate_gating_weights(encoder_output_1, encoder_output_2):
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)

    # print("entropy1:", entropy_1.shape)
    # print("entropy2:", entropy_2.shape)

    max_entropy = torch.max(entropy_1, entropy_2)

    gating_weight_1 = torch.exp(max_entropy - entropy_1)
    gating_weight_2 = torch.exp(max_entropy - entropy_2)

    sum_weights = gating_weight_1 + gating_weight_2

    gating_weight_1 /= sum_weights
    gating_weight_2 /= sum_weights

    return gating_weight_1, gating_weight_2

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        # query: [batch_size, query_len, embed_dim]
        # key, value: [batch_size, key_len, embed_dim]
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        output = self.norm(query + attn_output)  # Add & Norm
        return output



if __name__ == '__main__':
    batch_size, query_len, key_len, embed_dim = 32, 5, 5, 64
    query = torch.randn(batch_size, query_len, embed_dim)
    key = torch.randn(batch_size, key_len, embed_dim)
    value = torch.randn(batch_size, key_len, embed_dim)

    cross_attention = CrossAttention(embed_dim=embed_dim, num_heads=4)
    out = cross_attention(query, key, value)
    print(out.shape)