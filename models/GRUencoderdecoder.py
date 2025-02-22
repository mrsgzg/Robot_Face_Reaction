import torch
import torch.nn as nn

class GRUEncoderDecoder(nn.Module):
    def __init__(self, 
                 speaker_input_dim=161, 
                 audio_input_dim=39, 
                 output_dim=161, 
                 hidden_dim=128, 
                 num_layers=1):
        """
        Args:
            speaker_input_dim (int): Dimensionality of speaker face features.
            audio_input_dim (int): Dimensionality of audio features (e.g., MFCC).
            output_dim (int): Dimensionality of the output listener face features.
            hidden_dim (int): Hidden state dimensionality for all GRUs.
            num_layers (int): Number of GRU layers.
        """
        super(GRUEncoderDecoder, self).__init__()
        
        # GRU encoder for speaker face features
        self.speaker_encoder = nn.GRU(
            input_size=speaker_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # GRU encoder for audio features
        self.audio_encoder = nn.GRU(
            input_size=audio_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Fusion: Concatenate the final hidden states of both encoders and map to hidden_dim
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # GRU decoder: expects inputs (e.g., previous target tokens during teacher forcing)
        self.decoder = nn.GRU(
            input_size=output_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Linear layer to map decoder hidden states to the output listener face feature dimension
        self.out_fc = nn.Linear(hidden_dim, output_dim)

        print("finish init model")

    def forward(self, speaker_expr, speaker_mfcc, decoder_inputs):
        """
        Forward pass of the GRU-EncoderDecoder model.
        
        Args:
            speaker_expr (Tensor): Speaker face features, shape [batch, seq_len, 161].
            speaker_mfcc (Tensor): Speaker audio features (MFCC), shape [batch, seq_len, 39].
            decoder_inputs (Tensor): Inputs for the decoder (e.g., previous ground truth tokens during teacher forcing),
                                     shape [batch, seq_len, output_dim].
        
        Returns:
            Tensor: Generated listener face features, shape [batch, seq_len, output_dim].
        """
        batch_size = speaker_expr.size(0)
        
        # Encode speaker face features
        _, speaker_hidden = self.speaker_encoder(speaker_expr)  
        # Use final layer's hidden state: shape [batch, hidden_dim]
        speaker_hidden = speaker_hidden[-1]
        
        # Encode speaker audio features (MFCC)
        _, audio_hidden = self.audio_encoder(speaker_mfcc)
        audio_hidden = audio_hidden[-1]
        
        # Fuse the hidden states from both encoders
        fused = torch.cat((speaker_hidden, audio_hidden), dim=-1)  # [batch, hidden_dim * 2]
        decoder_init = torch.tanh(self.fusion_fc(fused))           # [batch, hidden_dim]
        
        # Prepare initial hidden state for the decoder: expand to [num_layers, batch, hidden_dim]
        decoder_init = decoder_init.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        # Decode to generate listener face features
        decoder_outputs, _ = self.decoder(decoder_inputs, decoder_init)
        outputs = self.out_fc(decoder_outputs)  # [batch, seq_len, output_dim]
        
        return outputs

