from fairseq.models import FairseqEncoderDecoderModel, FairseqEncoder


class FairseqContextModel(FairseqEncoderDecoderModel):
    """
    Base class for encoder-decoder models accepting a context
    which is encoded with a proper encoder.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
        context_encoder (FairseqEncoder): the encoder for the context
    """
    def __init__(self, encoder, decoder, context_encoder):
        super().__init__(encoder, decoder)
        assert isinstance(context_encoder, FairseqEncoder)
        self.context_encoder = context_encoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, context_tokens, context_lengths, **kwargs):
        """
        Run the forward pass for an encoder-decoder model with context.

        First, encode the context. Then, feed a batch of source tokens and the
        encoded context through the encoder. Lastly, feed the encoder output, the
        encoded context and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            context_enc = self.context_encoder(context_tokens, context_lengths)
            encoder_out = self.encoder(src_tokens, src_lengths, context_enc)
            return self.decoder(prev_output_tokens, encoder_out, context_enc)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            context_tokens (LongTensor): tokens representing the context of shape
                `(batch, ctx_len)`
            context_lengths (LongTensor): context lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        context_out = self.context_encoder(context_tokens, src_lengths=context_lengths, **kwargs)
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, context_out=context_out, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, context_out=context_out, **kwargs)
        return decoder_out

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, context_tokens, context_lengths, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        context_out = self.context_encoder(context_tokens, src_lengths=context_tokens, **kwargs)
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, context_out=context_out, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, context_out=context_out, **kwargs)
        return features
