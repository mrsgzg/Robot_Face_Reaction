
def get_model(model_name, **kwargs):
    """
    Factory function to create a model instance based on the given model name.

    Args:
        model_name (str): The type of model to create. Options include:
            - "gru_encoderdecoder"
            - "attention"
            - "transformer"
        **kwargs: Additional keyword arguments for model configuration.

    Returns:
        nn.Module: An instance of the requested model.
    """
    model_name = model_name.lower()
    
    if model_name == "gru_encoderdecoder":
        
        from models.GRUencoderdecoder import GRUEncoderDecoder
        
        return GRUEncoderDecoder()
    #elif model_name == "attention":
    #    from models.attention import AttentionModel  # Ensure AttentionModel exists in models/attention.py
    #    return AttentionModel(**kwargs)
    #elif model_name == "transformer":
    #    from models.Transformer import TransformerModel  # Ensure TransformerModel exists in models/Transformer.py
    #    return TransformerModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

#if __name__ == "__main__":
#    # Quick test: instantiate a default GRUEncoderDecoder model.
#    model = get_model("gru_encoderdecoder")
#    print(model)
