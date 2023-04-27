"""
Utils function for the transfer learning task
"""

def unfreeze_t5_model_decoder(model, unfreeze_n = 2):
    '''
    Util function to freeze an entire t5 model except for its unfreeze_n last layers
    '''

    # freeze the entire encoder
    for i, m in enumerate(model.encoder.block):        
        for parameter in m.parameters():
            parameter.requires_grad = False

    for i, m in enumerate(model.decoder.block):        
    # Only un-freeze the last n transformer blocks in the decoder
        if i  < 6 - unfreeze_n:
            for parameter in m.parameters():
                parameter.requires_grad = False
        else:
            for parameter in m.parameters():
                parameter.requires_grad = True
    return model