from enum import Enum
class config:
    #parameters
    class params(Enum):
        vocabulary = 500
        batch_size = 300
        epochs = 30

    #model Parameters
    class model_params(Enum):
        dense_size = 512
        dropout_size = 0.1
        labels = 25
        activation_function = 'relu'
        final_layer_activation = 'softmax'

    #compiler Parameters
    class compiler_params(Enum):
        optimizer =  'adam'
        loss = 'categorical_crossentropy'
