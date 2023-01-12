# example
class Config:
    # data
    BATCH_SIZE = 20
    SHUFFLE_BUFFER = 500

    # model
    NUM_MODELS = 100
    NUM_EPOCHS = 100
    NUM_ROUNDS = 100
    SERVER_LEARNING_RATE = 1.0
    CLIENT_LEARNING_RATE = 0.1

    # path
    DATA_PATH = f'../../../datasets/CPDP_datasets/'
    SAVE_PATH = f'../../result/'