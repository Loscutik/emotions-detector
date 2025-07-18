import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, Flatten, Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, Dropout, RandomFlip, RandomRotation
from tensorflow.keras import Model
from tensorflow.keras.initializers import random_uniform, glorot_uniform

def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers: RandomFlip and RandomRotation
    Returns:
        tf.keras.Sequential
    '''
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip(mode = 'horizontal', name='Flip'))
    data_augmentation.add(RandomRotation(factor=0.1,name='Rotation'))
    
    return data_augmentation

class BlockNumerator:
    """
    use this decorator to add a number to blocks of convolutional networks
    """
    def __init__(self):
        self.counter=1
        
    def __call__(self,func):
        def wrapper(*args, **kwargs):
            kwargs['block_number']=self.counter
            self.counter+=1
            return func(*args, **kwargs)
        return wrapper
    
batch_normalization_params=dict(axis=3,epsilon=0.00005) # Default axis

# MobileNet like
blocknumerator = BlockNumerator()

# residual bottleneck block
@blocknumerator
def bottleneck_block(X, f, filters, s = 1,expansion_rate=5, initializer=glorot_uniform, normalization=True, kernel_regularizer=None, dropout_rate=False, block_number=0):
    """
    Implementation of the residual bottleneck block.
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the inner CONV's window for the main path
    filters -- integer, defining the number of filters in the output layers, i.e. number of the output channels 
    s -- Integer, specifying the stride to be used
    expansion_rate -- integer, specifying the ratio between the size of the input bottleneck and the inner size 
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    normalization -- bool, if True, use BatchNormalization after the convolution layers
    kernel_regularizer -- regularizer applied to the kernel weights in Conv layers
    dropout_rate -- integer or bool, defining the rate of the dropout layer. If it is False, Dropout layer will not be applied.
    block_number -- unique integer identifying the block_number to add to the names of the layers
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # number of expanded channels
    expansion_channels = expansion_rate*X.shape[-1] 
    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = expansion_channels, kernel_size = 1, strides = (1,1), padding = 'same', kernel_initializer = initializer(),kernel_regularizer=kernel_regularizer, name=f'expand{block_number}')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name=f'expand_BN{block_number}')(X) 
    if dropout_rate:
        X = Dropout(dropout_rate,name=f'expand_dropout{block_number}')(X)
    X = Activation('relu', name=f'expand_relu{block_number}')(X)
    
    ## Second component of main path
    X = DepthwiseConv2D(kernel_size = f, strides = s, padding = 'same', depthwise_initializer = initializer(), name=f'depthwise{block_number}')(X) # depthwise_regularizer=kernel_regularizer,
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name=f'depthwise_BN{block_number}')(X) # Default axis
    X = Activation('relu', name=f'depthwise_relu{block_number}')(X)

    ## Third component of main path
    X = Conv2D(filters = filters, kernel_size = 1, strides = (1,1), padding = 'same', kernel_initializer = initializer(),kernel_regularizer=kernel_regularizer, name=f'pointwise{block_number}')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name=f'pointwise_BN{block_number}')(X) # Default axis
    if dropout_rate:
        X = Dropout(dropout_rate, name=f'pointwise_dropout{block_number}')(X)
        
    ## Final step: Add shortcut value to main path if the height and width have not changed
    if s==1 and X.shape[-1]==X_shortcut.shape[-1]:
        X = Add()([X, X_shortcut])

    return X

def create_MobNetLike(input_shape, classes, data_augmentation=None, first_conv={}, stages={}, last_expantion=4, dropout_rate=0.1, normalization=True, kernel_regularizer=None, addition_FCs=0, name='mobNet'):
    """
    creates a model of residual network using triple convolutional and identity blocks 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    data_augmentation -- data augmentation model
    first_conv -- dict of kind {"f":<integer>, "filters":<integers>, "s":<integer>}, where
        f -- specifies the shape of the window to be used in the first convolutional layer 
        filters -- defines the number of filters to be used in the first convolutional layer
        strides -- specifies the stride to be used in the first convolutional layer
    stages -- dict of kind {"num_blocks":<list of integer>, "f":<list of integer>, "filters":<list of integers>, "s":<list of integer>, expansion_rate: <list of integer>}, where
        num_blocks -- defines the number of the bottleneck blocks at each stage
        f -- specifies the shape of the window to be used in the depthwise layer
        filters -- filters -- defines the number of filters to be used in the pointwise layer in the block, i.e. number of output channels
        s -- specifies the stride to be used in the depthwise layer
        expansion_rate -- specifies the expansion rate of the bottleneck block
    last_expantion -- integer, specifies the expansion rate of the last convolutional 1x1 layer
    dropout_rate -- integer or bool, defining the rate of the dropout layer. If it is False, Dropout layer will not be applied.
    normalization -- bool, if True, use BatchNormalization after the convolution layers
    kernel_regularizer -- regularizer applied to the kernel weights
    addition_FCs -- defines the number of fully connected layer added before the softmax layer 
    name -- defines the name of the model
    
    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    if data_augmentation is not None:
        X = data_augmentation(X_input)
    else:
        X = X_input

    # Zero-Padding
    #X = ZeroPadding2D((3, 3))(X)
    
    # Stage 0
    X = Conv2D(first_conv['filters'], kernel_size = first_conv['f'], strides = first_conv['s'], padding='same', kernel_initializer = glorot_uniform(), kernel_regularizer=kernel_regularizer, name='initial_conv')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name='initial_BN')(X)
    X = Activation('relu', name='initial_relu')(X)
    
    num_stages = len(stages['num_blocks'])
    for i in range(num_stages):
        X = bottleneck_block(X, f=stages['f'][i], filters = stages['filters'][i], s=stages['s'][i], expansion_rate=stages['expansion_rate'][i], dropout_rate=dropout_rate, normalization=normalization, kernel_regularizer=kernel_regularizer)
        for _ in range(1,stages['num_blocks'][i]):
            X = bottleneck_block(X, f=stages['f'][i], filters = stages['filters'][i], s=1, expansion_rate=stages['expansion_rate'][i], dropout_rate=dropout_rate, normalization=normalization, kernel_regularizer=kernel_regularizer)

    X = Conv2D(last_expantion*X.shape[-1], kernel_size = 1, strides = 1, kernel_initializer = glorot_uniform(), kernel_regularizer=kernel_regularizer, name='last_conv1x1')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name='last_BN')(X)
    if dropout_rate:
        X = Dropout(dropout_rate, name='pointwise_dropout')(X)
    # AVGPOOL
    X = GlobalAveragePooling2D()(X) #default stride=pool_size 
    
    # output layer
    X = Flatten()(X)
    if addition_FCs > 0:
        neurons = X.shape[1] 
        for  i in range(addition_FCs):
            X = Dense(neurons, activation='relu', kernel_initializer = glorot_uniform(), kernel_regularizer=kernel_regularizer, name=f'FC{i}')(X)
            X = Dropout(0.2, name=f'FC{i}_dropout')(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(), name=f'softmax-{classes}')(X)
        
    # Create model
    model = Model(inputs = X_input, outputs = X, name=name)

    return model


# Residual CNN
blocknumerator_residual = BlockNumerator()

# 2-layers blocks
@blocknumerator_residual
def identity_block2(X, f, initializer=random_uniform, normalization=True, kernel_regularizer=None, dropout_rate=False, block_number=0):
    """
    Implementation of a#the convolutional block of the residual network with identity output size to input size.
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the CONV's window for the main path
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    normalization -- bool, if True, use BatchNormalization after the convolution layers
    kernel_regularizer -- regularizer applied to the kernel weights in Conv layers
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    filters = X.shape[-1]
    # First component of main path
    X = Conv2D(filters = filters, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(),kernel_regularizer=kernel_regularizer, name=f'identity_first{block_number}')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name=f'identity_first_BN{block_number}')(X)
    if dropout_rate:
        X = Dropout(dropout_rate,name=f'identity_first_dropout{block_number}')(X)
   # X = Activation('relu', name=f'identity_first_relu{block_number}')(X)
    
    ## Second component of main path
    X = Conv2D(filters = filters, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(),kernel_regularizer=kernel_regularizer, name=f'identity_second{block_number}')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name=f'identity_second_BN{block_number}')(X)
    if dropout_rate:
        X = Dropout(dropout_rate,name=f'identity_second_dropout{block_number}')(X)

    ## Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu',name=f'identity_after_add_relu{block_number}')(X) 

    return X

@blocknumerator_residual
def pooling_block(X, f, filters, s = 2, initializer=glorot_uniform, normalization=True, kernel_regularizer=None, dropout_rate=False, block_number=0):
    """
    Implementation of the convolutional block of the residual network with output size different to input size.
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- integer, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used in the MaxPooling
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer
    normalization -- bool, if True, use BatchNormalization after the convolution layers
    kernel_regularizer -- regularizer applied to the kernel weights in Conv layers
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """
       
    # Save the input value
    X_shortcut = X

    # Poll component of main path glorot_uniform()
    X = MaxPooling2D((s, s), strides=(s, s), name=f'pooling_max_pool{block_number}')(X)
    
    ## First component of main path
    X = Conv2D(filters = filters, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(),kernel_regularizer=kernel_regularizer, name=f'pooling_first{block_number}')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name=f'pooling_first_BN{block_number}')(X)
    if dropout_rate:
        X = Dropout(dropout_rate,name=f'pooling_first_dropout{block_number}')(X)
    # X = Activation('relu', name=f'pooling_first_relu{block_number}')(X)

    ## Second component of main path
    X = Conv2D(filters = filters, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(),kernel_regularizer=kernel_regularizer, name=f'pooling_second{block_number}')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name=f'pooling_second_BN{block_number}')(X)
    if dropout_rate:
        X = Dropout(dropout_rate,name=f'pooling_second_dropout{block_number}')(X)
    
    ##### SHORTCUT PATH ##### 
    X_shortcut = Conv2D(filters = filters, kernel_size = (s, s), strides=(s, s), padding='valid', kernel_initializer = initializer(),kernel_regularizer=kernel_regularizer, name=f'pooling_shortcut{block_number}')(X_shortcut)
    if normalization:
        X_shortcut =BatchNormalization(**batch_normalization_params, name=f'pooling_shortcut_BN{block_number}')(X_shortcut)
    if dropout_rate:
        X_shortcut = Dropout(dropout_rate,name=f'pooling_shortcut_dropout{block_number}')(X_shortcut)
    
    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu', name=f'pooling_after_add_relu{block_number}')(X)
    
    return X
    
def create_ResNet2(input_shape, classes, data_augmentation=None, stages={},last_window_expantion=2, dropout_rate=0.1, normalization=True, kernel_regularizer=None, addition_FCs=0, name='resNet2'):
    """
    creates a model of residual network using double convolutional and identity blocks 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    data_augmentation -- data augmentation model
    stages -- dict of kind {"identity_blocks":<list of integer>, "f":<list of integer>, "filters":<list of integers>, "s":<list of integer>}, where
        identity_blocks -- defines the number of identity_block for each stage
        f -- specifies the shape of the window to be used in convolutional and identical blocks
        filters -- defines the number of filters to be used in the convolutional layers in the block
        s -- specifies the stride to be used in the convolutional block
    last_window_expantion -- integer, specifies the expansion rate of the last convolutional layer, used to flattering the output
    normalization -- bool, if True, use BatchNormalization after the convolution layers
    kernel_regularizer -- regularizer applied to the kernel weights
    addition_FCs -- defines the number of fully connected layer added before the softmax layer 
    name -- defines the name of the model
    
    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    if data_augmentation is not None:
        X = data_augmentation(X_input)
    else:
        X = X_input
    # Zero-Padding
    #X = ZeroPadding2D((3, 3))(X)
    
    # # Stage 0
    # X = Conv2D(stages['filters'][0], kernel_size = stages['f'][0], strides = (stages['s'][0],stages['s'][0]), kernel_initializer = glorot_uniform(), kernel_regularizer=kernel_regularizer)(X)
    # if normalization:
    #     X = BatchNormalization(axis = 3)(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    # for _ in range(stages['identity_blocks'][0]):
    #     X = identity_block2(X, stages['f'][0], normalization=normalization, kernel_regularizer=kernel_regularizer)
    
    num_stages = len(stages['f'])
    for i in range(num_stages):
        X = pooling_block(X, f=stages['f'][i], filters = stages['filters'][i], s = stages['s'][i], dropout_rate=dropout_rate, normalization=normalization, kernel_regularizer=kernel_regularizer)
        for _ in range(stages['identity_blocks'][i]):
            X = identity_block2(X, f=stages['f'][i], dropout_rate=dropout_rate, normalization=normalization, kernel_regularizer=kernel_regularizer)

    #X = AveragePooling2D((2,2))(X) #default stride=pool_size
    X = Conv2D(filters=last_window_expantion*stages['filters'][-1], kernel_size = X.shape[1], strides = 1, padding='valid', kernel_initializer = glorot_uniform(), kernel_regularizer=kernel_regularizer, name='conv_flattering')(X)
    if normalization:
        X = BatchNormalization(**batch_normalization_params, name='flattering_BN')(X)
    # AVGPOOL
    X = GlobalAveragePooling2D()(X) #default stride=pool_size 
    if dropout_rate:
        X = Dropout(dropout_rate, name='flattering_dropout')(X)
    # output layer
    #X = Flatten()(X)
    if addition_FCs > 0:
        neurons = X.shape[1] 
        for  i in range(addition_FCs):
            X = Dense(neurons, activation='relu', kernel_initializer = glorot_uniform(), kernel_regularizer=kernel_regularizer, name=f'FC{i}')(X)
            X = Dropout(0.4, name=f'FC{i}_dropout')(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(), name=f'softmax-{classes}')(X)
        
    # Create model
    model = Model(inputs = X_input, outputs = X, name=name)

    return model    
