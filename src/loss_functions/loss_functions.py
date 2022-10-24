from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#          DSC++ loss          #
################################
def dice_plus_loss(gamma=2):
    """ Dice++ loss function as described in the paper.

    Args:
        gamma (float): controls the degree of penalisation for FN and FP predictions.
                       Higher gamma values favour low confidence predictions.
    """
    def loss_function(y_true, y_pred):
        epsilon = K.epsilon()
        axis = identify_axis(y_true.get_shape())

        y_pred = K.clip(y_pred,epsilon,1-epsilon)
        y_true = K.clip(y_true,epsilon,1-epsilon)

        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum((y_true * (1-y_pred))**gamma, axis=axis)
        fp = K.sum(((1-y_true) * y_pred)**gamma, axis=axis)
        dice_class = (2*tp + epsilon)/(2*tp + fn + fp + epsilon)
        loss = K.mean(1-dice_class)

        return loss

    return loss_function

################################
#      Cross entropy loss      #
################################
def cross_entropy(y_true, y_pred):
    """ Cross entropy loss."""
    axis = identify_axis(y_true.get_shape())
    # Clip values to prevent division by zero error
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * K.log(y_pred)

    cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
    return cross_entropy

################################
#   Dice cross entropy loss    #
################################
def dice_crossentropy(y_true,y_pred):
    """ Sum of the Dice loss and cross entropy loss"""
    dice_loss = dice_plus_loss(1.)(y_true,y_pred)
    crossentropy = K.categorical_crossentropy(y_true, y_pred)
    crossentropy = K.mean(crossentropy)
    return dice_loss + crossentropy
    

################################
#        Tversky++ loss        #
################################
def tversky_plus_plus_loss(gamma):
    """ Tversky loss with DSC++ modification"""
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred,epsilon,1-epsilon)
        y_true = K.clip(y_true,epsilon,1-epsilon)
        delta = 0.7
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum((y_true * (1-y_pred))**gamma, axis=axis)
        fp = K.sum(((1-y_true) * y_pred)**gamma, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
        loss = K.mean(1-dice_class)

        return loss

    return loss_function

################################
#     Focal Tversky++  loss    #
################################
def focal_tversky_plus_plus_loss(gamma):
    """Focal Tversky loss with DSC++ modification"""
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred,epsilon,1-epsilon)
        y_true = K.clip(y_true,epsilon,1-epsilon)
        delta = 0.7
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum((y_true * (1-y_pred))**gamma, axis=axis)
        fp = K.sum(((1-y_true) * y_pred)**gamma, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
        loss = K.mean(K.pow((1-dice_class), 0.75))

        return loss

    return loss_function


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.7, gamma=2.):
    """
    Args: 
        delta (float): controls weight given to false positive and false negatives, by default 0.7
        gamma (float): Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())  

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        #calculate losses separately for each class, only suppressing background class
        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss

    return loss_function

#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75, gamma_plus_plus=1):
    """This is the implementation for binary segmentation.
    Args:
        delta (float): controls weight given to false positive and false negatives, by default 0.7
        gamma (float): focal parameter controls degree of down-weighting of easy examples, by default 0.75
        gamma_plus_plus (float): focal parameter controlling penalty on FN and FP, by default 1
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred)**gamma_plus_plus, axis=axis)
        fp = K.sum(((1-y_true) * y_pred)**gamma_plus_plus, axis=axis)
        tversky_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_tversky = (1-tversky_class[:,0]) 
        fore_tversky = (1-tversky_class[:,1]) * K.pow(1-tversky_class[:,1], -gamma) 

        # Average class scores
        loss = K.mean(tf.stack([back_tversky,fore_tversky],axis=-1))
        return loss

    return loss_function


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5, gamma_plus_plus=1):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Args: 
        weight (float): represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
        delta (float): controls weight given to each class, by default 0.6
        gamma (float): focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
        gamma_plus_plus (float): focal parameter controlling penalty on FN and FP, by default 1
    """
    def loss_function(y_true,y_pred):
      asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma, gamma_plus_plus=gamma_plus_plus)(y_true,y_pred)
      asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      if weight is not None:
        return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)  
      else:
        return asymmetric_ftl + asymmetric_fl

    return loss_function