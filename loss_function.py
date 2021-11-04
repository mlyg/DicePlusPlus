from tensorflow.keras import backend as K

# Dice++ loss function for 2D image segmentation
def dice_plus_plus(gamma=2):

    def loss_function(y_true, y_pred):
        # change to axis=[1,2,3] for 3D image segmentation
        axis=[1,2]
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred,epsilon,1-epsilon)
        y_true = K.clip(y_true,epsilon,1-epsilon)
        delta = 0.5
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum((y_true * (1-y_pred))**gamma, axis=axis)
        fp = K.sum(((1-y_true) * y_pred)**gamma, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
        loss = K.mean(1-dice_class)
        return loss

    return loss_function
