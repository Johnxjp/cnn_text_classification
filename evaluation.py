from sklearn import metrics as sk_metrics


def eval_model(model, x, use_gpu=False):
    """Returns a torch tensor with the predicted classes"""
    model.eval()
    if use_gpu:
        x = x.cuda()
    return model.predict_classes(x)


def accuracy(y_true, y_pred):
    """Computes accuracy using sklearn. Arrays should be numpy arrays"""
    return sk_metrics.accuracy_score(y_true, y_pred)


def f1_score(y_true, y_pred):
    """Computes F1 scores for classes. Arrays should be numpy arrays"""
    return sk_metrics.f1_score(y_true, y_pred)
