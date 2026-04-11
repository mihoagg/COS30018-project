from sklearn import svm


def build_svm_model(C=1.0, kernel="rbf", gamma="scale"):
    """
    Builds a Support Vector Machine classifier with specified parameters.
    probability=True is required for predict_proba to work in prediction steps.
    """
    return svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
