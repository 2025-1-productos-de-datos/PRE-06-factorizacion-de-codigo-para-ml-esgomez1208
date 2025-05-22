def compare_models(current_model, best_model, x_test, y_test):

    if best_model is None or current_model.score(x_test, y_test) > best_model.score(
        x_test, y_test
    ):
        return current_model

    return current_model
