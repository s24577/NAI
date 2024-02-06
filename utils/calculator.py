from rouge import Rouge


def calculate_metrics(predicted_language, actual_language):
    rouge = Rouge()
    scores = rouge.get_scores(predicted_language, actual_language)

    return scores[0]['rouge-l']['p'], scores[0]['rouge-l']['r'], scores[0]['rouge-l']['f']
