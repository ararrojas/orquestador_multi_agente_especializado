from agents.requirements import missing_analyses


def _state(*tools):
    return {"contributions": [{"agent": "analyst", "step": 1, "instruction": "", "output": "", "tools_used": list(tools)}]}


def test_sentiment_question_requires_sentiment_tool():
    q = "Compara el sentimiento de las reseñas y el crecimiento porcentual"
    labels = [label for label, _ in missing_analyses(q, _state("growth_rate"))]
    assert labels == ["análisis de sentimiento de reseñas"]
    assert missing_analyses(q, _state("growth_rate", "sentiment_analysis")) == []


def test_question_without_sentiment_has_no_requirement():
    assert missing_analyses("¿Cuántos usuarios hubo en Q4?", _state()) == []
