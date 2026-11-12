from tools.analysis_tools import calculator, growth_rate, sentiment_analysis
from tools.search_tools import search_knowledge_base


def test_kb_search_finds_quarterly_report():
    out = search_knowledge_base.invoke({"query": "usuarios activos ingresos Q4 2024"})
    assert "kb-004" in out


def test_calculator_is_safe():
    assert "= 54.166667" in calculator.invoke({"expression": "(7.4 - 4.8) / 4.8 * 100"})
    assert calculator.invoke({"expression": "__import__('os')"}).startswith("ERROR")


def test_growth_rate():
    out = growth_rate.invoke({"values": [1200000, 1720000], "labels": ["Q1", "Q4"]})
    assert "+43.33%" in out


def test_sentiment_polarity():
    out = sentiment_analysis.invoke({"texts": ["La app es excelente", "Soporte muy lento y frustrante", "No es mala"]})
    lines = out.splitlines()
    assert "positivo" in lines[0] and "negativo" in lines[1] and "positivo" in lines[2]
