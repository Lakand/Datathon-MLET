import pytest
from src.utils import calculate_risk_level

@pytest.mark.parametrize("pedra, risco_esperado", [
    ("Ametista", "Baixo"),
    ("Topázio", "Baixo"),
    ("Ágata", "Alto"),
    ("Quartzo", "Alto"),
    ("ametista", "Baixo"), # Testando a robustez do .capitalize()
    ("PEDRA DESCONHECIDA", "Alto") # Testando o caso default (else)
])
def test_regra_negocio_risco(pedra, risco_esperado):
    """
    Testa se a função de cálculo de risco está seguindo
    as regras de negócio definidas pelo Passos Mágicos.
    """
    assert calculate_risk_level(pedra) == risco_esperado